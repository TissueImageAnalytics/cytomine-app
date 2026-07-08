# -*- coding: utf-8 -*-

# * Copyright (c) 2009-2026. Authors: see NOTICE file.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os
from typing import Any, Callable, Iterable

import cv2
import geojson
import numpy as np
import torch
import yaml
from imageio import imread
from skimage.morphology import remove_small_holes, remove_small_objects

from tiatoolbox.models.architecture import get_pretrained_model

__author__ = "Gozde Gunesli <gozde.gunesli@warwick.ac.uk>"

logger = logging.getLogger(__name__)

INPUT_DIR = "/inputs"
OUTPUT_DIR = "/outputs"
MODEL_DATA_DIR = "/models"
CACHE_DIR = "/temp"

NUCLICK_MODEL = "nuclick_light-pannuke"
PATCH_SIZE = 128
HALF_PATCH = PATCH_SIZE // 2
THRESH = 0.33
MIN_OBJECT_SIZE = 10
MIN_HOLE_SIZE = 30
BATCH_SIZE = 8


def write_array(array_path: str, array_data: Iterable[Any], format_fn: Callable[[Any], str]):
    """Write an array of data to files following App Engine conventions."""
    os.makedirs(array_path, exist_ok=True)
    with open(os.path.join(array_path, "array.yml"), "w+", encoding="utf8") as file:
        yaml.dump({"size": len(array_data)}, file)
    for i, data_item in enumerate(array_data):
        with open(os.path.join(array_path, f"{i}"), "w+", encoding="utf8") as file:
            file.write(format_fn(data_item))


def read_point_array(array_path: str):
    """Read an App Engine geometry-array input: array.yml + indexed GeoJSON items.

    Each item is a Point; returns a list of (x, y) in Cytomine space (bottom-left
    origin).
    """
    with open(os.path.join(array_path, "array.yml"), "r", encoding="utf8") as file:
        n = yaml.safe_load(file)["size"]
    points = []
    for i in range(n):
        with open(os.path.join(array_path, str(i)), "r", encoding="utf8") as file:
            geom = geojson.loads(file.read())
        if geom["type"] != "Point":
            raise ValueError(f"Expected Point geometry, got {geom['type']}")
        points.append(tuple(map(float, geom["coordinates"])))
    return points


def to_geojson_polygon_string(poly_coords):
    return geojson.dumps(geojson.Polygon([poly_coords]))


def build_input_patch(image_padded: np.ndarray, x: int, y: int, other_points_img):
    """Build the 5-channel NuClick input for a click at original-image (x, y).

    `image_padded` is the image reflect-padded by HALF_PATCH, so the patch whose
    centre is the click is image_padded[y:y+PATCH_SIZE, x:x+PATCH_SIZE] and the
    click always sits at the patch centre. Channels: 0-2 RGB, 3 inclusion (this
    click), 4 exclusion (other clicks falling in the patch, used as "not this
    nucleus" context to separate touching nuclei).

    Values stay in [0, 255]: the model's own forward divides the whole input by
    255, so RGB -> [0, 1] and the click maps -> 1. Pre-scaling here would shrink
    the click signal to ~1/255 and the model would see no click.
    """
    patch = image_padded[y:y + PATCH_SIZE, x:x + PATCH_SIZE].astype(np.float32)

    pos_map = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    pos_map[HALF_PATCH, HALF_PATCH] = 255.0

    neg_map = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    for ox, oy in other_points_img:
        opx, opy = ox - x + HALF_PATCH, oy - y + HALF_PATCH
        if 0 <= opx < PATCH_SIZE and 0 <= opy < PATCH_SIZE:
            neg_map[opy, opx] = 255.0

    inp = np.concatenate([patch, pos_map[..., None], neg_map[..., None]], axis=-1)
    return inp.transpose(2, 0, 1)


def mask_to_polygon_coords(mask: np.ndarray, x: int, y: int, image_height: int):
    """Convert a 128x128 binary mask to a closed polygon in Cytomine coordinates.

    Keeps only the connected component touching the patch centre (the click) and
    translates its contour back into image coords using the patch top-left
    (x - HALF_PATCH, y - HALF_PATCH), then applies the Cytomine y-flip. Returns
    None if no usable polygon could be extracted.
    """
    mask = mask.astype(np.uint8)
    if not mask[HALF_PATCH, HALF_PATCH]:
        return None

    _, labels = cv2.connectedComponents(mask)
    component = labels[HALF_PATCH, HALF_PATCH]
    if component == 0:
        return None
    instance_mask = (labels == component).astype(np.uint8)

    contours, _ = cv2.findContours(instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:
        return None

    x0, y0 = x - HALF_PATCH, y - HALF_PATCH
    coords = [
        (float(c[0][0] + x0), float(image_height - (c[0][1] + y0)))
        for c in contour
    ]
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    return coords


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    src_image_path = os.path.join(INPUT_DIR, "image")
    image = imread(src_image_path)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.shape[2] == 4:
        image = image[:, :, :3]
    if image.dtype != np.uint8:
        # 16-bit TIFFs etc. would otherwise break the /255 normalisation below.
        image = (image.astype(np.float32) * (255.0 / image.max())).clip(0, 255).astype(np.uint8)
    image_height, image_width = image.shape[:2]
    logger.info(f"Input image: shape={image.shape}, dtype={image.dtype}")

    points_cyt = read_point_array(os.path.join(INPUT_DIR, "points"))
    logger.info(f"Number of click points: {len(points_cyt)}")

    # Cytomine bottom-left origin -> image top-left origin, snapped to int pixels.
    points_img = []
    for x, y in points_cyt:
        xi, yi = int(round(x)), int(round(image_height - y))
        if 0 <= xi < image_width and 0 <= yi < image_height:
            points_img.append((xi, yi))
        else:
            logger.warning(f"Click ({x:.1f}, {y:.1f}) lies outside the image — skipping.")
    if not points_img:
        write_array(os.path.join(OUTPUT_DIR, "nuclei"), [], to_geojson_polygon_string)
        return

    image_padded = np.pad(
        image,
        ((HALF_PATCH, HALF_PATCH), (HALF_PATCH, HALF_PATCH), (0, 0)),
        mode="reflect",
    )

    inputs = [
        build_input_patch(image_padded, x, y, [p for j, p in enumerate(points_img) if j != i])
        for i, (x, y) in enumerate(points_img)
    ]
    batch = torch.from_numpy(np.stack(inputs)).float()  # (N, 5, 128, 128)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    weights_path = os.path.join(MODEL_DATA_DIR, f"{NUCLICK_MODEL}.pth")
    model, _ = get_pretrained_model(NUCLICK_MODEL, pretrained_weights=weights_path)
    model.to(device)

    # Run in chunks to bound memory. Manual forward + sigmoid (model has 1 output
    # channel; tiatoolbox's UNetModel.infer_batch assumes NHWC input and softmax,
    # which is wrong here).
    model.eval()
    preds_chunks = []
    with torch.inference_mode():
        for start in range(0, len(batch), BATCH_SIZE):
            chunk = batch[start:start + BATCH_SIZE].to(device)
            logits = model(chunk)
            preds_chunks.append(torch.sigmoid(logits).squeeze(1).cpu().numpy())
    preds = np.concatenate(preds_chunks, axis=0)  # (N, 128, 128)

    masks = preds > THRESH
    masks = remove_small_objects(masks, min_size=MIN_OBJECT_SIZE)
    masks = remove_small_holes(masks, area_threshold=MIN_HOLE_SIZE)

    polygons = []
    for (x, y), mask in zip(points_img, masks):
        coords = mask_to_polygon_coords(mask, x, y, image_height)
        if coords is not None:
            polygons.append(coords)
    logger.info(f"Produced {len(polygons)} polygons from {len(points_img)} clicks")

    write_array(
        array_path=os.path.join(OUTPUT_DIR, "nuclei"),
        array_data=polygons,
        format_fn=to_geojson_polygon_string,
    )


if __name__ == "__main__":
    main()
