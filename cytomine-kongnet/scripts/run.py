# -*- coding: utf-8 -*-

# * Copyright (c) 2009-2022. Authors: see NOTICE file.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import sys
import os
import shutil
import torch

from tiatoolbox.models.engine.nucleus_detector import NucleusDetector

from typing import Any, Callable, Iterable
import geojson
import zarr
import yaml
from imageio import imread
import imghdr

__author__ = "Gozde Gunesli <gozde.gunesli@warwick.ac.uk>"

logger = logging.getLogger(__name__)

INPUT_DIR = "/inputs"
OUTPUT_DIR = "/outputs"
MODEL_DATA_DIR = "/models"
CACHE_DIR = "/temp"  # to save temp files e.g., model outputs and input copy


def read_parameter(path: str, cast_fn: Callable[[str], Any], default: Any = None, raise_if_missing: bool = False):
    """
    Read a parameter from a file structured following App Engine conventions.

    Args:
        path (str): The path to the parameter file.
        cast_fn (Callable[[str], Any]): A function to cast the content of the file to the desired type.
        default (Any, optional): The default value to return if the file is not found. Defaults to None.
        raise_if_missing (bool, optional): Whether to raise a FileNotFoundError if the file is not found.
            If set to False, the default value will be returned. Defaults to False.

    Returns:
        Any: The parameter value casted to the desired type.

    Raises:
        FileNotFoundError: If the file is not found and raise_if_missing is set to True.
    """
    if not os.path.isfile(path):
        if raise_if_missing:
            raise FileNotFoundError(f"could not find parameter file '{path}'")
        else:
            return default
    with open(path, "r", encoding="utf8") as file:
        content = file.read()
        return cast_fn(content)


def write_array(array_path: str, array_data: Iterable[Any], format_fn: Callable[[Any], str]):
    """
    Write an array of data to files following App Engine conventions.

    Parameters:
        array_path (str): The path to the directory where the array files will be written.
        array_data (Iterable[Any]): The array of data to be written.
        format_fn (Callable[[Any], str]): A function that formats each data item before writing them into a file.

    Returns:
        None
    """
    os.makedirs(array_path, exist_ok=True)
    # writing array metadata
    with open(os.path.join(array_path, "array.yml"), "w+", encoding="utf8") as file:
        yaml.dump({"size": len(array_data)}, file)
    # writing array data content
    for i, data_item in enumerate(array_data):
        with open(os.path.join(array_path, f"{i}"), "w+", encoding="utf8") as file:
            file.write(format_fn(data_item))


def to_geojson_point_string(xy):
    """Convert an (x, y) coordinate pair to a GeoJSON Point string."""
    return geojson.dumps(geojson.Point((float(xy[0]), float(xy[1]))))


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # KongNet model and weights path
    kongnet_model = "KongNet_MONKEY_1"
    kongnet_weights_path = f"{MODEL_DATA_DIR}/{kongnet_model}.pth"

    # get input image info
    src_image_path = os.path.join(INPUT_DIR, "image", "0")  # index 0 of the array input
    image = imread(src_image_path)
    image_height = image.shape[0]
    logger.info(f"Input image height: {image_height}")
    file_ext = "." + (imghdr.what(src_image_path) or "png")
    logger.info(f"Input image file extension: {file_ext}")

    # rename & copy image to cache dir for processing
    image_path = os.path.join(CACHE_DIR, "input_image" + file_ext)
    shutil.copyfile(src_image_path, image_path)

    # Cell ID dict (KongNet_MONKEY_1 class_dict from tiatoolbox pretrained registry:
    # 0: Overall_Inflammatory, 1: Lymphocyte, 2: Monocyte).
    # Only class 0 (the combined inflammatory superset) is used.
    CELL_ID_DICT = {
        0: "inflammatory_cells",
    }

    # Set device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Use TIAToolbox NucleusDetector for KongNet model
    detector = NucleusDetector(
        model=kongnet_model,
        weights=kongnet_weights_path,
        num_workers=0,
        batch_size=8,
        device=device,
    )

    # Remove previous cache results if exists
    save_dir = os.path.join(CACHE_DIR, "kongnet_results")
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)

    output = detector.run(
        images=[image_path],
        save_dir=save_dir,
        patch_mode=False,
        output_type="zarr",
        overwrite=True,
        wsireader_kwargs={"mpp": 0.25},  # some (PNG) has no MPP metadata, KongNet models train at 0.25 mpp (40x)
    )

    # Open zarr store (one image processed, take the single result)
    # KongNet NucleusDetector stores per-detection: classes, x, y (parallel 1-D arrays)
    store_path = list(output.values())[0]
    nuc_det = zarr.open(store_path, mode="r")
    logger.info(f"Output keys: {list(nuc_det.keys())}")

    classes = nuc_det["classes"]
    xs = nuc_det["x"]
    ys = nuc_det["y"]
    n_nuclei = len(classes)
    logger.info(f"Number of detected nuclei: {n_nuclei}")

    # Collect coordinates per cell type and adjust to Cytomine coordinate system
    # Cytomine uses cartesian coords with (0,0) at bottom-left, so y is flipped
    coordinates = {t: [] for t in CELL_ID_DICT.values()}

    for i in range(n_nuclei):
        nuc_type = int(classes[i])
        if nuc_type in CELL_ID_DICT:
            x_cyt = float(xs[i])
            y_cyt = float(image_height - ys[i])
            coordinates[CELL_ID_DICT[nuc_type]].append((x_cyt, y_cyt))

    # Write outputs per cell type
    for cell_type, points in coordinates.items():
        write_array(
            array_path=os.path.join(OUTPUT_DIR, cell_type),
            array_data=points,
            format_fn=to_geojson_point_string,
        )


if __name__ == "__main__":
    main()
