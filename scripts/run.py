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
import numpy as np
import sys
import os
import shutil
import torch
from pathlib import Path
from shapely.geometry import Polygon

from tiatoolbox.models.engine.multi_task_segmentor import MultiTaskSegmentor

from typing import Any, Callable, Iterable
import geojson
import zarr
import yaml
from imageio import imread
import imghdr

__author__ = "Gozde Gunesli <gozde.gunesli@warwick.ac.uk> and Mostafa Jahanifar <mostafa.jahanifar@warwick.ac.uk> and Adam Shephard <adam.shephard@warwick.ac.uk>"

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


def to_geojson_string(poly_coords):
    """Convert a list of (x, y) coordinate tuples to a GeoJSON Polygon string."""
    return geojson.dumps(geojson.Polygon(poly_coords))


def contour_output_to_valid_poly_coords(contour, minx, miny):
    """Fix contour coordinates - adjust coordinate system & convert to to a valid polygon geometry."""

    if len(contour) < 3:
        return None
    
    # Cytomine cartesian coordinate system, (0,0) is bottom left co
    coords = [(float(minx + c[0]), float(miny - c[1])) for c in contour]

    if coords[0] != coords[-1]:
        coords.append(coords[0])  # ensure ring is closed
    
    poly = Polygon(coords)

    if not poly.is_valid:
        poly = poly.buffer(0) 
        logger.info("Polygon fixed with buffer(0).")

    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda p: p.area)  # keep largest fragment
        logger.info("MultiPolygon fixed.")

    if not poly.is_valid or poly.is_empty:
        logger.info("Polygon could not be fixed and will be skipped.")
        poly = None

    if poly:
        return list(poly.exterior.coords)
    else:
        return None


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # HoVerNet model selection
    hovernet_model = read_parameter(os.path.join(INPUT_DIR, "hovernet_model"), cast_fn=str, default="hovernet_fast-pannuke")
    assert hovernet_model in ["hovernet_fast-pannuke", "hovernet_fast-monusac", "hovernet_original-consep", "hovernet_original_kumar"], f"Unsupported HoVerNet model: {hovernet_model}"
    hovernet_weights_path = f"{MODEL_DATA_DIR}/{hovernet_model}.pth"

    # get input image info
    src_image_path = os.path.join(INPUT_DIR, "image")
    image = imread(src_image_path)
    image_height = image.shape[0]
    logger.info(f"Input image height: {image_height}")
    file_ext = "." + (imghdr.what(src_image_path) or "png")
    logger.info(f"Input image file extension: {file_ext}")

    # rename & copy image to cache dir for processing
    image_path = os.path.join(CACHE_DIR, "input_image" + file_ext)
    shutil.copyfile(src_image_path, image_path)

    # Cell ID dict
    CELL_ID_DICT = {  # 0: "background" is ignored
        1: "nuclei_neo",
        2: "nuclei_inf",
        3: "nuclei_con",
        4: "nuclei_dead",
        5: "nuclei_nnepi",
    }

    # Set device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Use TIAToolbox MultiTaskSegmentor for HoVerNet model
    segmentor = MultiTaskSegmentor(
        model=hovernet_model,
        weights=hovernet_weights_path,
        num_workers=0,
        batch_size=2,
        device=device,
    )

    # Remove previous cache results if exists
    save_dir = os.path.join(CACHE_DIR, "hovernet_results")
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)

    output = segmentor.run(
        images=[image_path],
        save_dir=save_dir,
        patch_mode=False, 
        output_type="zarr",
        overwrite=True,
        wsireader_kwargs={"mpp": 0.25},  # some (PNG) has no MPP metadata, HoVerNet models train at 0.25 mpp (40x)
    )

    # Open zarr store (one image processed, take the single result)
    store_path = list(output.values())[0] #output[image_path]
    nuc_seg = zarr.open(store_path, mode="r")
    logger.info(f"Output keys: {list(nuc_seg.keys())}")  

    n_nuclei = len(nuc_seg["type"])
    logger.info(f"Number of detected nuclei: {n_nuclei}")


    # Collect coordinates per cell type and adjust to Cytomine coordinate system
    # Cytomine uses cartesian coords with (0,0) at bottom-left, so y is flipped
    minx, miny = 0, image_height
    coordinates = {t: [] for t in CELL_ID_DICT.values()}

    for i in range(n_nuclei):
        nuc_type = int(nuc_seg["type"][i])
        if nuc_type in CELL_ID_DICT:
            contour = np.array(nuc_seg["contours"][i])  # shape [N_pts, 2], columns are [x, y]
            points = contour_output_to_valid_poly_coords(contour, minx, miny)
            if points:
                coordinates[CELL_ID_DICT[nuc_type]].append(points)

    # Write outputs per cell type
    for cell_type, points in coordinates.items():
        write_array(
            array_path=os.path.join(OUTPUT_DIR, cell_type),
            array_data=points,
            format_fn=to_geojson_string,
        )


if __name__ == "__main__":
    main()
