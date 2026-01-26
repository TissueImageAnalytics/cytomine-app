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

import numpy as np
import sys
import os
import shutil
import torch 
import joblib
from shapely import wkt

# sys.path.append('/home/adams/Projects/tiatoolbox_local/tiatoolbox/')
from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor

from typing import Any, Callable, Iterable
import geojson
import yaml
from imageio import imread
import imghdr

__author__ = "Gozde Gunesli <gozde.gunesli@warwick.ac.uk> and Mostafa Jahanifar <mostafa.jahanifar@warwick.ac.uk> and Adam Shephard <adam.shephard@warwick.ac.uk>"

INPUT_DIR = "/inputs"
OUTPUT_DIR = "/outputs"
MODEL_DATA_DIR = "/models"
CACHE_DIR = "/temp" # to save temp files e.g., model outputs and input copy



def read_parameter(path: str, cast_fn: Callable[[str], Any], default: Any=None, raise_if_missing: bool=False):
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


# Convert coordinates to geojson Polygon string
def to_geojson_string(coords):
    # Close the polygon
    coords.append(coords[0])
    return geojson.dumps(geojson.Polygon([coords], validate=True)) 


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
    print(f"Input image height: {image_height}")
    file_ext = "." + (imghdr.what(src_image_path) or "png")
    print(f"Input image file extension: {file_ext}")

    # rename & copy image to cache dir for processing
    image_path = os.path.join(CACHE_DIR, "input_image"+file_ext)
    shutil.copyfile(src_image_path, image_path) 

    # Use TIAToolbox nucleus instance segmentor engine for HoVerNet model
    inst_segmentor = NucleusInstanceSegmentor(
        pretrained_model=hovernet_model,
        pretrained_weights=hovernet_weights_path,
        num_loader_workers=0,
        num_postproc_workers=0,
        batch_size=2,
    )

    # Cell ID dict
    CELL_ID_DICT={  #0: "background", # ignored
                    1: "nuclei_neo",
                    2: "nuclei_inf",
                    3: "nuclei_con",
                    4: "nuclei_dead",
                    5: "nuclei_nnepi" }


    # Set device  
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Remove previous cache results if exists
    save_dir = os.path.join(CACHE_DIR, "hovernet_results")
    if os.path.isdir(save_dir):
      shutil.rmtree(save_dir)

    tile_output = inst_segmentor.predict(
        [image_path],
        save_dir=save_dir,
        mode="tile",
        device=device,
        crash_on_exception=True,
    )
    
    tile_preds = joblib.load(f"{tile_output[0][1]}.dat")
    print("Number of detected nuclei: %d" % len(tile_preds))
                               
    # Go over detections, convert and save seperately for Cytomine
    minx, miny = 0, image_height  # since image is processed as single ROI

    # Collect coordinates per cell type and adjust to Cytomine coordinate system
    coordinates = {type:[] for type in CELL_ID_DICT.values()}
    for nucleus in tile_preds:
        contours = tile_preds[nucleus]['contour']
        nuc_type = tile_preds[nucleus]['type']
        if nuc_type in CELL_ID_DICT.keys(): # skip background "0"
          coordinates[CELL_ID_DICT[nuc_type]].append([[float(minx + point[0]), float(miny - point[1])] for point in contours]) # Flip Y coordinates for Cytomine, ensure float type for geojson compatibility
    
    # Write outputs per cell type
    for cell_type, coords in coordinates.items():
        write_array(
            array_path=os.path.join(OUTPUT_DIR, cell_type),
            array_data=coords,
            format_fn=to_geojson_string
        )


if __name__ == "__main__":
    main()
