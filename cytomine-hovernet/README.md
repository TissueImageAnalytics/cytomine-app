# cytomine-hovernet

[Cytomine](https://cytomine.org) app for nucleus instance segmentation and classification, developed by the [TIA Centre](https://warwick.ac.uk/fac/cross_fac/tia/).

The app applies HoVer-Net via [TIAToolbox](https://github.com/TissueImageAnalytics/tiatoolbox), using the `hovernet_fast-pannuke` checkpoint pre-trained on the [PanNuke dataset](https://jgamper.github.io/PanNukeDataset/). It produces per-nucleus polygon geometries classified into five types: neoplastic epithelial, inflammatory, connective, dead, and non-neoplastic epithelial.

## Reference

Graham, Simon, et al. "HoVer-Net: Simultaneous segmentation and classification of nuclei in multi-tissue histology images." *Medical Image Analysis* 58 (2019): 101563.

## Pre-trained weights

Download [hovernet_fast-pannuke.pth](https://huggingface.co/TIACentre/TIAToolbox_pretrained_weights/resolve/main/hovernet_fast-pannuke.pth?download=true) from the [TIA Centre HuggingFace](https://huggingface.co/TIACentre) and place it in `models/`.
