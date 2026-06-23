# cytomine-interactive-segmentation-nuclick

[Cytomine](https://cytomine.org) app for interactive nucleus segmentation, developed by the [TIA Centre](https://warwick.ac.uk/fac/cross_fac/tia/).

The app applies NuClick via [TIAToolbox](https://github.com/TissueImageAnalytics/tiatoolbox), using the `nuclick_light-pannuke` checkpoint (a lightweight UNet variant trained on the [PanNuke dataset](https://jgamper.github.io/PanNukeDataset/)). Given a point click on a nucleus, it returns a polygon geometry outlining that nucleus. The input is a single `Point` (one nucleus per run).

## Reference

Alemi Koohbanani, Navid, et al. "NuClick: A deep learning framework for interactive segmentation of microscopic images." *Medical Image Analysis* 65 (2020): 101771.

## Pre-trained weights

Download [nuclick_light-pannuke.pth](https://huggingface.co/TIACentre/TIAToolbox_pretrained_weights/resolve/main/nuclick_light-pannuke.pth?download=true) from the [TIA Centre HuggingFace](https://huggingface.co/TIACentre) and place it in `models/`.
