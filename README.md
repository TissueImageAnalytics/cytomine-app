# TIA Centre Cytomine Apps

[Cytomine](https://cytomine.org) apps developed by the [TIA Centre](https://warwick.ac.uk/fac/cross_fac/tia/), University of Warwick.

This repository contains two apps, both wrapping pre-trained models from [TIAToolbox](https://github.com/TissueImageAnalytics/tiatoolbox):

- **[cytomine-hovernet](cytomine-hovernet/)** — nucleus instance segmentation and classification with HoVer-Net, trained on the PanNuke dataset.
- **[cytomine-kongnet](cytomine-kongnet/)** — nucleus detection with KongNet, trained on the MONKEY Challenge dataset.

See each app's README for model and reference details.

## Pre-trained model weights

Weights are hosted on the [TIA Centre HuggingFace](https://huggingface.co/TIACentre):

- HoVer-Net (PanNuke): [hovernet_fast-pannuke.pth](https://huggingface.co/TIACentre/TIAToolbox_pretrained_weights/resolve/main/hovernet_fast-pannuke.pth?download=true)
- KongNet (MONKEY): [KongNet_MONKEY_1.pth](https://huggingface.co/TIACentre/KongNet_pretrained_weights/resolve/main/KongNet_MONKEY_1.pth?download=true)
