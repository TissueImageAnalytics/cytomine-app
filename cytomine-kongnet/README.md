# cytomine-kongnet

[Cytomine](https://cytomine.org) app for nucleus detection, developed by the [TIA Centre](https://warwick.ac.uk/fac/cross_fac/tia/).

The app applies KongNet model via [TIAToolbox](https://github.com/TissueImageAnalytics/tiatoolbox), using the `KongNet_MONKEY_1` checkpoint pre-trained on the [MONKEY Challenge](https://monkey.grand-challenge.org/) dataset (PAS-stained kidney biopsies). The model has three output types (`Overall_Inflammatory`, `Lymphocyte`, and `Monocyte`). The app uses the combined `Overall_Inflammatory` class type as outputs.

KongNet [1] is a multi-head encoder–decoder architecture with an EfficientNetV2-L encoder, designed for nuclei detection and classification in whole slide images. It has achieved strong benchmark results:

- 1st on Track 1 and 2nd on Track 2 of the MONKEY Challenge [2].
- 1st place in the 2025 MIDOG Challenge [3].
- Top three in the PUMA Challenge [4].
- State-of-the-art detection performance on the PanNuke [5] and CoNIC [6] datasets.

## References

[1] J. Lv et al., "KongNet: A Multi-headed Deep Learning Model for Detection and Classification of Nuclei in Histopathology Images," [arXiv:2510.23559](https://arxiv.org/abs/2510.23559), 2025.

[2] L. Studer, "Structured description of the MONKEY challenge," Sept. 2024.

[3] J. Ammeling, M. Aubreville, S. Banerjee, C. A. Bertram, K. Breininger, D. Hirling, P. Horvath, N. Stathonikos, and M. Veta, "Mitosis domain generalization challenge 2025," Mar. 2025.

[4] M. Schuiveling, H. Liu, D. Eek, G. Breimer, K. Suijkerbuijk, W. Blokx, and M. Veta, "A novel dataset for nuclei and tissue segmentation in melanoma with baseline nuclei segmentation and tissue segmentation benchmarks," *GigaScience*, vol. 14, Jan. 2025.

[5] J. Gamper, N. A. Koohbanani, K. Benes, S. Graham, M. Jahanifar, S. A. Khurram, A. Azam, K. Hewitt, and N. Rajpoot, "PanNuke dataset extension, insights and baselines," 2020.

[6] S. Graham et al., "CoNIC Challenge: Pushing the frontiers of nuclear detection, segmentation, classification and counting," *Medical Image Analysis*, vol. 92, p. 103047, 2024.

## Pre-trained weights

Download [KongNet_MONKEY_1.pth](https://huggingface.co/TIACentre/KongNet_pretrained_weights/resolve/main/KongNet_MONKEY_1.pth?download=true) from the [TIA Centre HuggingFace](https://huggingface.co/TIACentre) and place it in `models/`.
