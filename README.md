# isbiANHIR

This project is a challange which about Automatic Non-rigid Histological Image Registration (ANHIR). And this is a part of the IEEE International Symposium on Biomedical Imaging (ISBI) 2019.


Image registration is a image processing technique which can be used to align two or more images in to a single scene. The visual comparison of successive tissue slices that align multiple images to a common frame is one of the simplest but the most useful features in digital pathology(<https://anhir.grand-challenge.org/Home/>). Image registration give the possiblility for pathologist to assess the histology and expression of multiple markers in a patient in a single region.
This project focus on the registration accuracy and speed of the registration algorithm which automatic registrate a set of large images from the same tissue samples but stained with different biomarkers.

## Dataset:
The Dataset is downloaded from: https://anhir.grand-challenge.org/Download/.
Which contains a set of images and a landmarks files

## Evaluation:
The evaluate function is based on the rTRE for each pair of landmarks in registered images pairs which is the competition criteria function.
