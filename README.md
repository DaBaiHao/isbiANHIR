# Automatic Non-rigid Histological Image Registration
<p align="right">Name: Hao Bai<br/>
ID: 180223545</p>
## Introduction
This project is a challange which about Automatic Non-rigid Histological Image Registration (ANHIR). And this is a part of the IEEE International Symposium on Biomedical Imaging (ISBI) 2019.

Image registration is a image processing technique which can be used to align two or more images in to a single scene. The visual comparison of successive tissue slices that align multiple images to a common frame is one of the simplest but the most useful features in digital pathology. Image registration give the possiblility for pathologist to assess the histology and expression of multiple markers in a patient in a single region.
This project focus on the registration accuracy and speed of the registration algorithm which automatic registrate a set of large images from the same tissue samples but stained with different biomarkers.

## Dataset and ethical use of data:
The Dataset is downloaded from [dataset webpage](https://anhir.grand-challenge.org/Download/) which offerd by IEEE International Symposium on Biomedical Imaging (ISBI) 2019 contains a set of images and a landmarks files. More detailly information about the provider of each image can be found [Dataset Information](https://anhir.grand-challenge.org/Dataset/). This dataset will be only used in this challange and never be used in commercial. The dataset licence [CC-BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/2.0/)

## Bibliography must be followed:
 - [Benchmarking of Image Registration Methods for Differently Stained Histological Slides](https://ieeexplore.ieee.org/document/8451040) Borovec J, Munoz-Barrutia A, Kybic J.

 - [independent segmentation of whole slide images: A case study in renal histology](https://ieeexplore.ieee.org/document/8363824) Gupta L, Klinkhammer BM, Boor P, Merhof D, Gadermayr M. Stain

 - [AIDPATH: Academia and Industry Collaboration for Digital Pathology](http://aidpath.eu/?page_id=279) Bueno G., Deniz O.

## Evaluation:
The evaluate function is based on the rTRE for each pair of landmarks in registered images pairs which is the competition criteria function. Detailly algorithm can be founded in [Evaluation webpage](https://anhir.grand-challenge.org/Evaluation/).

## deisgened Soluation:
There designed soluation is to use deep learning and ConvNets to solve the probelm which is a kind of supervised learning. However, the un-supervised learning also can be used in Image Registration.

## Current state
 - To get familiar with the ConvNets used in image processing, the top soluation of the challange [cat vs dog] on kaggle is being followed to learn the keras using and tensorflow using. The [code](https://github.com/DaBaiHao/CatvsDog/tree/master/catvsdog), [training result](https://github.com/DaBaiHao/CatvsDog/blob/master/train/first_train.txt) and [training model](https://github.com/DaBaiHao/CatvsDog/tree/master/catvsdog/logs) can be find in my [github page](https://github.com/DaBaiHao/CatvsDog).
 - The Bibliography paper is currently being read this days.
 - The next step is write a simple CNN promgram for [kaggle histopathologic cancer detection competitions](https://www.kaggle.com/c/histopathologic-cancer-detection). Because the this competitions is samilar to the cat and dog competitions, and the training data set is easy to use.
