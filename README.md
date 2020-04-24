# Respiratory-Sound-Classification
Jupyter notbook demo code and workflow for Respiratory sound classification.

## Abstract
This repository uses experimental code for detecting respiratory diseases using audio recordings. Two methods have been used in the experiments

* Generate MFCC coefficients from the audio samples and training a simple hand coded CNN.
* Generate power-melspectrograms from the audio samples and train usual deep learning image recognition models (we use vgg19) assuming them as images.

We don't do any preprocessing at this stage as this is just a proof of concept code, but several methods for audio augmentation like time stretching, amplitude suppression, white noise addition etc can be introduced.

## Dataset
https://www.kaggle.com/vbookshelf/respiratory-sound-database is used as our dataset.