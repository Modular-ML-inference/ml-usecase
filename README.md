# ML use case

This repository contains ML models, test data, and model support files from the fall detection and scratch detection use cases.

## Fall detection

### Contents

- model: fall detection model in TensorflowLite format
- model support: mean and std calculated from the train data for X, Y, Z, VECTOR_LENGTH (Signal Vector Magnitude) values for data normalization purposes
- data: test data from one accelerometer recorded for 11 hours with the 2Hz frequency and reported in tag's units in the range [-2048, 2047], the data format is specified below

### Test data format

Each row in the test data represents one second of observations. Due to the 2Hz frequency, each second has two sets of acceleration values reported by the sensor. 
Therefore, columns have the naming format of `t{obs_number}_{axis}`

- t1_x - axis X, 1st observation in the second
- t1_y - axis Y, 1st observation in the second
- t1_z - axis Z, 1st observation in the second
- t2_x - axis X, 2nd observation in the second
- t2_y - axis Y, 2nd observation in the second
- t2_z - axis Z, 2nd observation in the second

Test data does not include the vector length, it is computed additionally during pre-processing. 

## Scratch detection

### Contents

- data: images_for_inference.txt - list of image names that were used for experiments 
- model_architecture.py - creates Mask R-CNN model based on the pre-trained weights from [PyTorch Torchvision MaskRCNN_ResNet50_FPN_V2_Weights](https://pytorch.org/vision/main/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn_v2.html) with architecture changes applied in the use case. Note: architecture modifications were tailored to the use case training dataset and may require additional tuning in the presence of a different dataset used for experiments
- create_dataset.ipynb - jupyter notebook that allows for creating an image dataset with the same probability distribution of the number of detected scratches as the original evaluation dataset from the use case. Requires the damage detection dataset and trained Mask R-CNN model for better reproducibility. Returns the list of image names for inference purposes. Note: contains the confidence threshold and image size parameters. To achieve full reproducibility both parameters should be equal to those used in the inference pipeline
- processing_inference.py - a class devoted to pre-/post-processing activities such as image resizing, confidence filtering, and output formatting

## Licensing
The repository is released under the Apache 2.0 license. See [LICENSE](LICENSE) for more information.