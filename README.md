# ML use case

This repository contains ML models, test data, and model support files from the fall detection and scratch detection use cases described in the paper ["Flexible Deployment of Machine Learning Inference Pipelines in the Cloud–Edge–IoT Continuum"](https://www.mdpi.com/2079-9292/13/10/1888).

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

## Citation

If you found the ml-usecase code useful, please consider starring ⭐ us on GitHub and citing 📚 us in your research!

```
Bogacka, K.; Sowiński, P.; Danilenka, A.; Biot, F.M.; Wasielewska-Michniewska, K.; Ganzha, M.; Paprzycki, M.; Palau, C.E.
Flexible Deployment of Machine Learning Inference Pipelines in the Cloud–Edge–IoT Continuum.
Electronics 2024, 13, 1888. https://doi.org/10.3390/electronics13101888 
```

```bibtex
@Article{electronics13101888,
AUTHOR = {Bogacka, Karolina and Sowiński, Piotr and Danilenka, Anastasiya and Biot, Francisco Mahedero and Wasielewska-Michniewska, Katarzyna and Ganzha, Maria and Paprzycki, Marcin and Palau, Carlos E.},
TITLE = {Flexible Deployment of Machine Learning Inference Pipelines in the Cloud–Edge–IoT Continuum},
JOURNAL = {Electronics},
VOLUME = {13},
YEAR = {2024},
NUMBER = {10},
ARTICLE-NUMBER = {1888},
URL = {https://www.mdpi.com/2079-9292/13/10/1888},
ISSN = {2079-9292},
ABSTRACT = {Currently, deploying machine learning workloads in the Cloud–Edge–IoT continuum is challenging due to the wide variety of available hardware platforms, stringent performance requirements, and the heterogeneity of the workloads themselves. To alleviate this, a novel, flexible approach for machine learning inference is introduced, which is suitable for deployment in diverse environments—including edge devices. The proposed solution has a modular design and is compatible with a wide range of user-defined machine learning pipelines. To improve energy efficiency and scalability, a high-performance communication protocol for inference is propounded, along with a scale-out mechanism based on a load balancer. The inference service plugs into the ASSIST-IoT reference architecture, thus taking advantage of its other components. The solution was evaluated in two scenarios closely emulating real-life use cases, with demanding workloads and requirements constituting several different deployment scenarios. The results from the evaluation show that the proposed software meets the high throughput and low latency of inference requirements of the use cases while effectively adapting to the available hardware. The code and documentation, in addition to the data used in the evaluation, were open-sourced to foster adoption of the solution.},
DOI = {10.3390/electronics13101888}
}

```

## Authors

- [Karolina Bogacka](https://orcid.org/0000-0002-7109-891X) ([GitHub](https://github.com/Karolina-Bogacka))
- [Anastasiya Danilenka](https://orcid.org/0000-0002-3080-0303) ([GitHub](https://github.com/adanilenka))

## Licensing
The repository is released under the Apache 2.0 license. See [LICENSE](LICENSE) for more information.
