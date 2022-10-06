# ORNet
ORNet is a regularized network for medical imaging.

This is the official repo for paper "Disease-Grading Networks with Ordinal Regularization for Medical Imaging" submitted to `Neurocomputing`.

Please note that some hyper-parameters(such as learing rate, batch size, etc.) may affect the performance, which can vary among different tasks/environments/software/hardware/random seeds, and thus careful tunning is required.

## Introduction
The severity of diseases develops gradually, and early screening is critical to apply timely medical interventions. Previous deep learning classification methods for disease grading have ignored the ordinal relationships among stages of disease severity, but this study shows they can be used to boost disease-grading performance. In this paper, we design an ordinal regularized module to represent the orderliness in disease severity, which can be flexibly embedded into general classification networks to grade diseases more accurately. In addition, this ordinal regularized module also predicts the progress of disease development. The proposed method is evaluated on three public benchmark datasets: the IDRiD challenge dataset, LUng Nodule Analysis 2016 (LUNA16) dataset, and Messidor dataset. Experiments show that the proposed method is not only superior to the baselines from common classification models but also outperforms deep learning approaches, especially on the IDRiD challenge dataset, where our method has a joint accuracy of 68.0\%. Furthermore, the proposed method achieves excellent performance in both single-disease and joint-disease grading tasks on the aforementioned datasets, and it can be applied to other disease-grading tasks.

## Installation
### Requirements
- Linux with Python = 3.6, Titan X GPU
- PyTorch = 1.4.0
- Torchvision = 0.5.0
- Timm = 0.4.12

## Folder tree

```
├── checkpoint.md
├── Data
│   ├── idrid
│   │   ├── clahe.py
│   │   ├── crop.py
│   │   ├── process.py
│   │   ├── readme.txt
│   │   ├── run_gray.sh
│   │   ├── test_resize1024
│   │   └── train_resize1024
│   └── messidor
│       ├── messidor_resize512
│       └── readme.txt
├── datautil
│   ├── getdataloader.py
│   ├── imgutil.py
│   └── __init__.py
├── ground
│   ├── idrid
│   │   ├── a. IDRiD_Disease Grading_Training Labels.csv
│   │   ├── b. IDRiD_Disease Grading_Testing Labels.csv
│   │   ├── split_txt.py
│   │   ├── test.txt
│   │   ├── train.txt
│   │   ├── valid_0.15.txt
│   │   └── valid.txt
│   └── messidor
│       ├── test0.txt
│       ├── test1.txt
│       ├── test2.txt
│       ├── test3.txt
│       ├── test4.txt
│       ├── test5.txt
│       ├── test6.txt
│       ├── test7.txt
│       ├── test8.txt
│       ├── test9.txt
│       ├── train0.txt
│       ├── train1.txt
│       ├── train2.txt
│       ├── train3.txt
│       ├── train4.txt
│       ├── train5.txt
│       ├── train6.txt
│       ├── train7.txt
│       ├── train8.txt
│       ├── train9.txt
│       ├── valid0.txt
│       ├── valid1.txt
│       ├── valid2.txt
│       ├── valid3.txt
│       ├── valid4.txt
│       ├── valid5.txt
│       ├── valid6.txt
│       ├── valid7.txt
│       ├── valid8.txt
│       └── valid9.txt
├── network
│   ├── __init__.py
│   ├── jointNet.py
│   ├── preweight
│   │   └── readme.txt
│   └── singleNet.py
├── outputs
├── nodule_runa16
├── README.md
├── run_single_idrid.sh
├── scripts
│   ├── run_joint_idrid.sh
│   ├── run_joint_messidor.sh
│   └── run_single_messidor.sh
├── train_r50_joint.py
├── train_r50_single.py
├── tree.txt
└── utils.py
```

## Quick Start
### Data Preparation
- data split (see in `./ground`)
- data processing (see in `./Data`). 

### Train ORNet
- single task
```bash
bash run_single_idrid.sh
```

- joint task
```bash
bash scripts/run_joint_idrid.sh
```

You can see more options from
```bash
python train_single.py -h
```


## Performance
- single task

| method   | backbone | dataset | disease | accuracy |
|----------|----------|---------|---------|----------|
| ResNet50 | -        | IDRiD   | DR      | 72.82   |
| ORNet(alpha=0.1)    | ResNet50        | IDRiD   | DR      | 75.73   |
| ORNet(alpha=0.125)   | ResNet50        | IDRiD   | DR      | 76.70   |
| ORNet(alpha=0.15)    | ResNet50        | IDRiD   | DR      | 73.79   |


- joint task

| method   | backbone | dataset | disease | Joint accuracy | DR accuracy | DME accuracy |
|----------|----------|---------|---------|----------|---------|----------|
| ResNet50 | -        | IDRiD   | DR&DME      | 62.14   | 71.84 | 79.61|
| ORNet(alpha=0.1) | ResNet50| IDRiD   | DR&DME  | 67.96   |75.73   |81.55   |

| method   | backbone | dataset | disease | Joint accuracy | DR accuracy | DME accuracy |
|----------|----------|---------|---------|----------|---------|----------|
| ResNet50 | -        | Messidor   | DR&DME      | 83.25   | 91.75 | 90.58 |
| ORNet(alpha=0.1) | ResNet50   | Messidor   | DR&DME   | 84.83   |93.17   |91.17   |
| ORNet(alpha=0.2) | ResNet50   | Messidor   | DR&DME   | 84.00   |92.17   |90.67   |
| ORNet(alpha=0.4) | ResNet50   | Messidor   | DR&DME   | 84.67   |92.83   |91.25  |
| ORNet(alpha=0.6) | ResNet50   | Messidor   | DR&DME   | 84.67   |92.59   |91.33   |

For medssidor dataset, the results are the mean values of 10-fold cross-validation.

## Citing ORNet
If you find this repo useful for your research, please consider citing the paper (waiting for review results).
```
@inproceedings{tang2022disease,
  title={Disease-Grading Networks with Ordinal Regularization for Medical Imaging},
  author={Tang, Wenqiang and Yang, zhouwang and Song, Yanzhi},
  booktitle={},
  pages={},
  year={2022}
}
```
