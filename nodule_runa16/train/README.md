# Nodule Classification



## Files


| files | instructions |
|----------|----------|
| `src` | model building, model training and other files |
| `*.py` | training main files |
| `*.sh` | running files |



## Train

- train 3d-ResNet classification model

 ```bash
 bash run_10fold_3dresnet.sh
 ```

- train ordinal logistic regression model

 ```bash
bash run_10fold_olr.sh
 ```

- train normal  regression model

 ```bash
bash run_10fold_nreg.sh
 ```

- train DPN model (classification)

 ```bash
bash run_10fold_dpn92.sh
 ```

