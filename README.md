# Hyperspectral Band Selection with Dynamic Graph Enhancement and Hierarchical Feature Fusion

This repo provides a Python implementation of paper “Hyperspectral 
Band Selection with Dynamic Graph Enhancement and Hierarchical 
Feature Fusion,” IEEE Transactions on Geoscience and Remote Sensing
(IEEE TGRS), DOI: xxxxx. The code will be made publicly available upon acceptance of the paper.

This code is provided solely for academic and research purposes. 
Redistribution of the code, in whole or in part, including sharing 
download links or uploading to public websites, is not permitted.

If you use this code in your research, please cite our paper. We hope this code will be helpful to your work.

## Environment

- Operating System : Windows 11
- Python : 3.8.20 
- CUDA : 11.8
- cuDNN: 8.9
- torchvision: 0.18.0
- numpy : 1.24.3
- scikit-image : 0.20.0
- scikit-learn : 1.3.2
- scipy : 1.10.1

The code was developed and tested on an NVIDIA GPU with CUDA 11.8.
CPU-only execution is not supported.

## Dataset

The experiments are conducted on a publicly available hyperspectral dataset.

IP and PU datasets are available at: http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes.

DC dataset is available at: https://engineering.purdue.edu/~biehl/MultiSpec/hyperspectral.html

You need to download the above datasets and then move the files to `./data` folder.

## Usage

### Step 1: Configure dataset path

Before running the code, please open `main.py` and modify the dataset path
to match the location of your dataset on your local machine.

### Step 2: Set hyperparameters

You may adjust the following hyperparameters in `main.py` according to your needs:

`k`:  the number of neighbors in the BASG initialization

`lambda_da`:  the weight coefficient for balancing the losses L<sub>AE and L<sub>DEGCN

`mu`: the fusion coefficient for integrating new and old graph

### Step 3: Run the code

After setting the dataset path and parameters, run: 
`python main.py`

### Outputs

After execution, the following files will be generated:

`selected_bands.txt`:
contains the indices of the selected spectral bands.

`loss_curve.png`:
the training loss curve saved as an image file.

## Citation

If you find this work useful, please cite:

```bibtex
@article{xxxxxx,
  title={XXX},
  author={xxxx},
  journal={xxxx},
  year={xx}
}

