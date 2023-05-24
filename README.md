# VAE-PRS: A Variational Encoder-based Regression Model for Polygenic Trait Prediction
This repository contains the source code for VAE-PRS, a novel approach for polygenic trait prediction that employs a variational autoencoder-based regression model.

## Overview
The pipeline involves the following steps:

1. **Preprocessing**: The script preprocess.sh extracts selected variants and individuals from the entire genotype data and creates numpy arrays for each individual, where each line is a sample and each column is a genetic variant.

2. **LD Pruning and P-value Thresholding**: The script prune.sh uses PLINK to further prune selected variants with r threshold 0.1 and p-value threshold 1e-6.

3. **Model Training**: The script vae_modules.eur.py is used to train the VAE-PRS model with a selected trait.

4. **Evaluation**: The script explainer.eur.py interprets the trained model using SHAP.

## Installation
Before you begin, make sure you have installed the plink-pipelines python package and PLINK(https://www.cog-genomics.org/plink/2.0/). You can install it using pip:

```shell
pip install plink-pipelines
```

Please also make sure to create and activate a conda environment with dependencies for **VAE-PRS**.

```shell
conda env create -f environment.yml
conda activate vae_prs_env
```

## Usage
## Step 1: Preprocessing

To preprocess the data, run the following command:

```shell
bash preprocess.sh $trait
```
Where `$trait` is the phenotypic trait of interest.

## Step 2: LD Pruning and P-value Thresholding

To perform LD pruning and p-value thresholding, run the following command:

```shell
bash prune.sh $trait
```
## Step 3: Model Training

To train the VAE-PRS model, run the following command:

```shell
python vae_modules.eur.py --trait $trait
```
## Step 4: Evaluation

To interpret the trained model using SHAP, run the following command:

```shell
python explainer.eur.py
```
## Contact
If you have any issues or questions, feel free to open an issue in the Github repository or contact us at xli8@unc.edu.

