# VAE-PRS: A Variational Encoder-based Regression Model for Polygenic Trait Prediction
This repository contains the source code for VAE-PRS, a novel approach for polygenic trait prediction that employs a variational autoencoder-based regression model.

## Overview
The pipeline involves the following steps:

1. Preprocessing: The script preprocess.sh extracts selected variants and individuals from the entire genotype data and creates numpy arrays for each individual, where each line is a sample and each column is a genetic variant.

2. LD pruning and p-value thresholding: The script prune.sh uses PLINK to prune variants with r threshold 0.1 and p-value threshold 1e-6.

3. Model Training: The script vae_modules.eur.py is used to train the VAE-PRS model with a selected trait.

4. Evaluation: The script explainer.eur.py interprets the trained model using SHAP.

## Installation
Before you begin, make sure you have installed the plink-pipelines python package. You can install it using pip:

`pip install plink-pipelines`

## Usage
## Step 1: Preprocessing

To preprocess the data, run the following command:

`bash preprocess.sh $trait`
Where `$trait` is the trait of interest.

## Step 2: LD Pruning and P-value Thresholding

To perform LD pruning and p-value thresholding, run the following command:

`bash prune.sh $trait`
## Step 3: Model Training

To train the VAE-PRS model, run the following command:

`python vae_modules.eur.py --trait $trait`
## Step 4: Evaluation

To interpret the trained model using SHAP, run the following command:

`python explainer.eur.py`
## Contact
If you have any issues or questions, feel free to open an issue in the Github repository or contact us.

