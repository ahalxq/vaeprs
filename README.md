# VAE-PRS: A Variational Encoder-based Regression Model for Polygenic Trait Prediction
This repository contains the source code for VAE-PRS, a novel approach for polygenic trait prediction that employs a variational autoencoder-based regression model.

## Overview
The pipeline involves the following steps:

1. **Preprocessing**: The script `preprocess.sh` extracts selected variants and individuals from the entire genotype data and creates numpy arrays for each individual.

2. **LD Pruning and P-value Thresholding**: The script `prune.sh` uses PLINK to further prune selected variants with r threshold 0.1 and p-value threshold 1e-6.

3. **Model Training**: The script `vae_modules.eur.py` is used to train the VAE-PRS model with a selected trait.

4. **Evaluation**: The script `explainer.eur.py` interprets the trained model using SHAP.

## Installation
Before you begin, make sure to create and activate a conda environment with dependencies for **VAE-PRS**. 

```shell
conda env create -f environment.yml
conda activate vae_prs_env
```

Please also make sure you have installed the plink-pipelines python package and PLINK(https://www.cog-genomics.org/plink/2.0/). You can install it using pip:

```shell
pip install plink-pipelines
```

## Usage
## Step 1: Preprocessing

To preprocess the data, please have the plink-formated genotype files and sample ids for training (`train.eid`) and testing (test.eid`) ready under `$indir` defined at the top of `preprocess.sh` and run the following command:

```shell
bash preprocess.sh $trait $feature
```
Where `$trait` is the phenotypic trait of interest and `$feature` is the name for selected variants.

The sample ids should be formatted as:
```shell
1
2
3
4
5
6
7
8
```
## Step 2: LD Pruning and P-value Thresholding

To perform LD pruning and p-value thresholding, GWAS summary statistics will also be needed under `$indir` and run the following command:

```shell
bash prune.sh $trait $feature
```
In this script, I assumed the GWAS summary statistics are tab separated and have p-value in column 14. This could be changed accordingly.
## Step 3: Model Training

To train the VAE-PRS model, make sure to have the corresponding phenotype table ready under `--home` run the following command:

```shell
python vae_modules.eur.py --trait $trait --feature $feature --home $input_dir --outdir $output_dir
```
The phenotype table should be tab separated and follow format:
```shell
FID IID trait_1	trait_2	trait_3
1	1	0.10247	0.352177	0.487053
2	2	-1.13322	0.145163	-1.28443
3	3	-0.38803	0.464737	0.433657
4	4	-0.13985	1.19718	-0.49138
5	5	-0.146531	0.824333	1.45632
6	6	-0.678787	1.73982	2.05471
7	7	1.28606	0.339639  0.772062
8	8	-0.371633	-0.938614	-0.0811301
```
## Step 4: Evaluation

To interpret the trained model using SHAP, run the following command:

```shell
python explainer.eur.py
```
## Contact
If you have any issues or questions, feel free to open an issue in the Github repository or contact us at xli8@unc.edu.

