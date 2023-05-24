#!/bin/bash
home="/proj/yunligrp/users/xiaoqi/prs_dl/"

outdir="/work/users/x/i/xiaoqil/prsdl/"
trait=$1

conda activate vae_prs_env

cd $home/output/GWAS/
grep -wf ukb_EUR_${trait}.t100k.variants.list ${trait}/ukb_EUR_${trait}_complete.regenie | awk -F" " '$14 < 1e-6 {print $1}' > ${trait}/ukb_EUR_${trait}.t100k.pe6.variants.list
wc -l ${trait}/ukb_EUR_${trait}.t100k.pe6.variants.list 

cd $home/
plink --bfile $outdir/$trait/ukb_EUR_${trait}_train.t100k/ukb_EUR_${trait}_train.t100k --extract $home/output/GWAS/${trait}/ukb_EUR_${trait}.t100k.pe6.variants.list --indep-pairwise 50 5 0.1 --out $outdir/$trait/ukb_EUR_${trait}_train.t100k/ukb_EUR_${trait}_train.t100k.pe6.LDpruned 
plink --bfile $outdir/$trait/ukb_EUR_${trait}_train.t100k/ukb_EUR_${trait}_train.t100k --extract $outdir/$trait/ukb_EUR_${trait}_train.t100k/ukb_EUR_${trait}_train.t100k.pe6.LDpruned.prune.in --make-bed --out $outdir/$trait/ukb_EUR_${trait}_train.t100k/ukb_EUR_${trait}_train.t100k.pe6.LDpruned
mkdir $outdir/$trait/ukb_EUR_${trait}_train.t100k.pe6.LDpruned
mv $outdir/$trait/ukb_EUR_${trait}_train.t100k/ukb_EUR_${trait}_train.t100k.pe6.LDpruned* $outdir/$trait/ukb_EUR_${trait}_train.t100k.pe6.LDpruned

mkdir $outdir/$trait/ukb_EUR_${trait}_test.t100k.pe6.LDpruned
# plink --bfile $outdir/$trait/ukb_EUR_${trait}_test.t100k/ukb_EUR_${trait}_test.t100k --indep-pairwise 50 5 0.1 --out $outdir/$trait/ukb_EUR_${trait}_test.t100k/ukb_EUR_${trait}_test.t100k.pe6.LDpruned
plink --bfile $outdir/$trait/ukb_EUR_${trait}_test.t100k/ukb_EUR_${trait}_test.t100k --extract $outdir/$trait/ukb_EUR_${trait}_train.t100k.pe6.LDpruned/ukb_EUR_${trait}_train.t100k.pe6.LDpruned.prune.in --make-bed --out $outdir/$trait/ukb_EUR_${trait}_test.t100k/ukb_EUR_${trait}_test.t100k.pe6.LDpruned
mv $outdir/$trait/ukb_EUR_${trait}_test.t100k/ukb_EUR_${trait}_test.t100k.pe6.LDpruned* $outdir/$trait/ukb_EUR_${trait}_test.t100k.pe6.LDpruned

# ## EUR
[ -d $outdir/$trait/npy_ukb_EUR_${trait}_train.t100k.pe6.LDpruned ] && rm -rf $outdir/$trait/npy_ukb_EUR_${trait}_train.t100k.pe6.LDpruned
mkdir $outdir/$trait/npy_ukb_EUR_${trait}_train.t100k.pe6.LDpruned
plink_pipelines --raw_data_path $outdir/$trait/ukb_EUR_${trait}_train.t100k.pe6.LDpruned --output_folder $outdir/$trait/npy_ukb_EUR_${trait}_train.t100k.pe6.LDpruned
[ -d $outdir/$trait/npy_ukb_EUR_${trait}_test.t100k.pe6.LDpruned ] && rm -rf $outdir/$trait/npy_ukb_EUR_${trait}_test.t100k.pe6.LDpruned
mkdir $outdir/$trait/npy_ukb_EUR_${trait}_test.t100k.pe6.LDpruned
plink_pipelines --raw_data_path $outdir/$trait/ukb_EUR_${trait}_test.t100k.pe6.LDpruned --output_folder $outdir/$trait/npy_ukb_EUR_${trait}_test.t100k.pe6.LDpruned
