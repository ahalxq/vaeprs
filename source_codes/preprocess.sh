#!/bin/bash
indir="./vaeprs/simulated"
outdir="./vaeprs/simulated"
trait=$1
feature=$2

conda activate vae_prs_env

[ ! -d $outdir/$trait ] && mkdir $outdir/$trait

## subset variants and split data into training & testing sets 
plink2 --bfile $indir/example --keep $indir/train.eid --extract $indir/${trait}.${feature}.variants.list --make-bed --out $outdir/$trait/${trait}_train.${feature}
 ! -d $outdir/$trait/${trait}_train.${feature}] && mkdir $outdir/$trait/${trait}_train.${feature}
mv $outdir/$trait/${trait}_train.${feature}.??? $outdir/$trait/${trait}_train.${feature}
rm $outdir/$trait/${trait}_train.${feature}/*psam  $outdir/$trait/${trait}_train.${feature}/*pgen $outdir/$trait/${trait}_train.${feature}/*pvar

plink2 --bfile $indir/example --keep $indir/test.eid --extract $indir/${trait}.${feature}.variants.list --make-bed --out $outdir/$trait/${trait}_test.${feature}
[ ! -d $outdir/$trait/${trait}_test.${feature}] && mkdir $outdir/$trait/${trait}_test.${feature}
mv $outdir/$trait/${trait}_test.${feature}.??? $outdir/$trait/${trait}_test.${feature}
rm $outdir/$trait/${trait}_test.${feature}/*psam $outdir/$trait/${trait}_test.${feature}/*pgen $outdir/$trait/${trait}_test.${feature}/*pvar

## convert plink to npys separated by sample ids
[ -d $outdir/$trait/npy_${trait}_train.${feature} ] && rsync -a --delete tmp/ $outdir/$trait/npy_${trait}_train.${feature}/
mkdir $outdir/$trait/npy_${trait}_train.${feature}
plink_pipelines --raw_data_path $outdir/$trait/${trait}_train.${feature} --output_folder $outdir/$trait/npy_${trait}_train.${feature}
[ -d $outdir/$trait/npy_${trait}_test.${feature} ] && rsync -a --delete tmp/ $outdir/$trait/npy_${trait}_test.${feature}/
mkdir $outdir/$trait/npy_${trait}_test.${feature}
plink_pipelines --raw_data_path $outdir/$trait/${trait}_test.${feature} --output_folder $outdir/$trait/npy_${trait}_test.${feature}
