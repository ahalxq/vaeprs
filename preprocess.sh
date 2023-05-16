#!/bin/bash
home="/proj/yunligrp/users/xiaoqi/prs_dl/"
outdir="/work/users/x/i/xiaoqil/prsdl/"
# outdir="/pine/scr/x/i/xiaoqil/prs_dl/"
# outdir="/proj/yunligrp/users/xiaoqi/prs_dl/data/"
trait=$1

#### Filter variants and combine significant ones together across chromosomes
cd $home/output/GWAS/
## Combine summary statistics across chromosomes together
paste -d' ' <(cat ${trait}/ukb_EUR_chr{1..22}_${trait}.regenie | awk '$2 != "GENPOS"{print substr($1":"$2":"$5":"$4,1,100)}') <(cat ${trait}/ukb_EUR_chr{1..22}_${trait}.regenie | awk '$2 != "GENPOS"' | cut -d" " -f 1-2,4-) <(cat ${trait}/ukb_EUR_chr{1..22}_${trait}.regenie | awk '$2 != "GENPOS"{print 10^-$13}') |  cut -d" " -f-13,15 | awk '$13!="NA"' > ${trait}/ukb_EUR_${trait}_complete.regenie
sed -i "1iID CHROM GENPOS ALLELE0 ALLELE1 A1FREQ INFO SAMPLE_SIZE TEST BETA SE CHISQ LOG10P P" ${trait}/ukb_EUR_${trait}_complete.regenie

# ## Sort and select variants based on specific GWAS traits and take union across all traits of interest
sort -g --parallel=20 --buffer-size=40G -k13,13 -r ${trait}/ukb_EUR_${trait}_complete.regenie | head -n100001 | cut -d" " -f1 > ukb_EUR_${trait}.t100k.variants.list
wc -l ukb_EUR_${trait}.t100k.variants.list

cd $home/data
[ ! -d $outdir/$trait ] && mkdir $outdir/$trait
# ## create the list of ready-to-merge files 
### EUR
# for i in {1..22}; do echo $outdir/ukb_imp_chr$i >> ukb_EUR.merge_list.txt; done
# echo $outdir/ukb_imp_chrX >> ukb_EUR.merge_list.txt
# for i in {1..22}; do echo geno/ukb_EUR_test_chr$i.nodup >> ukb_EUR_test.merge_list.txt; done
# echo geno/ukb_EUR_test_chrX.nodup >> ukb_EUR_test.merge_list.txt

# paste $home/ukb_icd10/EUR.train.txt $home/ukb_icd10/EUR.train.txt > $home/ukb_icd10/EUR.train.eid
# sed -i '1i#FID\tIID' $home/ukb_icd10/EUR.train.eid
# paste $home/ukb_icd10/EUR.test.txt $home/ukb_icd10/EUR.test.txt > $home/ukb_icd10/EUR.test.eid
# sed -i '1i#FID\tIID' $home/ukb_icd10/EUR.test.eid


# ## subset variants and combine genotype across chrs 
### EUR 
plink2 --pmerge-list $home/data/ukb_EUR.merge_list.txt bfile --keep $home/ukb_icd10/EUR.train.eid --extract $home/output/GWAS/ukb_EUR_${trait}.t100k.variants.list --make-bed --out $outdir/$trait/ukb_EUR_${trait}_train.t100k
# mkdir $outdir/$trait/ukb_EUR_${trait}_train.t100k
# mv $outdir/$trait/ukb_EUR_${trait}_train.t100k* $outdir/$trait/ukb_EUR_${trait}_train.t100k
# rm $outdir/$trait/ukb_EUR_${trait}_train.t100k/*psam  $outdir/$trait/ukb_EUR_${trait}_train.t100k/*pgen $outdir/$trait/ukb_EUR_${trait}_train.t100k/*pvar
[ ! -d $outdir/$trait/ukb_EUR_${trait}_train.t100k] && mkdir $outdir/$trait/ukb_EUR_${trait}_train.t100k
mv $outdir/$trait/ukb_EUR_${trait}_train.t100k.??? $outdir/$trait/ukb_EUR_${trait}_train.t100k
rm $outdir/$trait/ukb_EUR_${trait}_train.t100k/*psam  $outdir/$trait/ukb_EUR_${trait}_train.t100k/*pgen $outdir/$trait/ukb_EUR_${trait}_train.t100k/*pvar
# 
plink2 --bfile $outdir/ukb_EUR_test --keep $home/ukb_icd10/EUR.test.eid --extract $home/output/GWAS/ukb_EUR_${trait}.t100k.variants.list --make-bed --out $outdir/$trait/ukb_EUR_${trait}_test.t100k
[ ! -d $outdir/$trait/ukb_EUR_${trait}_test.t100k] && mkdir $outdir/$trait/ukb_EUR_${trait}_test.t100k
mv $outdir/$trait/ukb_EUR_${trait}_test.t100k.??? $outdir/$trait/ukb_EUR_${trait}_test.t100k
rm $outdir/$trait/ukb_EUR_${trait}_test.t100k/*psam $outdir/$trait/ukb_EUR_${trait}_test.t100k/*pgen $outdir/$trait/ukb_EUR_${trait}_test.t100k/*pvar
## convert plink to npys separated by sample ids
# cd ..
conda activate vcfnp
# ## EUR
[ -d $outdir/$trait/npy_ukb_EUR_${trait}_train.t100k ] && rsync -a --delete tmp/ $outdir/$trait/npy_ukb_EUR_${trait}_train.t100k/
mkdir $outdir/$trait/npy_ukb_EUR_${trait}_train.t100k
plink_pipelines --raw_data_path $outdir/$trait/ukb_EUR_${trait}_train.t100k --output_folder $outdir/$trait/npy_ukb_EUR_${trait}_train.t100k
[ -d $outdir/$trait/npy_ukb_EUR_${trait}_test.t100k ] && rsync -a --delete tmp/ $outdir/$trait/npy_ukb_EUR_${trait}_test.t100k/
mkdir $outdir/$trait/npy_ukb_EUR_${trait}_test.t100k
plink_pipelines --raw_data_path $outdir/$trait/ukb_EUR_${trait}_test.t100k --output_folder $outdir/$trait/npy_ukb_EUR_${trait}_test.t100k
