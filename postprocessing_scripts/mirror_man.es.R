library(ggplot2)
library(ggExtra)
library(grid)
library(gtable)
library(dplyr)

args <- commandArgs(TRUE)
trait <- args[1]

library(karyoploteR)

  dg1 <- read.table(paste0("../output/shap/ukb_EUR_",trait,".t100k.txt"),header=F);
  colnames(dg1) = c("chr","pos","rsname","pval");
  dg1  <- dg1 %>% select("chr","pos","rsname","pval") %>% mutate(pval = pval*10000)
  dg1_format = subset(dg1,select = c("chr","pos","pos","rsname","pval"))
  dg1_format = dg1_format[order(dg1$chr,dg1$pos),];
  # dg1_format$chr <- paste0("chr",dg1_format$chr)
  dg1_format$rsname = as.character(dg1_format$rsname)
  gwas1 <- toGRanges(dg1_format);
  vae_max = max(dg1_format$pval)
  
  dg2 <- read.table(gzfile(paste0("../output/GWAS/",trait,"/ukb_EUR_",trait,"_t100k.beta.regenie")),header=F)
  colnames(dg2) = c("chr","pos","rsname","pval");
  dg2  <- dg2 %>% select("chr","pos","rsname","pval") %>% mutate(pval = abs(pval), chr = paste0("chr",chr))
  dg2_format = subset(dg2,select = c("chr","pos","pos","rsname","pval")) 
  dg2_format = dg2_format[order(dg2$chr,dg2$pos),];
  # dg2_format$chr <- paste0("chr",dg2_format$chr)
  dg2_format$rsname = as.character(dg2_format$rsname)
  gwas2 <- toGRanges(dg2_format);
  gwas_max = 1
  
  merged_data <- merge(dg1_format, dg2_format, by = "rsname") %>%
    select(pval1 = pval.x, pval2 = pval.y) %>%
    mutate(pval1 = scale(pval1),pval2 = scale(pval2))
  cat(trait," Pearson:")
  cor(merged_data$pval1, merged_data$pval2, method = c("pearson"))
  cat(trait," Spearman:")
  cor(merged_data$pval1, merged_data$pval2, method = c("spearman"))
## spearman
  
  # # Install the 'psych' package if not already installed
  # if (!requireNamespace("psych", quietly = TRUE)) {
  #   install.packages("psych")
  # }
  # 
  # library(psych)
  # 
  # # Set different thresholds
  # thresholds <- c(10, 20, 25, 27.5, 50)
  # 
  # # Calculate Cohen's kappa for each threshold
  # for (threshold in thresholds) {
  #   cat("Threshold:", threshold, "\n")
  #   
  #   # Create binary variables based on the threshold
  #   pval1_binary <- ifelse(merged_data$pval1 <= threshold, 1, 0)
  #   pval2_binary <- ifelse(merged_data$pval2 <= threshold, 1, 0)
  #   
  #   # Create contingency table
  #   contingency_table <- table(pval1_binary, pval2_binary)
  #   cat("Contingency table:\n")
  #   print(contingency_table)
  #   
  #   # Calculate Cohen's kappa
  #   kappa <- cohen.kappa(contingency_table)$kappa
  #   cat("Cohen's kappa:", kappa, "\n\n")
  # }
  
  # plot(merged_data$pval2)
  
  head(gwas1)
  head(gwas2)
  
  pdf(paste0("../output/figs/mirror_",trait,".beta.pdf"), width = 20, height = 4)
  kp <- plotKaryotype(genome="hg38",chromosomes = c("chr1","chr2","chr3","chr4","chr5","chr6","chr7","chr8","chr9","chr10",
                        "chr11","chr12","chr13","chr14","chr15","chr16","chr17","chr18","chr19","chr20","chr21","chr22"),
                         plot.type=4,ideogram.plotter = NULL,labels.plotter = NULL)
  #kpAddCytobandsAsLine(kp)
  # change y limit to see rank of shap values
  kpAddLabels(kp, labels = paste0("SHAP"), srt=90, pos=3, r0=0.5, r1=1, cex=1, label.margin = 0.04)
  kpAxis(kp, r0=0.5, ymin=0, ymax = vae_max)
  kp <- kpPlotManhattan(kp, data=gwas1, ymin=0, ymax = vae_max, points.col = "2blues",r0=0.5, r1=1, suggestiveline = 10,genomewideline = 0,suggestive.col="red", genomewide.col = "black",points.cex=0.4, logp = FALSE)
  kpAddChromosomeNames(kp, srt=45)
  
  kpAddLabels(kp, labels = paste0("|beta|"), srt=90, pos=3, r0=0, r1=0.5, cex=1, label.margin = 0.04)
  kpAxis(kp, ymin=0, ymax = gwas_max, r0=0.5, r1=0)
  kp <- kpPlotManhattan(kp, data=gwas2, ymin=0, ymax = gwas_max, r0=0.5, r1=0, suggestiveline = 7.30103, genomewideline = 0, suggestive.col="red", genomewide.col = "black", points.cex=0.4, logp = FALSE)
  
  dev.off()


