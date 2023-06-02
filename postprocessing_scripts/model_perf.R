library(tidyverse)

perf <- read.table("../output/perf.0525.txt", header = TRUE, fill=TRUE)

dat <- pivot_longer(perf, cols=-1, names_to = "method", values_to = "PCC")

sampleSize <- rep("355k", nrow(dat))
sampleSize[grep("s5k",dat$method)] <- "5k"
sampleSize[grep("s10k",dat$method)] <- "10k"
sampleSize[grep("s50k",dat$method)] <- "50k"
sampleSize[grep("s100k",dat$method)] <- "100k"
sampleSize[grep("s150k",dat$method)] <- "150k"
sampleSize[grep("s200k",dat$method)] <- "200k"

method <- rep("VAE",nrow(dat))
method[grep("en",dat$method)] <- "EN-p.t"
method[grep("vae.s10k.p.t",dat$method)] <- "VAE-p.t"
method[grep("vae.s50k.p.t",dat$method)] <- "VAE-p.t"
method[grep("vae.s100k.p.t",dat$method)] <- "VAE-p.t"
method[grep("vae.s150k.p.t",dat$method)] <- "VAE-p.t"
method[grep("vae.s200k.p.t",dat$method)] <- "VAE-p.t"
method[grep("vae.s355k.p.t",dat$method)] <- "VAE-p.t"

method[grep("PRSice",dat$method)] <- "PRSice"

ver <- rep("v1", nrow(dat))
ver[grep("v2",dat$method)] <- "v2"
ver[grep("v3",dat$method)] <- "v3"

comb <- data.frame(trait=dat$trait,method,sampleSize,version = ver,PCC=dat$PCC)

comb$sampleSize <- factor(comb$sampleSize, levels = c("5k", "10k","50k","100k","150k","200k","355k"))
summarized <- comb %>% group_by(trait, method, sampleSize) %>% summarize(median = median(PCC), low = min(PCC), up = max(PCC))

png("../output/perf.png", width = 1500, height = 800, res = 120)
summarized %>% ggplot(aes(x = sampleSize, y = median, fill = method)) + 
  facet_wrap(~trait)+ 
  theme_bw() +
  theme(text = element_text(size = 20))  + 
  geom_col(position = "dodge") + 
  geom_point(data = summarized, aes(x = sampleSize, y = median), position = position_dodge(width = .9), alpha = .5) + 
  labs(y = "PCC", x = "Sample Size") + scale_fill_discrete(name = "Method") + 
  scale_fill_manual(values=c("#F8766D", "#00BA38", "#619CFF", "#C77CFF")) 

#  + geom_bar(stat="identity", position=position_dodge()) 
dev.off()

# data <- read.table('../output/memory.add.txt', header = TRUE, )
# 
# # Melt dataframe to long format
# data_melted <- data %>% pivot_longer(-trait, names_to = 'method', values_to = 'memory')
# 
# # Create violin plot
# 
# ggplot(data_melted, aes(x = method, y = memory)) +
#   geom_boxplot(scale = 'width') +
#   labs(title = 'Computation Time for Three Different Methods', x = 'Trait', y = 'Memory (GB)')
