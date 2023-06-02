library(tidyverse)

perf <- read.table("../output/perf.355k.txt", header = TRUE, fill=TRUE)

dat <- pivot_longer(perf, cols=-1, names_to = "method", values_to = "PCC")

sampleSize <- rep("355k", nrow(dat))

method <- rep("VAE",nrow(dat))
method[grep("en",dat$method)] <- "EN-p.t"
method[grep("vae.s355k.p.t",dat$method)] <- "VAE-p.t"

method[grep("PRSice",dat$method)] <- "PRSice"

ver <- rep("v1", nrow(dat))
ver[grep("v2",dat$method)] <- "v2"
ver[grep("v3",dat$method)] <- "v3"

comb <- data.frame(trait=dat$trait,method,sampleSize,version = ver,PCC=dat$PCC)

summarized <- comb %>% group_by(trait, method, sampleSize) %>% summarize(median = median(PCC), low = min(PCC), up = max(PCC))

png("../output/perf.355k.png", width = 1800, height = 800, res = 120)
summarized %>% ggplot(aes(x = trait, y = median, fill = method)) + 
  theme_bw() +
  theme(text = element_text(size = 20))  + 
  geom_col(position = "dodge")  + 
  labs(y = "PCC", x = "Blood Cell Traits") + 
  scale_fill_discrete(name = "Method") + 
  scale_fill_manual(values=c("#F8766D", "#00BA38", "#619CFF", "#C77CFF")) + 
  coord_cartesian(ylim = c(.2,.6))

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
