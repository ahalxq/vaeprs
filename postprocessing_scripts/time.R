library(tidyverse)
library(ggplot2)

data <- read.table("output/comp_time.add.txt", header = TRUE, fill=TRUE)
data$en <- as.difftime(data$en, format="%H:%M:%S")
data$vae <- as.difftime(data$vae, format="%d-%H:%M:%S")
data$vae.p.t <- as.difftime(data$vae.p.t, format="%H:%M:%S")

dat <- pivot_longer(data, cols=-1, names_to = "method", values_to = "time")
method <- rep("VAE",nrow(dat))
method[grep("vae-p.t",dat$method)] <- "VAE-p.t"
method[grep("en",dat$method)] <- "EN-p.t"


# Automatic scale selection
png("../output/compTime.png", width = 1200, height = 800, res = 120)

ggplot(data = dat, aes(x = time, y = trait, fill = method)) + theme_bw() + geom_col(position = "dodge")

dev.off()