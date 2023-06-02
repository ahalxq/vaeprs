library(tidyverse)
t <- read.table("tmp")
tmp <-matrix(unlist(t), 5,15)
write.table(tmp, "tmp", row.names=FALSE,sep="\t", quote = FALSE)
