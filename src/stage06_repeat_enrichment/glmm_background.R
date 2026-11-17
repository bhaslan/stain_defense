suppressPackageStartupMessages(library(glmmTMB))

out <- commandArgs(trailingOnly = TRUE)[1]
cells <- read.delim(file.path(out, "cells.tsv"), check.names = FALSE)
classes <- jsonlite::fromJSON(file.path(out, "classes.json"))
groups <- grep("^w_", names(cells), value = TRUE)

cells$class <- factor(ifelse(cells$class_index == 0, "background", classes[pmax(cells$class_index, 1)]), levels = c("background", classes))
cells$focal <- as.numeric(cells$class_index > 0)
cells$genome <- factor(cells$genome)
cells$sample <- factor(cells$sample)

fit <- glmmTMB(cbind(k, n - k) ~ class + side + (1 | genome) + (0 + focal | sample), data = cells, family = binomial(link = "cloglog"))

modes <- ranef(fit)$cond
offset <- modes$genome[as.character(cells$genome), 1] + ifelse(cells$focal == 1, modes$sample[as.character(cells$sample), 1], 0)
design <- model.matrix(~ class + side, data = cells)
background_design <- design
background_design[, grep("^class", colnames(design))] <- 0
sample_effect <- ifelse(cells$focal == 1, modes$sample[as.character(cells$sample), 1], 0)
weights <- as.matrix(cells[groups])
rate <- function(x) 1 - exp(-exp(x))

rates <- function(beta) {
  focal <- colSums(weights * rate(drop(design %*% beta) + offset))
  background <- colSums(weights * rate(drop(background_design %*% beta) + offset - sample_effect))
  rbind(focal = focal, background = background)
}

estimate <- rates(fixef(fit)$cond)
summary <- data.frame(group = sub("^w_", "", groups), focal_rate = estimate["focal", ], background_rate = estimate["background", ],
                      difference = estimate["focal", ] - estimate["background", ], ratio = estimate["focal", ] / estimate["background", ],
                      converged = fit$sdr$pdHess)
write.table(summary, file.path(out, "glmm_summary.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
cat("ALL_DONE\n")
