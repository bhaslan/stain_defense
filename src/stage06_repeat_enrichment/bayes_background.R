suppressPackageStartupMessages({
  library(cmdstanr)
  library(posterior)
  library(jsonlite)
})

options(cmdstanr_no_ver_check = TRUE)
here <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
out <- commandArgs(trailingOnly = TRUE)[1]
settings <- fromJSON(file.path(out, "settings.json"))
cells <- read.delim(file.path(out, "cells.tsv"), check.names = FALSE)
groups <- grep("^w_", names(cells), value = TRUE)
classes <- fromJSON(file.path(out, "classes.json"))

data <- list(R = nrow(cells), G = max(cells$genome), S = max(cells$sample), C = length(classes), M = length(groups),
             n = cells$n, k = cells$k, class_index = cells$class_index, genome = cells$genome, sample = cells$sample,
             side = cells$side, W = as.matrix(cells[groups]))

model <- cmdstan_model(file.path(here, "bayes_background.stan"), dir = settings$build_dir)
fit <- model$sample(data = data, chains = settings$chains, parallel_chains = settings$chains, iter_warmup = settings$warmup,
                    iter_sampling = settings$sampling, seed = settings$seed, adapt_delta = settings$adapt_delta, refresh = 0, show_messages = FALSE)

draws <- as_draws_df(fit$draws(c("sigma_genome", "sigma_sample", "focal_rate", "background_rate", "difference", "ratio")))
interval <- function(x) c(median = median(x), low = unname(quantile(x, 0.025)), high = unname(quantile(x, 0.975)), above_zero = mean(x > 0))

rows <- lapply(seq_along(groups), function(m) {
  values <- c(focal_rate = median(draws[[sprintf("focal_rate[%d]", m)]]), background_rate = median(draws[[sprintf("background_rate[%d]", m)]]))
  difference <- interval(draws[[sprintf("difference[%d]", m)]])
  ratio <- interval(draws[[sprintf("ratio[%d]", m)]])[c("median", "low", "high")]
  data.frame(group = sub("^w_", "", groups[m]), t(values), t(setNames(difference, paste0("difference_", names(difference)))),
             t(setNames(ratio, paste0("ratio_", names(ratio)))), check.names = FALSE)
})
summary <- do.call(rbind, rows)
write.table(summary, file.path(out, "bayes_summary.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)

diagnostics <- fit$diagnostic_summary(quiet = TRUE)
parameters <- summarise_draws(fit$draws(c("alpha", "delta", "beta", "sigma_genome", "sigma_sample")), "rhat", "ess_bulk")
sampler <- data.frame(divergences = sum(diagnostics$num_divergent), treedepth_hits = sum(diagnostics$num_max_treedepth),
                      max_rhat = max(parameters$rhat), min_ess_bulk = min(parameters$ess_bulk))
write.table(sampler, file.path(out, "bayes_sampler.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
cat("ALL_DONE\n")
