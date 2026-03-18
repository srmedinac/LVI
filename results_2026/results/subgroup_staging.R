options(warn = -1)
library(survival)
library(survminer)
library(ggplot2)
library(gridExtra)
library(grid)

save_path <- "/media/smedin7/data/LVI/results_2026/results"

save_km <- function(km_plot, filename, width = 10, height = 8) {
  if (is.null(km_plot)) return(invisible(NULL))
  plot_grob <- ggplotGrob(km_plot$plot)
  table_grob <- ggplotGrob(km_plot$table)
  combined <- arrangeGrob(plot_grob, table_grob, ncol = 1, heights = c(2, 0.5))
  ggsave(filename, plot = combined, width = width, height = height, dpi = 600)
  cat("Saved:", filename, "\n")
}

make_subgroup_km <- function(subset_data, title_suffix, filename) {
  n <- nrow(subset_data)
  if (n < 10) {
    cat("Skipping", title_suffix, "- only", n, "patients\n")
    return(NULL)
  }

  tab <- table(subset_data$risk_group)
  if (length(tab) < 2 || any(tab < 3)) {
    cat("Skipping", title_suffix, "- insufficient group sizes:", paste(tab), "\n")
    return(NULL)
  }

  subset_data$risk_group <- factor(subset_data$risk_group, levels = c("Low Risk", "High Risk"))
  fit <- survfit(Surv(os_time_years, os_event) ~ risk_group, data = subset_data)
  cox <- coxph(Surv(os_time_years, os_event) ~ risk_group, data = subset_data)
  s <- summary(cox)

  hr <- s$conf.int[1, "exp(coef)"]
  lo <- s$conf.int[1, "lower .95"]
  hi <- s$conf.int[1, "upper .95"]
  p <- s$coefficients[1, "Pr(>|z|)"]
  hr_text <- sprintf("HR: %.2f (95%% CI: %.2f-%.2f)", hr, lo, hi)

  cat(sprintf("  %s (n=%d): HR=%.2f (%.2f-%.2f), p=%.4f\n", title_suffix, n, hr, lo, hi, p))

  p_km <- ggsurvplot(
    fit, data = subset_data,
    pval = TRUE, conf.int = TRUE, risk.table = TRUE,
    title = paste0("Perivascular AI Risk — ", title_suffix, " (n=", n, ")"),
    conf.int.style = "ribbon",
    xlab = "Time (years)", ylab = "Overall Survival",
    legend = "top",
    legend.title = "Group", legend.labs = c("Low Risk", "High Risk"),
    palette = c("#00BFC4", "#F8766D"),
    linetype = c("solid", "dashed"),
    surv.median.line = "hv",
    risk.table.height = 0.2,
    font.main = c(18, "bold"),
    conf.int.alpha = 0.1,
    font.x = c(17), font.y = c(17), font.legend = c(17),
    font.tickslab = c(16),
    risk.table.fontsize = 6,
    pval.size = 6.5
  )

  max_x <- max(subset_data$os_time_years, na.rm = TRUE)
  p_km$plot <- p_km$plot +
    annotate("text", x = max_x * 0.02, y = 0.15, label = hr_text,
             hjust = 0, size = 6.5)

  save_km(p_km, file.path(save_path, filename))
  return(p_km)
}

# ==============================================================================
# EMORY SUBGROUPS
# ==============================================================================
cat("\n========== EMORY SUBGROUP ANALYSES ==========\n\n")

emory <- read.csv(file.path(save_path, "emory_test_staging.csv"))

cat("--- Emory: pT3-4 ---\n")
make_subgroup_km(
  emory[emory$pT_group == "pT3-4", ],
  "Emory pT3-4", "km_subgroup_emory_pT3_4.png"
)

cat("\n--- Emory: N+ ---\n")
make_subgroup_km(
  emory[!is.na(emory$pN_group) & emory$pN_group == "N+", ],
  "Emory Node-Positive", "km_subgroup_emory_Npos.png"
)

cat("\n--- Emory: N0 ---\n")
make_subgroup_km(
  emory[!is.na(emory$pN_group) & emory$pN_group == "N0", ],
  "Emory Node-Negative", "km_subgroup_emory_N0.png"
)

cat("\n--- Emory: pT3-4 N+ ---\n")
make_subgroup_km(
  emory[emory$pT_group == "pT3-4" & !is.na(emory$pN_group) & emory$pN_group == "N+", ],
  "Emory pT3-4 N+", "km_subgroup_emory_pT3_4_Npos.png"
)

cat("\n--- Emory: pT3-4 N0 ---\n")
make_subgroup_km(
  emory[emory$pT_group == "pT3-4" & !is.na(emory$pN_group) & emory$pN_group == "N0", ],
  "Emory pT3-4 N0", "km_subgroup_emory_pT3_4_N0.png"
)

# ==============================================================================
# TCGA SUBGROUPS
# ==============================================================================
cat("\n\n========== TCGA SUBGROUP ANALYSES ==========\n\n")

tcga <- read.csv(file.path(save_path, "tcga_clinical_staging.csv"))
tcga$os_time_years <- tcga$os_time / 365.25
tcga$risk_group <- factor(tcga$risk_group, levels = c("Low Risk", "High Risk"))

cat("--- TCGA: pT3-4 ---\n")
make_subgroup_km(
  tcga[tcga$pT_group == "pT3-4", ],
  "TCGA pT3-4", "km_subgroup_tcga_pT3_4.png"
)

cat("\n--- TCGA: N+ ---\n")
make_subgroup_km(
  tcga[!is.na(tcga$pN_group) & tcga$pN_group == "N+", ],
  "TCGA Node-Positive", "km_subgroup_tcga_Npos.png"
)

cat("\n--- TCGA: N0 ---\n")
make_subgroup_km(
  tcga[!is.na(tcga$pN_group) & tcga$pN_group == "N0", ],
  "TCGA Node-Negative", "km_subgroup_tcga_N0.png"
)

cat("\n--- TCGA: pT3-4 N+ ---\n")
make_subgroup_km(
  tcga[tcga$pT_group == "pT3-4" & !is.na(tcga$pN_group) & tcga$pN_group == "N+", ],
  "TCGA pT3-4 N+", "km_subgroup_tcga_pT3_4_Npos.png"
)

cat("\n--- TCGA: pT3-4 N0 ---\n")
make_subgroup_km(
  tcga[tcga$pT_group == "pT3-4" & !is.na(tcga$pN_group) & tcga$pN_group == "N0", ],
  "TCGA pT3-4 N0", "km_subgroup_tcga_pT3_4_N0.png"
)

cat("\n========== ALL SUBGROUP ANALYSES COMPLETE ==========\n")
