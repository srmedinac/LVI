options(warn = -1)
library(survival)
library(survminer)
library(ggplot2)
library(gridExtra)
library(grid)

save_path <- "/media/smedin7/data/LVI/results_2026/results"
data <- read.csv(file.path(save_path, "emory_test_clinical_merged.csv"))
data$os_time_years <- data$os_time / 365.25

CUTPOINT <- quantile(data$risk_score_new, 0.35)
data$risk_group <- factor(
  ifelse(data$risk_score_new > CUTPOINT, "High Risk", "Low Risk"),
  levels = c("Low Risk", "High Risk")
)
data$pT_stage <- ifelse(grepl("pT3|pT4", data$pT), "pT3-4", "pT0-2")
data$lvi_status <- ifelse(data$LVI == "Present", "LVI+", "LVI-")

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
# SUBGROUP ANALYSES
# ==============================================================================

cat("\n========== SUBGROUP KM ANALYSES ==========\n\n")

# 1. LVI-Negative only (key clinical message!)
cat("--- LVI-Negative subgroup ---\n")
lvi_neg <- data[data$lvi_status == "LVI-", ]
make_subgroup_km(lvi_neg, "LVI-Negative Patients", "km_subgroup_lvi_negative.png")

# 2. LVI-Positive only
cat("\n--- LVI-Positive subgroup ---\n")
lvi_pos <- data[data$lvi_status == "LVI+", ]
make_subgroup_km(lvi_pos, "LVI-Positive Patients", "km_subgroup_lvi_positive.png")

# 3. pT0-2 only
cat("\n--- pT0-2 subgroup ---\n")
pt_low <- data[data$pT_stage == "pT0-2", ]
make_subgroup_km(pt_low, "pT0-2 Patients", "km_subgroup_pT0_2.png")

# 4. pT3-4 only
cat("\n--- pT3-4 subgroup ---\n")
pt_high <- data[data$pT_stage == "pT3-4", ]
make_subgroup_km(pt_high, "pT3-4 Patients", "km_subgroup_pT3_4.png")

# 5. NAC vs no NAC
cat("\n--- No NAC subgroup ---\n")
no_nac <- data[data$NAC == 0, ]
make_subgroup_km(no_nac, "No Neoadjuvant Chemo", "km_subgroup_no_nac.png")

cat("\n========== SUBGROUP ANALYSIS COMPLETE ==========\n")
