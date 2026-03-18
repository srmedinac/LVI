options(warn = -1)
library(survival)
library(survminer)
library(ggplot2)
library(gridExtra)
library(grid)

save_path <- "/media/smedin7/data/LVI/results_2026/results"

# Load TCGA inference results
results <- read.csv(file.path(save_path, "tcga_inference_results.csv"))
results$os_time_years <- results$os_time / 365.25

save_km <- function(km_plot, filename, width = 10, height = 8) {
  if (is.null(km_plot)) return(invisible(NULL))
  plot_grob <- ggplotGrob(km_plot$plot)
  table_grob <- ggplotGrob(km_plot$table)
  combined <- arrangeGrob(plot_grob, table_grob, ncol = 1, heights = c(2, 0.5))
  ggsave(filename, plot = combined, width = width, height = height, dpi = 600)
  cat("Saved:", filename, "\n")
}

# ==============================================================================
# JOINT MODEL — Main TCGA KM (median split)
# ==============================================================================
cat("\n========== TCGA KM ANALYSES ==========\n\n")

joint <- results[results$model == "Joint", ]
cat(sprintf("Joint model: %d TCGA patients\n", nrow(joint)))

# Median split
median_risk <- median(joint$risk_score)
joint$risk_group_median <- factor(
  ifelse(joint$risk_score > median_risk, "High Risk", "Low Risk"),
  levels = c("Low Risk", "High Risk")
)

cat(sprintf("Median cutpoint: %.4f\n", median_risk))
cat(sprintf("High Risk: %d, Low Risk: %d\n",
            sum(joint$risk_group_median == "High Risk"),
            sum(joint$risk_group_median == "Low Risk")))

fit_med <- survfit(Surv(os_time_years, os_event) ~ risk_group_median, data = joint)
cox_med <- coxph(Surv(os_time_years, os_event) ~ risk_group_median, data = joint)
s_med <- summary(cox_med)

hr_med <- s_med$conf.int[1, "exp(coef)"]
lo_med <- s_med$conf.int[1, "lower .95"]
hi_med <- s_med$conf.int[1, "upper .95"]
p_med <- s_med$coefficients[1, "Pr(>|z|)"]
c_med <- s_med$concordance[1]
hr_text_med <- sprintf("HR: %.2f (95%% CI: %.2f-%.2f)", hr_med, lo_med, hi_med)

cat(sprintf("Median split: HR=%.2f (%.2f-%.2f), p=%.4f, C-index=%.3f\n",
            hr_med, lo_med, hi_med, p_med, c_med))

p_km_med <- ggsurvplot(
  fit_med, data = joint,
  pval = TRUE, conf.int = TRUE, risk.table = TRUE,
  title = sprintf("TCGA External Validation — Perivascular AI Risk (n=%d)", nrow(joint)),
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

max_x <- max(joint$os_time_years, na.rm = TRUE)
p_km_med$plot <- p_km_med$plot +
  annotate("text", x = max_x * 0.02, y = 0.15, label = hr_text_med,
           hjust = 0, size = 6.5)

save_km(p_km_med, file.path(save_path, "km_tcga_joint_median.png"))

# ==============================================================================
# JOINT MODEL — Emory training set 35th percentile cutpoint (external validation)
# ==============================================================================
cat("\n--- Emory train 35th pct cutpoint (fixed) ---\n")

cutpoint_35 <- -2.7603  # From Emory training set
joint$risk_group_35 <- factor(
  ifelse(joint$risk_score > cutpoint_35, "High Risk", "Low Risk"),
  levels = c("Low Risk", "High Risk")
)

cat(sprintf("Emory train 35th pct cutpoint: %.4f\n", cutpoint_35))
cat(sprintf("High Risk: %d, Low Risk: %d\n",
            sum(joint$risk_group_35 == "High Risk"),
            sum(joint$risk_group_35 == "Low Risk")))

fit_35 <- survfit(Surv(os_time_years, os_event) ~ risk_group_35, data = joint)
cox_35 <- coxph(Surv(os_time_years, os_event) ~ risk_group_35, data = joint)
s_35 <- summary(cox_35)

hr_35 <- s_35$conf.int[1, "exp(coef)"]
lo_35 <- s_35$conf.int[1, "lower .95"]
hi_35 <- s_35$conf.int[1, "upper .95"]
p_35 <- s_35$coefficients[1, "Pr(>|z|)"]
c_35 <- s_35$concordance[1]
hr_text_35 <- sprintf("HR: %.2f (95%% CI: %.2f-%.2f)", hr_35, lo_35, hi_35)

cat(sprintf("35th pct: HR=%.2f (%.2f-%.2f), p=%.4f, C-index=%.3f\n",
            hr_35, lo_35, hi_35, p_35, c_35))

p_km_35 <- ggsurvplot(
  fit_35, data = joint,
  pval = TRUE, conf.int = TRUE, risk.table = TRUE,
  title = sprintf("TCGA External Validation — Emory Cutpoint (n=%d)", nrow(joint)),
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

p_km_35$plot <- p_km_35$plot +
  annotate("text", x = max_x * 0.02, y = 0.15, label = hr_text_35,
           hjust = 0, size = 6.5)

save_km(p_km_35, file.path(save_path, "km_tcga_joint_35pct.png"))

# ==============================================================================
# ABLATION MODELS (median split)
# ==============================================================================
cat("\n--- Ablation models ---\n")

for (model_name in c("Survival-Only", "LVI-Only")) {
  sub <- results[results$model == model_name, ]
  med_risk <- median(sub$risk_score)
  sub$risk_group <- factor(
    ifelse(sub$risk_score > med_risk, "High Risk", "Low Risk"),
    levels = c("Low Risk", "High Risk")
  )

  fit <- survfit(Surv(os_time_years, os_event) ~ risk_group, data = sub)
  cox <- coxph(Surv(os_time_years, os_event) ~ risk_group, data = sub)
  s <- summary(cox)

  hr <- s$conf.int[1, "exp(coef)"]
  lo <- s$conf.int[1, "lower .95"]
  hi <- s$conf.int[1, "upper .95"]
  p <- s$coefficients[1, "Pr(>|z|)"]
  hr_text <- sprintf("HR: %.2f (95%% CI: %.2f-%.2f)", hr, lo, hi)

  cat(sprintf("  %s: HR=%.2f (%.2f-%.2f), p=%.4f\n", model_name, hr, lo, hi, p))

  p_km <- ggsurvplot(
    fit, data = sub,
    pval = TRUE, conf.int = TRUE, risk.table = TRUE,
    title = sprintf("TCGA — %s (n=%d)", model_name, nrow(sub)),
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

  max_x_sub <- max(sub$os_time_years, na.rm = TRUE)
  p_km$plot <- p_km$plot +
    annotate("text", x = max_x_sub * 0.02, y = 0.15, label = hr_text,
             hjust = 0, size = 6.5)

  fname <- tolower(gsub("-", "_", model_name))
  save_km(p_km, file.path(save_path, sprintf("km_tcga_%s.png", fname)))
}

# ==============================================================================
# TCGA MVA (Joint model)
# ==============================================================================
cat("\n--- TCGA Multivariable Analysis ---\n")

# Load clinical data for covariates
clinical <- read.csv("/media/smedin7/data/LVI/TCGA_clinical_data_CSV.csv")
clinical$patient_id <- clinical$bcr_patient_barcode

# Merge with predictions
tcga_mva <- merge(joint, clinical, by = "patient_id")

# Stage: extract T stage from AJCC
tcga_mva$pT_high <- as.integer(grepl("III|IV", tcga_mva$ajcc_pathologic_tumor_stage))

# Age
tcga_mva$age <- tcga_mva$age_at_initial_pathologic_diagnosis

# Sex
tcga_mva$male <- as.integer(tcga_mva$gender == "MALE")

# Risk group (Emory train cutpoint)
tcga_mva$risk_high <- as.integer(tcga_mva$risk_group_35 == "High Risk")

cat(sprintf("MVA dataset: %d patients\n", nrow(tcga_mva)))

# UVA
cat("\nUnivariate Cox:\n")
uva <- coxph(Surv(os_time_years, os_event) ~ risk_high, data = tcga_mva)
s_uva <- summary(uva)
cat(sprintf("  AI Risk: HR=%.2f (%.2f-%.2f), p=%.4f\n",
            s_uva$conf.int[1,1], s_uva$conf.int[1,3], s_uva$conf.int[1,4],
            s_uva$coefficients[1,5]))

# MVA
cat("\nMultivariate Cox:\n")
mva <- coxph(Surv(os_time_years, os_event) ~ risk_high + pT_high + age + male,
             data = tcga_mva)
s_mva <- summary(mva)
print(s_mva)

cat("\n========== TCGA ANALYSIS COMPLETE ==========\n")
