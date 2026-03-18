options(warn = -1)

# ==============================================================================
# MULTIVARIABLE COX ANALYSIS + FOREST PLOTS (EMORY TEST SET)
# ==============================================================================

required_packages <- c("survival", "survminer", "dplyr", "ggplot2", "forestmodel")

for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, dependencies = TRUE, repos = "https://cloud.r-project.org")
    library(pkg, character.only = TRUE)
  }
}

save_path <- "/media/smedin7/data/LVI/results_2026/results"

# ==============================================================================
# LOAD AND PREPARE DATA
# ==============================================================================

data <- read.csv(file.path(save_path, "emory_test_clinical_merged.csv"))
cat("Loaded", nrow(data), "test patients\n\n")

# Time in years (5-year censored)
data$os_time_years <- data$os_time / 365.25

# Risk group (median split)
median_risk <- median(data$risk_score_new)
data$risk_group <- factor(
  ifelse(data$risk_score_new > median_risk, "High Risk", "Low Risk"),
  levels = c("Low Risk", "High Risk")
)

# Clean pT → binary (pT ≤ 2 vs pT ≥ 3)
data$pT_stage <- ifelse(
  grepl("pT3|pT4", data$pT), "pT3-4", "pT0-2"
)
data$pT_stage <- factor(data$pT_stage, levels = c("pT0-2", "pT3-4"))

# Clean pN → binary (pN0 vs pN+)
data$pN_status <- ifelse(data$pN == "pN0" | data$pN == "pNX", "pN0/X", "pN+")
data$pN_status <- factor(data$pN_status, levels = c("pN0/X", "pN+"))

# Clean LVI
data$lvi_status <- ifelse(data$LVI == "Present", "Present", "Absent")
data$lvi_status <- factor(data$lvi_status, levels = c("Absent", "Present"))

# Sex
data$sex <- factor(data$Sex, levels = c("Male", "Female"))

# Race → binary (White vs Non-White, due to small numbers)
data$race <- ifelse(data$Race == "White", "White", "Non-White")
data$race <- factor(data$race, levels = c("White", "Non-White"))

# NAC
data$nac <- factor(data$NAC, levels = c(0, 1), labels = c("No", "Yes"))

# Age (continuous, per 10 years for interpretability)
data$age_10 <- data$Age / 10

cat("Variable distributions in test set:\n")
cat("  Risk group:", table(data$risk_group), "\n")
cat("  pT stage:", table(data$pT_stage), "\n")
cat("  pN status:", table(data$pN_status), "\n")
cat("  LVI:", table(data$lvi_status), "\n")
cat("  Sex:", table(data$sex), "\n")
cat("  Race:", table(data$race), "\n")
cat("  NAC:", table(data$nac), "\n")
cat("  Age: mean", round(mean(data$Age), 1), "SD", round(sd(data$Age), 1), "\n\n")

# ==============================================================================
# UNIVARIABLE ANALYSIS
# ==============================================================================

cat("========== UNIVARIABLE COX REGRESSION ==========\n\n")

univar_vars <- list(
  "AI Risk Group" = "risk_group",
  "pT Stage (3-4 vs 0-2)" = "pT_stage",
  "pN Status (N+ vs N0)" = "pN_status",
  "LVI (Present vs Absent)" = "lvi_status",
  "Age (per 10 years)" = "age_10",
  "Sex (Female vs Male)" = "sex",
  "Race (Non-White vs White)" = "race",
  "NAC (Yes vs No)" = "nac"
)

for (name in names(univar_vars)) {
  var <- univar_vars[[name]]
  formula <- as.formula(paste("Surv(os_time_years, os_event) ~", var))
  fit <- coxph(formula, data = data)
  s <- summary(fit)
  hr <- s$conf.int[1, "exp(coef)"]
  lo <- s$conf.int[1, "lower .95"]
  hi <- s$conf.int[1, "upper .95"]
  p <- s$coefficients[1, "Pr(>|z|)"]
  cat(sprintf("  %-35s HR=%.2f (%.2f-%.2f), p=%.4f\n", name, hr, lo, hi, p))
}

# ==============================================================================
# MULTIVARIABLE ANALYSIS
# ==============================================================================

cat("\n========== MULTIVARIABLE COX REGRESSION ==========\n\n")

mva_fit <- coxph(
  Surv(os_time_years, os_event) ~ risk_group + pT_stage + pN_status +
    lvi_status + age_10 + sex + nac,
  data = data
)

mva_summary <- summary(mva_fit)
print(mva_summary)

cat("\nC-index:", round(mva_summary$concordance[1], 3), "\n")

# ==============================================================================
# FOREST PLOT — MULTIVARIABLE
# ==============================================================================

cat("\n========== GENERATING FOREST PLOTS ==========\n\n")

# Try forestmodel first, fall back to manual
tryCatch({
  p_forest <- forest_model(
    mva_fit,
    format_options = forest_model_format_options(
      text_size = 4,
      point_size = 3,
      banded = TRUE
    )
  ) +
    theme_minimal(base_size = 14) +
    labs(title = "Multivariable Cox Regression — Emory Test Set") +
    theme(
      plot.title = element_text(face = "bold", size = 16, hjust = 0),
      plot.margin = margin(10, 20, 10, 10)
    )

  ggsave(
    file.path(save_path, "forest_mva_emory_test.png"),
    plot = p_forest, width = 12, height = 6, dpi = 600
  )
  cat("Saved: forest_mva_emory_test.png\n")
}, error = function(e) {
  cat("forestmodel failed:", conditionMessage(e), "\n")
  cat("Generating manual forest plot...\n")

  # Manual forest plot
  coef_df <- as.data.frame(mva_summary$conf.int)
  coef_df$variable <- rownames(coef_df)
  coef_df$p <- mva_summary$coefficients[, "Pr(>|z|)"]
  coef_df$label <- sprintf("%.2f (%.2f-%.2f)", coef_df[,1], coef_df[,3], coef_df[,4])

  coef_df$variable <- gsub("risk_groupHigh Risk", "AI Risk (High vs Low)", coef_df$variable)
  coef_df$variable <- gsub("pT_stagepT3-4", "pT Stage (3-4 vs 0-2)", coef_df$variable)
  coef_df$variable <- gsub("pN_statuspN\\+", "pN Status (N+ vs N0)", coef_df$variable)
  coef_df$variable <- gsub("lvi_statusPresent", "LVI (Present vs Absent)", coef_df$variable)
  coef_df$variable <- gsub("age_10", "Age (per 10 years)", coef_df$variable)
  coef_df$variable <- gsub("sexFemale", "Sex (Female vs Male)", coef_df$variable)
  coef_df$variable <- gsub("nacYes", "NAC (Yes vs No)", coef_df$variable)

  coef_df$variable <- factor(coef_df$variable, levels = rev(coef_df$variable))

  p <- ggplot(coef_df, aes(x = `exp(coef)`, y = variable)) +
    geom_point(size = 3) +
    geom_errorbarh(aes(xmin = `lower .95`, xmax = `upper .95`), height = 0.2) +
    geom_vline(xintercept = 1, linetype = "dashed", color = "grey50") +
    geom_text(aes(label = label, x = max(`upper .95`) * 1.1), hjust = 0, size = 3.5) +
    scale_x_log10() +
    labs(x = "Hazard Ratio (95% CI)", y = "",
         title = "Multivariable Cox Regression — Emory Test Set (n=82)") +
    theme_minimal(base_size = 14) +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      axis.text.y = element_text(size = 12),
      panel.grid.minor = element_blank()
    )

  ggsave(
    file.path(save_path, "forest_mva_emory_test.png"),
    plot = p, width = 12, height = 5, dpi = 600
  )
  cat("Saved: forest_mva_emory_test.png\n")
})

# ==============================================================================
# OPTIMAL CUTPOINT (MAXSTAT)
# ==============================================================================

cat("\n========== OPTIMAL CUTPOINT (MAXSTAT) ==========\n\n")

tryCatch({
  library(maxstat)
  ms <- maxstat.test(
    Surv(os_time_years, os_event) ~ risk_score_new,
    data = data, smethod = "LogRank"
  )
  cat("Optimal cutpoint:", ms$estimate, "\n")
  cat("Maxstat p-value:", format.pval(ms$p.value, digits = 4), "\n")

  # Re-stratify with optimal cutpoint
  data$risk_optimal <- factor(
    ifelse(data$risk_score_new > ms$estimate, "High Risk", "Low Risk"),
    levels = c("Low Risk", "High Risk")
  )

  fit_opt <- coxph(Surv(os_time_years, os_event) ~ risk_optimal, data = data)
  s_opt <- summary(fit_opt)
  hr <- s_opt$conf.int[1, "exp(coef)"]
  lo <- s_opt$conf.int[1, "lower .95"]
  hi <- s_opt$conf.int[1, "upper .95"]
  p <- s_opt$coefficients[1, "Pr(>|z|)"]
  ci <- s_opt$concordance[1]
  cat(sprintf("Optimal cutpoint HR: %.2f (%.2f-%.2f), p=%.4f, C-index=%.3f\n", hr, lo, hi, p, ci))
  cat("Group sizes:", table(data$risk_optimal), "\n")

  # KM plot with optimal cutpoint
  km_opt <- survfit(Surv(os_time_years, os_event) ~ risk_optimal, data = data)

  p_km <- ggsurvplot(
    km_opt, data = data,
    pval = TRUE, conf.int = TRUE, risk.table = TRUE,
    title = paste0("Joint Model — Optimal Cutpoint (Emory Test, n=", nrow(data), ")"),
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

  hr_text <- sprintf("HR: %.2f (95%% CI: %.2f-%.2f)", hr, lo, hi)
  max_x <- max(data$os_time_years, na.rm = TRUE)
  p_km$plot <- p_km$plot +
    annotate("text", x = max_x * 0.02, y = 0.15, label = hr_text,
             hjust = 0, size = 6.5)

  # Save using the same method as km_plots.R
  plot_grob <- ggplotGrob(p_km$plot)
  table_grob <- ggplotGrob(p_km$table)
  combined <- gridExtra::arrangeGrob(plot_grob, table_grob, ncol = 1, heights = c(2, 0.5))
  ggsave(
    file.path(save_path, "km_joint_test_optimal_cutpoint.png"),
    plot = combined, width = 10, height = 8, dpi = 600
  )
  cat("Saved: km_joint_test_optimal_cutpoint.png\n")

}, error = function(e) {
  cat("maxstat failed:", conditionMessage(e), "\n")
  cat("Install with: install.packages('maxstat')\n")
})

cat("\n========== ANALYSIS COMPLETE ==========\n")
