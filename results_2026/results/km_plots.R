options(warn = -1)

# ==============================================================================
# LVI PREDICTION - KAPLAN-MEIER ANALYSIS (EMORY TEST SET)
# ==============================================================================

required_packages <- c("survival", "survminer", "dplyr", "ggplot2", "gridExtra", "grid")

for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
    library(pkg, character.only = TRUE)
  }
}

# ==============================================================================
# CONFIGURATION
# ==============================================================================

save_path <- "/media/smedin7/data/LVI/results_2026/results"

# ==============================================================================
# PLOTTING FUNCTIONS
# ==============================================================================

create_km_plot_os <- function(data, title, subgroup_label = NULL,
                               legend_labs = c("Low Risk", "High Risk"),
                               palette = c("#00BFC4", "#F8766D")) {
  fit <- survfit(Surv(os_time_years, os_event) ~ risk_group, data = data)
  cox_temp <- coxph(Surv(os_time_years, os_event) ~ risk_group, data = data)
  cox_summary <- summary(cox_temp)

  hr <- cox_summary$conf.int[1, "exp(coef)"]
  hr_lower <- cox_summary$conf.int[1, "lower .95"]
  hr_upper <- cox_summary$conf.int[1, "upper .95"]
  p_value <- cox_summary$coefficients[1, "Pr(>|z|)"]
  hr_text <- sprintf("HR: %.2f (95%% CI: %.2f-%.2f)", hr, hr_lower, hr_upper)

  p <- ggsurvplot(
    fit, data = data,
    pval = TRUE, conf.int = TRUE, risk.table = TRUE,
    title = title,
    conf.int.style = "ribbon",
    xlab = "Time (years)", ylab = "Overall Survival",
    legend = "top",
    legend.title = "Group", legend.labs = legend_labs,
    palette = palette,
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

  # Add HR annotation
  max_x <- max(data$os_time_years, na.rm = TRUE)
  p$plot <- p$plot +
    annotate("text", x = max_x * 0.02, y = 0.15, label = hr_text,
             hjust = 0, size = 6.5)

  return(p)
}

create_km_plot_lvi <- function(data, title,
                                palette = c("#00BFC4", "#F8766D")) {
  fit <- survfit(Surv(os_time_years, os_event) ~ lvi_predicted, data = data)
  cox_temp <- coxph(Surv(os_time_years, os_event) ~ lvi_predicted, data = data)
  cox_summary <- summary(cox_temp)

  hr <- cox_summary$conf.int[1, "exp(coef)"]
  hr_lower <- cox_summary$conf.int[1, "lower .95"]
  hr_upper <- cox_summary$conf.int[1, "upper .95"]
  hr_text <- sprintf("HR: %.2f (95%% CI: %.2f-%.2f)", hr, hr_lower, hr_upper)

  p <- ggsurvplot(
    fit, data = data,
    pval = TRUE, conf.int = TRUE, risk.table = TRUE,
    title = title,
    conf.int.style = "ribbon",
    xlab = "Time (years)", ylab = "Overall Survival",
    legend = "top",
    legend.title = "Predicted LVI",
    legend.labs = c("LVI-Negative", "LVI-Positive"),
    palette = palette,
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

  max_x <- max(data$os_time_years, na.rm = TRUE)
  p$plot <- p$plot +
    annotate("text", x = max_x * 0.02, y = 0.15, label = hr_text,
             hjust = 0, size = 6.5)

  return(p)
}

save_km_plot <- function(km_plot, filename, width = 10, height = 8) {
  if (is.null(km_plot)) return(invisible(NULL))
  plot_grob <- ggplotGrob(km_plot$plot)
  table_grob <- ggplotGrob(km_plot$table)
  combined_plot <- arrangeGrob(plot_grob, table_grob, ncol = 1, heights = c(2, 0.5))
  ggsave(filename, plot = combined_plot, width = width, height = height, dpi = 600)
  cat("Saved:", filename, "\n")
}

print_km_stats <- function(data, label, group_var = "risk_group") {
  cat("\n--- KM/Cox summary:", label, "---\n")

  tab <- table(data[[group_var]])
  ev <- tapply(data$os_event, data[[group_var]], sum)
  print(data.frame(group = names(tab), n = as.integer(tab),
                   events = as.integer(ev[names(tab)])))

  formula <- as.formula(paste("Surv(os_time_years, os_event) ~", group_var))
  cfit <- coxph(formula, data = data)
  cs <- summary(cfit)
  hr <- cs$conf.int[1, "exp(coef)"]
  l <- cs$conf.int[1, "lower .95"]
  u <- cs$conf.int[1, "upper .95"]
  p <- cs$coefficients[1, "Pr(>|z|)"]
  cidx <- cs$concordance[1]

  cat(sprintf("HR (High vs Low): %.2f (95%% CI %.2f-%.2f), p=%.4f\n", hr, l, u, p))
  cat(sprintf("Harrell C-index: %.3f\n", cidx))
}

# ==============================================================================
# LOAD DATA
# ==============================================================================

cat("\n========== LOADING PREDICTIONS ==========\n\n")

models <- c("joint", "survival_only", "lvi_only")
model_labels <- c("Joint Model", "Survival-Only Model", "LVI-Only Model")

for (i in seq_along(models)) {
  model <- models[i]
  label <- model_labels[i]

  pred_file <- file.path(save_path, paste0(model, "_test_predictions.csv"))
  if (!file.exists(pred_file)) {
    cat("Skipping", label, "- file not found\n")
    next
  }

  data <- read.csv(pred_file)
  cat(label, "- loaded", nrow(data), "test patients\n")

  # Convert time from days to years
  data$os_time_years <- data$os_time / 365.25

  # Risk stratification by median risk score
  median_risk <- median(data$risk_score)
  data$risk_group <- factor(
    ifelse(data$risk_score > median_risk, "High Risk", "Low Risk"),
    levels = c("Low Risk", "High Risk")
  )

  # LVI prediction (median split — model probs don't cross 0.5)
  median_lvi <- median(data$lvi_prob)
  data$lvi_predicted <- factor(
    ifelse(data$lvi_prob > median_lvi, "LVI-Positive", "LVI-Negative"),
    levels = c("LVI-Negative", "LVI-Positive")
  )

  # Ground truth LVI stratification
  data$lvi_status <- factor(
    ifelse(data$lvi_true == 1, "LVI-Positive", "LVI-Negative"),
    levels = c("LVI-Negative", "LVI-Positive")
  )

  # ── Risk Score KM ──────────────────────────────────────────────────────
  cat("\n")
  print_km_stats(data, paste(label, "- Risk Score Stratification"))

  km_risk <- create_km_plot_os(
    data,
    title = paste0(label, " - Emory Test Set (n=", nrow(data), ")")
  )

  save_km_plot(km_risk,
               file.path(save_path, paste0("km_", model, "_test_risk.png")))

  # ── Predicted LVI KM ──────────────────────────────────────────────────
  if (nlevels(droplevels(data$lvi_predicted)) == 2) {
    print_km_stats(data, paste(label, "- Predicted LVI Stratification"),
                   group_var = "lvi_predicted")

    km_lvi_pred <- create_km_plot_lvi(
      data,
      title = paste0(label, " - Predicted LVI (Emory Test, n=", nrow(data), ")")
    )

    save_km_plot(km_lvi_pred,
                 file.path(save_path, paste0("km_", model, "_test_lvi_predicted.png")))
  } else {
    cat("Skipping LVI predicted KM for", label, "- only one group\n")
  }

  # ── Ground Truth LVI KM (only for Joint which has best LVI pred) ─────
  if (model == "joint") {
    print_km_stats(data, "Ground Truth LVI Stratification",
                   group_var = "lvi_status")

    fit_gt <- survfit(Surv(os_time_years, os_event) ~ lvi_status, data = data)
    cox_gt <- coxph(Surv(os_time_years, os_event) ~ lvi_status, data = data)
    cs_gt <- summary(cox_gt)
    hr_gt <- cs_gt$conf.int[1, "exp(coef)"]
    hr_gt_l <- cs_gt$conf.int[1, "lower .95"]
    hr_gt_u <- cs_gt$conf.int[1, "upper .95"]
    hr_gt_text <- sprintf("HR: %.2f (95%% CI: %.2f-%.2f)", hr_gt, hr_gt_l, hr_gt_u)

    km_gt <- ggsurvplot(
      fit_gt, data = data,
      pval = TRUE, conf.int = TRUE, risk.table = TRUE,
      title = paste0("Ground Truth LVI - Emory Test Set (n=", nrow(data), ")"),
      conf.int.style = "ribbon",
      xlab = "Time (years)", ylab = "Overall Survival",
      legend = "top",
      legend.title = "Ground Truth LVI",
      legend.labs = c("LVI-Negative", "LVI-Positive"),
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

    max_x <- max(data$os_time_years, na.rm = TRUE)
    km_gt$plot <- km_gt$plot +
      annotate("text", x = max_x * 0.02, y = 0.15, label = hr_gt_text,
               hjust = 0, size = 6.5)

    save_km_plot(km_gt,
                 file.path(save_path, "km_ground_truth_lvi_test.png"))
  }

  cat("\n")
}

# ==============================================================================
# JOINT MODEL - TRAIN SET
# ==============================================================================

cat("\n========== JOINT MODEL - TRAIN SET ==========\n\n")

train_file <- file.path(save_path, "joint_train_predictions.csv")
if (file.exists(train_file)) {
  train_data <- read.csv(train_file)
  train_data$os_time_years <- train_data$os_time / 365.25
  median_risk_train <- median(train_data$risk_score)
  train_data$risk_group <- factor(
    ifelse(train_data$risk_score > median_risk_train, "High Risk", "Low Risk"),
    levels = c("Low Risk", "High Risk")
  )

  print_km_stats(train_data, "Joint Model - Train Set")

  km_train <- create_km_plot_os(
    train_data,
    title = paste0("Joint Model - Emory Train Set (n=", nrow(train_data), ")")
  )

  save_km_plot(km_train,
               file.path(save_path, "km_joint_train_risk.png"))
}

cat("\n========== KM ANALYSIS COMPLETE ==========\n\n")
cat("All plots saved to:", save_path, "\n\n")
