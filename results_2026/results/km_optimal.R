options(warn = -1)
library(survival)
library(survminer)
library(dplyr)
library(ggplot2)
library(gridExtra)
library(grid)

save_path <- "/media/smedin7/data/LVI/results_2026/results"

# ==============================================================================
# LOAD DATA
# ==============================================================================

data <- read.csv(file.path(save_path, "emory_test_clinical_merged.csv"))
data$os_time_years <- data$os_time / 365.25

# 35th percentile cutpoint
CUTPOINT <- quantile(data$risk_score_new, 0.35)
cat("Cutpoint (35th percentile):", CUTPOINT, "\n\n")

data$risk_group <- factor(
  ifelse(data$risk_score_new > CUTPOINT, "High Risk", "Low Risk"),
  levels = c("Low Risk", "High Risk")
)

cat("Groups:\n")
tab <- table(data$risk_group)
ev <- tapply(data$os_event, data$risk_group, sum)
print(data.frame(group = names(tab), n = as.integer(tab), events = as.integer(ev[names(tab)])))

# Clinical variables
data$pT_stage <- factor(ifelse(grepl("pT3|pT4", data$pT), "pT3-4", "pT0-2"), levels = c("pT0-2", "pT3-4"))
data$pN_status <- factor(ifelse(data$pN %in% c("pN0", "pNX"), "pN0/X", "pN+"), levels = c("pN0/X", "pN+"))
data$lvi_status <- factor(ifelse(data$LVI == "Present", "Present", "Absent"), levels = c("Absent", "Present"))
data$sex <- factor(data$Sex, levels = c("Male", "Female"))
data$nac <- factor(data$NAC, levels = c(0, 1), labels = c("No", "Yes"))
data$age_10 <- data$Age / 10

# ==============================================================================
# UNIVARIABLE
# ==============================================================================

cat("\n--- Univariable ---\n")
fit_uva <- coxph(Surv(os_time_years, os_event) ~ risk_group, data = data)
s <- summary(fit_uva)
hr <- s$conf.int[1, "exp(coef)"]
lo <- s$conf.int[1, "lower .95"]
hi <- s$conf.int[1, "upper .95"]
p <- s$coefficients[1, "Pr(>|z|)"]
ci <- s$concordance[1]
cat(sprintf("HR=%.2f (%.2f-%.2f), p=%.4f, C-index=%.3f\n", hr, lo, hi, p, ci))

# ==============================================================================
# MULTIVARIABLE
# ==============================================================================

cat("\n--- Multivariable ---\n")
fit_mva <- coxph(
  Surv(os_time_years, os_event) ~ risk_group + pT_stage + pN_status +
    lvi_status + age_10 + sex + nac,
  data = data
)
s2 <- summary(fit_mva)
hr2 <- s2$conf.int[1, "exp(coef)"]
lo2 <- s2$conf.int[1, "lower .95"]
hi2 <- s2$conf.int[1, "upper .95"]
p2 <- s2$coefficients[1, "Pr(>|z|)"]
cat(sprintf("HR=%.2f (%.2f-%.2f), p=%.4f, C-index=%.3f\n", hr2, lo2, hi2, p2, s2$concordance[1]))

# ==============================================================================
# KM PLOT — UNIVARIABLE (main figure)
# ==============================================================================

hr_text <- sprintf("HR: %.2f (95%% CI: %.2f-%.2f)", hr, lo, hi)

km_fit <- survfit(Surv(os_time_years, os_event) ~ risk_group, data = data)

p_km <- ggsurvplot(
  km_fit, data = data,
  pval = TRUE, conf.int = TRUE, risk.table = TRUE,
  title = paste0("Perivascular AI Risk — Emory Test Set (n=", nrow(data), ")"),
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

max_x <- max(data$os_time_years, na.rm = TRUE)
p_km$plot <- p_km$plot +
  annotate("text", x = max_x * 0.02, y = 0.15, label = hr_text,
           hjust = 0, size = 6.5)

plot_grob <- ggplotGrob(p_km$plot)
table_grob <- ggplotGrob(p_km$table)
combined <- arrangeGrob(plot_grob, table_grob, ncol = 1, heights = c(2, 0.5))
ggsave(file.path(save_path, "km_emory_test_35pct.png"),
       plot = combined, width = 10, height = 8, dpi = 600)
cat("\nSaved: km_emory_test_35pct.png\n")

# ==============================================================================
# FOREST PLOT — MVA
# ==============================================================================

library(forestmodel)

p_forest <- forest_model(
  fit_mva,
  format_options = forest_model_format_options(
    text_size = 4, point_size = 3, banded = TRUE
  )
) +
  theme_minimal(base_size = 14) +
  labs(title = "Multivariable Cox Regression — Emory Test Set (n=82)") +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0),
    plot.margin = margin(10, 20, 10, 10)
  )

ggsave(file.path(save_path, "forest_mva_35pct.png"),
       plot = p_forest, width = 12, height = 6, dpi = 600)
cat("Saved: forest_mva_35pct.png\n")

cat("\n========== DONE ==========\n")
