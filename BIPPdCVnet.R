library(dplyr) 
library(tidyverse)
library(missForest)
library(psych)
library(lavaan)
library(dCVnet)
library(patchwork)
library(corrplot)

#Load objects----------------------------------------------------------

dt<-read.csv("psypred.csv")

#Missing values imputation
set.seed(42)
dt_imp <- missForest(dt)
dt<-dt_imp$ximp
save(dt, file="psypred.RData")

# Data adjustments-----------------------------------------------------
load("psypred.RData")

# outcome data:
ovars <- c("srs8",
           "sdq8int", "sdq8ext", 
           "TMCQ.SU", "TMCQ.EC", "TMCQ.NA",
           "scas8_total")
odat <- dt %>% select(one_of(ovars))

# Control Variables: extract and center 
# Control Variable for outcomes:
ocon <- dt %>% 
  select(Age8) %>% 
  mutate_all(scale, center = TRUE, scale = FALSE)

# Control Variable for predictors:
pcon <- dt %>% 
  select(Age2, Age4, Sex, Ethnicity, low_mat_edu, IMD, GA) %>% 
  mutate_all(scale, center = TRUE, scale = FALSE)

# predictor data:
pdat <- dt %>% 
  select(!all_of(ovars)) %>% 
  select(!all_of(colnames(ocon))) %>% 
  select(!all_of(colnames(pcon))) %>% 
  select(-Mother_Edu)

## Outcome dimension reduction ####
# Inspect bivariate relationships:
odat %>% 
  car::scatterplotMatrix()

# correlation matrix ('X' -> p>0.05 unadjusted):
odat %>% 
  cor() %>% 
  ggcorrplot::ggcorrplot(p.mat = ggcorrplot::cor_pmat(odat))

# Dimension reduction using Parallel Analysis
psych::fa.parallel(odat, fm = "pa")
# suggests two factors.

# efa in lavaan 
oefa <- lavaan::efa(data = map_dfc(odat, scale),
                    rotation = "oblimin",
                    nfactors = 1:3)

summary(oefa) # agrees the two factor result
summary(oefa[[2]])

# Solution identifies:
#   f1: Externalizing + Surgency + lack of Effortful Control + Negative affect
#   f2: Internalizing + SRS + Negative affect + lack of Surgency + Anxiety

# Extract scores for two factors:
odat_efa <- lavPredict(oefa[[2]]) %>% as.data.frame()

# check correlations:
ggcorrplot::ggcorrplot(cor(cbind(odat, odat_efa)))

# Figure 2: Plot heatmaps for loadings
loading <- matrix(c(0, 0.548,
                    0, 0.677,
                    0.730, 0,
                    0.438, -0.465,
                    -0.761,0,
                    0.317, 0.594,
                    0, 0.932), nrow = 7, ncol = 2, byrow = TRUE)

colnames(loading) <- c("Externalising", "Internalising")
rownames(loading) <- c("Autistic symptoms", "Internalising symptoms",
                       "Externalising symptoms", "Surgency", "Effortful Control", 
                       "Negative affectivity", "Anxiety symptoms")

col <- colorRampPalette(c("blue", "white", "red"))(100)
corrplot(loading, is.corr=F, method="color", col=col,
         tl.col="black",tl.cex=1,  
         cl.align="l", cl.ratio = 1,
         na.label="square",na.label.col="white",
         addgrid.col="grey")

## Control Variables ####
# outcome adjustment
# iterate through outcome variables:
# raw outcomes (7 variables)
odat_adj <- map_dfc(odat,
                    ~ {mod <- lm(y ~ ., data = data.frame(y = .x, ocon))
                      r <- resid(mod)
                      r + coef(mod)[[1]]}
)

# factor outcomes (f1 and f2)
odat_efa_adj <- map_dfc(odat_efa,
                        ~ {mod <- lm(y ~ ., data = data.frame(y = .x, ocon))
                          r <- resid(mod)
                          r + coef(mod)[[1]]}
)

# predictor adjustment
# Always adjust for Sex, Ethnicity, GA, IMD and low_mat_edu
# maternal anxiety (STAI) and cognitive stimulating parenting are not adjusted for age.

# set adjustment for age at assessment 2 as default:
preds_metadata <- data.frame(pred = colnames(pdat),type = "Age4")

# STAI and cogstimpareting should not be adjusted:
preds_metadata$type[colnames(pdat) %in% c("STAI", 
                                          "cogstimparenting4_raw")] <- "none"

# Bayley, PARCA, Mchat adjusted by age at assessment 1
preds_metadata$type[starts_with("BY", vars = colnames(pdat))] <- "Age2"
preds_metadata$type[starts_with("PR", vars = colnames(pdat))] <- "Age2"
preds_metadata$type[starts_with("Mchat", vars = colnames(pdat))] <- "Age2"

# Predictors adjusted for age at assessment 2: WPPSI, BRIEF, card sort, digit span, track it,
# Attention Network Task, emotion recognition and empathy
pdat_adj <- map2_dfc(pdat,
                     preds_metadata$type,
                     ~ {
                       cvars <- switch(
                         .y,
                         none = pcon[, !(colnames(pcon) %in% c("Age2", "Age4"))],
                         Age2 = pcon[, !(colnames(pcon) %in% c("Age4"))],
                         Age4 = pcon[, !(colnames(pcon) %in% c("Age2"))]
                       )
                       # to debug / check:
                       cat(paste0("cvar_ncol: ", ncol(cvars), "\n"))
                       # fit model:
                       mod <- lm(y ~ ., data = data.frame(y = .x, cvars))
                       # extract residuals adjusting for CV
                       r <- resid(mod)
                       # add back the intercept == mean score:
                       r + coef(mod)[[1]]
                     }
)

# Write out data####

save(dt,
  # outcome variables with and without adjustment for control variables
  odat, odat_adj, 
  # Factors derived from outcome variables with and without adjustment for
  #    control variables:
  odat_efa, odat_efa_adj,
  # predictor variables with and without adjustment for control variables:
  pdat, pdat_adj,
  # control variables for reference:
  ocon, pcon,
  # Write above as compressed R objects to following file:
  file = "dCVnet_prepped.RData")


# dCVnet analysis start####
load("data/dCVnet_prepped.RData")
## data description###
# # dt: the raw dataset
# # odat, odat_adj: 7 outcome variables with/without adjustment
# # odat_efa, odat_efa_adj: 2 representative outcome factors with/without adjustment
# # pdat, pdat_adj: predictors with/without adjustment
# # ocon, pcon: control variables for reference

## model strategy####
# # outcome:
#   - 2 separate models for odat_efa_adj
# # predictors:
#   - All predictors available in pdat_adj

## model parameters settings####
#     - k = 10-fold
#     - lambda = 1000.
#     - alpha = 0.5

# model running####
# externalising only:
m_f1 <- dCVnet(
  y = odat_efa_adj$f1,
  data = pdat_adj,
  family = "gaussian",
  alphalist = 0.2,
  nlambda = 100,
  nrep_inner = 30,
  nrep_outer = 100
)
save(m_f1, file = "dCVnet_m_f1.RData")

# internalising only:
m_f2 <- dCVnet(
  y = odat_efa_adj$f2,
  data = pdat_adj,
  family = "gaussian",
  alphalist = 0.2,
  nlambda = 100,
  nrep_inner = 30,
  nrep_outer = 100
)
save(m_f2, file = "dCVnet_m_f2.RData")

# Results output####
theme_set(theme_light())

# #load models:
#   f1, all predictors:
load("dCVnet_m_f1.RData")
 
#   f2, all predictors:
load("dCVnet_m_f2.RData")

# # Inspect prediction performance 

# extract cross-validated prediction performance and make tidy data for plotting:

# Helper functions:
tidy_gaussian_output <- function(obj, outcome_label) {
  obj %>% 
    pivot_longer(starts_with("Rep")) %>% 
    mutate(outcome = outcome_label, modeltype = "separate")
}

# assemble tidy performance data:
pps <- list(
  # outcome f1 :
  m_f1 = m_f1 %>% 
    performance() %>% 
    summary() %>% 
    tidy_gaussian_output("f1"),
  # outcome f2 :
  m_f2 = m_f2 %>% 
    performance() %>% 
    summary() %>% 
    tidy_gaussian_output("f2"))

  # merge into single data.frame:
  data.table::rbindlist(use.names = TRUE) %>% 
  as.data.frame()

modelindex<-pps %>% 
    # Select model performance measures:
    filter(Measure %in% c("RMSE", "MAE", "r2")) %>%
    # Set factor names for measure:
    mutate(Measure = factor(Measure, levels = c("RMSE", "MAE", "r2"))) %>%
    # Set factor names for the prediction model outcome:
    mutate(outcome = factor(outcome,
                            levels = c("f1", "f2"),
                            labels = c("f1\nexternalising",
                                       "f2\ninternalising")))
  f1index<-subset.data.frame(modelindex,modelindex$outcome == "f1\nexternalising")
  f2index<-subset.data.frame(modelindex,modelindex$outcome == "f2\ninternalising")
  
  f1_RMSE<-mean(f1index$value[f1index$Measure == "RMSE"])
  f1_R2<-mean(f1index$value[f1index$Measure == "r2"])
  f1_MAE<-mean(f1index$value[f1index$Measure == "MAE"])
  
  f2_RMSE<-mean(f2index$value[f2index$Measure == "RMSE"])
  f2_R2<-mean(f2index$value[f2index$Measure == "r2"])
  f2_MAE<-mean(f2index$value[f2index$Measure == "MAE"])

# Inspect coefficients ----

# Merge and format two models:
cpd <- list(
  f1 = m_f1 %>%
    coefficients_summary(m_f1) %>%
    select(Predictor, ProductionModel) %>%
    mutate(Outcome = "f1"),
  f2 = m_f2 %>%
    coefficients_summary(m_f1) %>%
    select(Predictor, ProductionModel) %>%
    mutate(Outcome = "f2")
) %>% do.call(rbind, .) %>%
  pivot_wider(names_from = Outcome, values_from = ProductionModel)

knitr::kable(cpd) %>% gsub("0.0000000", "        -", .)

