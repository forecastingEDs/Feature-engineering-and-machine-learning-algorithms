### To use the modeltime package in Rstudio it is necessary to load some Python functions (we use Rstudio loading some functions that are from Python). Run step 1.
### step 1
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
remove.packages("modeltime", lib="~/R/win-library/4.1") # remove the package, if installed.
install.packages("modeltime", dependencies = TRUE)
remotes::install_github("business-science/modeltime", dependencies = TRUE)
# Enter one or more numbers, or an empty line to skip updates: 2 # select option 2: CRAN packages only #

### step 2
### Install and load the following R packages ----
library(recipes)
library(tune)
library(keras)
library(ggplot2)
library(modeltime.ensemble)
library(modeltime)
library(lubridate)
library(modeltime.resample)
library(tidyquant)
library(yardstick)
library(plotly)
library(xgboost)
library(rsample)
library(targets)
library(tidymodels)
library(modeltime)
library(modeltime.resample)
library(timetk)
library(tidyverse)
library(tidyquant)
library(LiblineaR)
library(parsnip)
library(ranger)
library(readxl)
library(lifecycle)
library(bonsai)
library(lightgbm)
library(treesnip)
library(rio)
library(colino)
library(doParallel)
library(RcppParallel)
library(FSelectorRcpp)
library(catboost)

# Parallel processing to speed up grid search 
doParallel::registerDoParallel(cores=20)
parallel_stop()

### Load Data ----
### In this step you must import the data from the Excel spreadsheet to Rstudio keeping the original file name "atends_temperature_calendar" so that the next steps work correctly.
With Rstudio open, in the upper right corner you have the option/tab >import Dataset>from excel>Browse and select the "datasets"


# Splitting datasets into time series cross-validation training and testing ----

data_tbl <- datasets %>%
  select(id, Date, attendences, average_temperature, min, max,  sunday, monday, tuesday, wednesday, thursday, friday, saturday, Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec) %>%
  set_names(c("id", "date", "value","tempe_verage", "tempemin", "tempemax", "sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"))

data_tbl


# Selecting one ("arma") from the 11 datasets ----
### Split the dataset into analyis/assessment set
emergency_tscv <- data_tbl %>% filter(id == "arma") %>%
  time_series_cv (
    assess     = "7 days",
    skip        = "7 days",
    cumulative = TRUE,
    slice_limit = 5
  ) 

emergency_tscv

emergency_tscv  %>%  tk_time_series_cv_plan ()

# View time series in training and testing split ----
emergency_tscv %>%
  tk_time_series_cv_plan() %>%
  plot_time_series_cv_plan(date, value, .interactive = F ,
                           .title = "arma")
emergency_tscv %>%
  tk_time_series_cv_plan() %>%
  plot_time_series_cv_plan(date, value, 
                           .facet_ncol = 2, .interactive = F)


####  Feature selection step using a random forest ---- top_p = 10, or threshold = 0.70 ----

recipe_spec <- recipe(value ~ ., 
                      data = training(emergency_tscv$splits[[1]])) %>%
  step_timeseries_signature(date) %>%
  step_rm(matches("(.iso$)|(.xts$)|(.lbl$)|(hour)|(minute)|(second)|(am.pm)|(date_year$)|(date_day$)|(id$)|(date_month$)|(date_wday$)")) %>%
  step_normalize (date_index.num, date_mday7, date_week4, date_week3, date_week2, date_week,date_mweek, date_yday, date_qday, date_mday,date_quarter,date_half, tempe_verage,tempemin,tempemax, -all_outcomes())%>%
  step_select_forests(all_predictors(), scores = TRUE, threshold = 0.70, outcome = "value")

recipe_spec %>% prep() %>% juice() %>% glimpse()
summary(recipe_spec)

### step 3
### Training the six ML algorithms with grid-search
# Model 1: grid-search Xgboost ----
wflw_fit_xgboost <- workflow() %>%
  add_model(
    boost_tree("regression", min_n = tune(),
               trees = tune(),
               tree_depth = tune(),
               learn_rate = tune(),
               loss_reduction = tune(),
               sample_size = tune()) %>% set_engine("xgboost", num.threads = 20))%>%
  add_recipe(recipe_spec %>% step_rm(date)) %>%
  tune_grid(grid = 25, recipe_spec, resamples = emergency_tscv, control = control_grid(verbose = TRUE, parallel_over = "resamples", allow_par = TRUE),
            metrics = metric_set(rmse))

xgboost_tune_RF_7 <- wflw_fit_xgboost
show_best(xgboost_tune_RF_7, metric = "rmse", n=5)

xgboost_tune_RF_45 <- wflw_fit_xgboost
show_best(xgboost_tune_RF_45, metric = "rmse", n=5)
xgboost_best <- 
  wflw_fit_xgboost %>% 
  select_best(metric = "rmse")
xgboost_best 


metrics <- show_best(xgboost_tune_RF_7, metric = "rmse", n=5)
export(metrics, "xgboost_tune_RF_7_ARMA.xlsx") 


metrics <- show_best(xgboost_tune_RF_45, metric = "rmse", n=5)
export(metrics, "xgboost_tune_RF_45_ARMA.xlsx") 


# Model 1: Xgboost ----
wflw_fit_xgboost_best_RF_7 <- workflow() %>%
  add_model(
    boost_tree("regression", 
               trees = 748,
               min_n = 30,
               tree_depth = 1,
               learn_rate = 0.0455,
               loss_reduction = 3.10e-10,
               sample_size = 0.332) %>% set_engine("xgboost") 
  ) %>%
  add_recipe(recipe_spec %>% step_rm(date)) %>%
  fit(training(emergency_tscv$splits[[1]]))



# Model 1: Xgboost ----
wflw_fit_xgboost_best_RF_45 <- workflow() %>%
  add_model(
    boost_tree("regression",
               trees = 1073,
               min_n = 34,
               tree_depth = 3,
               learn_rate = 0.00921,
               loss_reduction = 7.06,
               sample_size = 0.101) %>% set_engine("xgboost") 
  ) %>%
  add_recipe(recipe_spec %>% step_rm(date)) %>%
  fit(training(emergency_tscv$splits[[1]]))


wflw_fit_xgboost_best_RF_45 %>% 
  extract_recipe() %>%
  tidy(number = 4, type = "scores")


SCORES <- wflw_fit_xgboost_best_RF_45 %>% 
  extract_recipe() %>% 
  tidy(number = 4, type = "scores")
view(SCORES)

SCORES %>%
  mutate(variable = fct_reorder(variable, score)) %>%
  ggplot( aes(x=variable, y=score)) +
  geom_bar(stat="identity", fill="black", alpha=.6, width=.4) +
  coord_flip() +
  xlab("") +
  theme_bw() + labs(x = "Final predictor variables", y = "Importance scores (%)", title = "ARMA - XGBoost")

# Model 2: grid-search LightGBM ----
wflw_fit_lightgbm <- workflow() %>%
  add_model(
    boost_tree("regression", min_n = tune(),
               trees = tune(),
               tree_depth = tune(),
               learn_rate = tune(),
               loss_reduction = tune(),
               sample_size = tune()) %>% set_engine("lightgbm", num.threads = 20) 
  )  %>%
  add_recipe(recipe_spec %>% step_rm(date)) %>%
  tune_grid(grid = 25, recipe_spec, resamples = emergency_tscv, control = control_grid(verbose = TRUE, parallel_over = "resamples", allow_par = TRUE),
            metrics = metric_set(rmse))  # parallel_over = "everything"


lightgbm_tune_RF_7 <- wflw_fit_lightgbm
show_best(lightgbm_tune_RF_7, metric = "rmse", n=5)


lightgbm_tune_RF_45 <- wflw_fit_lightgbm
show_best(lightgbm_tune_RF_45, metric = "rmse", n=5)


metrics <- show_best(lightgbm_tune_RF_7, metric = "rmse", n=5)
export(metrics, "lightgbm_tune_7_ARMA.xlsx") 


metrics <- show_best(lightgbm_tune_RF_45, metric = "rmse", n=5)
export(metrics, "lightgbm_tune_45_ARMA.xlsx") 

lightgbm_best <- 
  wflw_fit_lightgbm %>% 
  select_best(metric = "rmse")
lightgbm_best 

# # Model 2: LightGBM ----
wflw_fit_lightgbm_best_RF_7 <- workflow() %>%
  add_model(
    boost_tree("regression",
               trees = 1706,
               min_n = 33,
               tree_depth =  13,
               learn_rate = 0.000982,
               loss_reduction = 0.0000170,
               sample_size = 0.193) %>% set_engine("lightgbm")
  ) %>%
  add_recipe(recipe_spec %>% step_rm(date)) %>%
  fit(training(emergency_tscv$splits[[1]]))

# # Model 2: LightGBM ----
wflw_fit_lightgbm_best_RF_45 <- workflow() %>%
  add_model(
    boost_tree("regression", 
               trees = 263,
               min_n = 37,
               tree_depth = 10,
               learn_rate = 0.00813,
               loss_reduction = 0.00000000415,
               sample_size = 0.348) %>% set_engine("lightgbm")
  ) %>%
  add_recipe(recipe_spec %>% step_rm(date)) %>%
  fit(training(emergency_tscv$splits[[1]]))


wflw_fit_lightgbm_best_RF_45 %>% 
  extract_recipe() %>%
  tidy(number = 4, type = "scores")


SCORES <- wflw_fit_lightgbm_best_RF_45 %>% 
  extract_recipe() %>% 
  tidy(number = 4, type = "scores")
view(SCORES)

SCORES %>%
  mutate(variable = fct_reorder(variable, score)) %>%
  ggplot( aes(x=variable, y=score)) +
  geom_bar(stat="identity", fill="black", alpha=.6, width=.4) +
  coord_flip() +
  xlab("") +
  theme_bw() + labs(x = "Final predictor variables", y = "Importance scores (%)", title = "ARMA - LightGBM")


# Model 3: grid-search Random Forest by ranger ---- 
wflw_fit_rf <- workflow() %>%
  add_model(
    rand_forest("regression", mtry = tune(),  trees = tune(), min_n = tune()) %>% set_engine("ranger", num.threads = 20)
  ) %>%
  add_recipe(recipe_spec %>% step_rm(date)) %>% # Add preprocessing steps (Note that "date" column is removed since Machine Learning algorithms don't typically know how to deal with date or date-time features)
  tune_grid(grid = 25, recipe_spec, resamples = emergency_tscv, control = control_grid(verbose = TRUE, parallel_over = "resamples", allow_par = TRUE),
            metrics = metric_set(rmse))


RF_tune_RF_7 <- wflw_fit_rf
show_best(RF_tune_RF_7, metric = "rmse", n =5)

RF_tune_RF_45 <- wflw_fit_rf
show_best(RF_tune_RF_45, metric = "rmse", n =5)


RF_best <- 
  wflw_fit_rf %>% 
  select_best(metric = "rmse")
RF_best



metrics <- show_best(RF_tune_RF_7, metric = "rmse", n=5)
export(metrics, "RF_tune_7_ARMA.xlsx") 


metrics <- show_best(RF_tune_RF_45, metric = "rmse", n=5)
export(metrics, "RF_tune_RF_45_ARMA.xlsx") 



# Model 3: Random Forest ----
wflw_fit_rf_best_RF_7 <- workflow() %>%
  add_model(
    rand_forest("regression", mtry = 1,
                trees = 1304,
                min_n = 15) %>% set_engine("ranger")
  ) %>%
  add_recipe(recipe_spec %>% step_rm(date)) %>% # Add preprocessing steps (Note that "date" column is removed since Machine Learning algorithms don't typically know how to deal with date or date-time features)
  fit(training(emergency_tscv$splits[[1]]))


# Model 3: Random Forest ----
wflw_fit_rf_best_RF_45 <- workflow() %>%
  add_model(
    rand_forest("regression",mtry = 2,
                trees = 146, min_n = 4) %>% set_engine("ranger")
  ) %>%
  add_recipe(recipe_spec %>% step_rm(date)) %>% # Add preprocessing steps (Note that "date" column is removed since Machine Learning algorithms don't typically know how to deal with date or date-time features)
  fit(training(emergency_tscv$splits[[1]]))


wflw_fit_rf_best_RF_45 %>% 
  extract_recipe() %>%
  tidy(number = 4, type = "scores")


SCORES <- wflw_fit_rf_best_RF_45 %>% 
  extract_recipe() %>% 
  tidy(number = 4, type = "scores")
view(SCORES)

SCORES %>%
  mutate(variable = fct_reorder(variable, score)) %>%
  ggplot( aes(x=variable, y=score)) +
  geom_bar(stat="identity", fill="black", alpha=.6, width=.4) +
  coord_flip() +
  xlab("") +
  theme_bw() + labs(x = "Final predictor variables", y = "Importance scores (%)", title = "ARMA - RF")


# Radial Basis Function Support Vector Machine
# Model 4: SVM-RBF grid-search  -----
wflw_svm_rbf <- workflow() %>%
  add_model(
    svm_rbf ("regression", cost = tune(), rbf_sigma = tune(), margin =tune()) %>% set_engine("kernlab", num.threads = 20))%>%
  add_recipe(recipe_spec %>% step_rm(date)) %>%
  tune_grid(grid = 25, recipe_spec, resamples = emergency_tscv, control = control_grid(verbose = TRUE, parallel_over = "resamples", allow_par = TRUE),
            metrics = metric_set(rmse))


svm_rbf_tune_RF_7 <- wflw_svm_rbf
show_best(svm_rbf_tune_RF_7, metric = "rmse", n =5)


svm_rbf_tune_RF_45 <- wflw_svm_rbf
show_best(svm_rbf_tune_RF_45, metric = "rmse", n =5)
svm_rbf_best <- 
  wflw_svm_rbf %>% 
  select_best(metric = "rmse")
svm_rbf_best


metrics <- show_best(svm_rbf_tune_RF_7, metric = "rmse", n=5)
export(metrics, "svm_rbf_tune_RF_7_ARMA.xlsx") 


metrics <- show_best(svm_rbf_tune_RF_45, metric = "rmse", n=5)
export(metrics, "svm_rbf_tune_RF_45_ARMA.xlsx") 


# Model 4: SVM-RBF -----
wflw_fit_svm_rbf_RF_7 <- workflow() %>%
  add_model(
    svm_rbf("regression",
            cost = 7.77,
            rbf_sigma = 0.00577,
            margin = 0.198) %>% set_engine("kernlab") 
  ) %>%
  add_recipe(recipe_spec %>% step_rm(date)) %>%
  fit(training(emergency_tscv$splits[[1]]))


# Model 4: SVM-RBF -----
wflw_fit_svm_rbf_RF_45 <- workflow() %>%
  add_model(
    svm_rbf("regression",
            cost = 3.20,
            rbf_sigma = 0.000571,
            margin = 0.100) %>% set_engine("kernlab") 
  ) %>%
  add_recipe(recipe_spec %>% step_rm(date)) %>%
  fit(training(emergency_tscv$splits[[1]]))

wflw_fit_svm_rbf_RF_45 %>% 
  extract_recipe() %>%
  tidy(number = 4, type = "scores")


SCORES <- wflw_fit_svm_rbf_RF_45 %>% 
  extract_recipe() %>% 
  tidy(number = 4, type = "scores")
view(SCORES)


SCORES %>%
  mutate(variable = fct_reorder(variable, score)) %>%
  ggplot( aes(x=variable, y=score)) +
  geom_bar(stat="identity", fill="black", alpha=.6, width=.4) +
  coord_flip() +
  xlab("") +
  theme_bw() + labs(x = "Final predictor variables", y = "Importance scores (%)", title = "ARMA - SVM-RBF")

# Model 5: NNAR grid-search -----
# Model 5: grid-search ----
wflw_fit_NNAR <- workflow() %>% 
  add_model(
    nnetar_reg(penalty = tune(), hidden_units =tune(), num_networks=tune()) %>%
      set_engine("nnetar",num.threads = 20)) %>%
  add_recipe(recipe_spec) %>%
  tune_grid(grid = 25, recipe_spec, resamples = emergency_tscv, control = control_grid(verbose = TRUE, parallel_over = "resamples", allow_par = TRUE),
            metrics = metric_set(rmse))



NNAR_tune_RF_7 <- wflw_fit_NNAR
show_best(NNAR_tune_RF_7, metric = "rmse", n =5)

NNAR_tune_RF_45 <- wflw_fit_NNAR
show_best(NNAR_tune_RF_45, metric = "rmse", n=5)

NNAR_best <- 
  wflw_fit_NNAR %>% 
  select_best(metric = "rmse")
NNAR_best 


metrics <- show_best(NNAR_tune_RF_7, metric = "rmse", n=5)
export(metrics, "NNAR_tune_RF_7_ARMA.xlsx") 

metrics <- show_best(NNAR_tune_RF_45, metric = "rmse", n=5)
export(metrics, "NNAR_tune_RF_45_ARMA.xlsx") 


# Model 5: NNAR -----
wflw_fit_ANN_best_RF_7 <- workflow() %>% 
  add_model(
    nnetar_reg(hidden_units = 1, num_networks =10, penalty = 0.928) %>%
      set_engine("nnetar")) %>%
  add_recipe(recipe_spec) %>%
  fit(training(emergency_tscv$splits[[1]]))

# Model 5: NNAR -----
wflw_fit_ANN_best_RF_45 <- workflow() %>% 
  add_model(
    nnetar_reg(hidden_units = 3, num_networks = 8, penalty = 0.000000870) %>%
      set_engine("nnetar")) %>%
  add_recipe(recipe_spec) %>%
  fit(training(emergency_tscv$splits[[1]]))

wflw_fit_ANN_best_RF_45 %>% 
  extract_recipe() %>%
  tidy(number = 4, type = "scores")


SCORES <- wflw_fit_ANN_best_RF_45 %>% 
  extract_recipe() %>% 
  tidy(number = 4, type = "scores")
view(SCORES)


SCORES %>%
  mutate(variable = fct_reorder(variable, score)) %>%
  ggplot( aes(x=variable, y=score)) +
  geom_bar(stat="identity", fill="black", alpha=.6, width=.4) +
  coord_flip() +
  xlab("") +
  theme_bw() + labs(x = "Final predictor variables", y = "Importance scores (%)", title = "ARMA - NNAR")


# Model 6: GLMNET grid-search  ----
wflw_fit_glmnet <- workflow() %>%
  add_model(
    linear_reg(penalty = tune(),mixture = tune()
    ) %>% set_engine("glmnet", num.threads = 20))%>%
  add_recipe(recipe_spec %>% step_rm(date)) %>%
  tune_grid(grid = 25, recipe_spec, resamples = emergency_tscv, control = control_grid(verbose = TRUE, parallel_over = "resamples", allow_par = TRUE),
            metrics = metric_set(rmse))


glmnet_tune_RF_7 <- wflw_fit_glmnet
show_best(glmnet_tune_RF_7, metric = "rmse", n=5)


glmnet_tune_RF_45 <- wflw_fit_glmnet
show_best(glmnet_tune_RF_45, metric = "rmse", n=5)


glmnet_best <- 
  wflw_fit_glmnet %>% 
  select_best(metric = "rmse")
glmnet_best 


metrics <- show_best(glmnet_tune_RF_7 , metric = "rmse", n=5)
export(metrics, "glmnet_tune_RF_7_ARMA.xlsx") 


metrics <- show_best(glmnet_tune_RF_45, metric = "rmse", n=5)
export(metrics, "glmnet_tune_RF_45_ARMA.xlsx") 


# Model 6: GLMNET  ----
wflw_fit_glmnet_best_RF_7 <- workflow() %>%
  add_model(
    linear_reg(penalty = 0.0000000146, 
               mixture = 0.980) %>% set_engine("glmnet"))%>%
  add_recipe(recipe_spec %>% step_rm(date)) %>%
  fit(training(emergency_tscv$splits[[1]]))


# Model 6: GLMNET  ----
wflw_fit_glmnet_best_RF_45 <- workflow() %>%
  add_model(
    linear_reg(penalty = 0.773,
               mixture = 0.211) %>% set_engine("glmnet"))%>%
  add_recipe(recipe_spec %>% step_rm(date)) %>%
  fit(training(emergency_tscv$splits[[1]]))


wflw_fit_glmnet_best_RF_45 %>% 
  extract_recipe() %>%
  tidy(number = 4, type = "scores")


SCORES <- wflw_fit_glmnet_best_RF_45 %>% 
  extract_recipe() %>% 
  tidy(number = 4, type = "scores")
view(SCORES)

SCORES %>%
  mutate(variable = fct_reorder(variable, score)) %>%
  ggplot( aes(x=variable, y=score)) +
  geom_bar(stat="identity", fill="black", alpha=.6, width=.4) +
  coord_flip() +
  xlab("") +
  theme_bw() + labs(x = "Final predictor variables", y = "Importance scores (%)", title = "ARMA - GLMNET")




### step 4
### Create Modeltime Table ----
### The Modeltime Table organizes the models with IDs and creates generic descriptions to help us keep track of our models. 
### Let's add the models to a modeltime_table()
### table modeltime ----

model_tbl <- modeltime_table(wflw_fit_xgboost_best_RF_7,
                             wflw_fit_lightgbm_best_RF_7,
                             wflw_fit_rf_best_RF_7,
                             wflw_fit_svm_rbf_RF_7,
                             wflw_fit_ANN_best_RF_7,
                             wflw_fit_glmnet_best_RF_7
)

model_tbl

model_tbl <- modeltime_table(wflw_fit_xgboost_best_RF_45,
                             wflw_fit_lightgbm_best_RF_45,
                             wflw_fit_rf_best_RF_45,
                             wflw_fit_svm_rbf_RF_45,
                             wflw_fit_ANN_best_RF_45,
                             wflw_fit_glmnet_best_RF_45
)

model_tbl



# 5-fold cross-validation prediction of time series ----
resample_results <- model_tbl %>%
  modeltime_fit_resamples(
    resamples = emergency_tscv,
    control   = control_resamples(allow_par = TRUE, verbose = TRUE)
  )


resample_results

##---## Results by 5-fold resampling -----
resample_results %>%
  modeltime_resample_accuracy(summary_fns = NULL, yardstick::metric_set(mape, smape, mase, rmse)) %>%
  table_modeltime_accuracy(.interactive = FALSE)

resample_results_fitted 

##---## Average results of five folds of resampling -----
resample_results %>%
  modeltime_resample_accuracy(summary_fns = mean, yardstick::metric_set(mape, smape, mase, rmse)) %>%
  table_modeltime_accuracy(.interactive = FALSE)


## Charts to compare the performance of the models across the four evaluation metrics (MAPE, sMAPE, MASE, and RMSE)----
resample_results %>%
  plot_modeltime_resamples(yardstick::metric_set(mape, smape, mase, rmse),
                           .point_size  = 4,
                           .summary_line_size  =  1,
                           .point_alpha = 0.8,
                           .interactive = FALSE,
                           .title =  "ARMA - 7-day test set",
                           .color_lab =  "Algorithms"
  )

resample_results %>%
  plot_modeltime_resamples(yardstick::metric_set(mape, smape, mase, rmse),
                           .point_size  = 4,
                           .summary_line_size  =  1,
                           .point_alpha = 0.8,
                           .interactive = FALSE,
                           .title =  "ARMA - 45-day test set",
                           .color_lab =  "Algorithms"
  )
