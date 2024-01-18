# Package loading
library(ForecastTB)
library(ggplot2)
library(zoo)
library(forecast)
library(lmtest)
library(urca)
library(stats)
library(nnfor)
library(forecastHybrid)
library(pastecs)
library(forecastML)
library(Rcpp)
library(modeltime.ensemble)
library(tidymodels)
library(modeltime)
library(timetk)
library(lubridate)
library(tidyverse)
library(modeltime.h2o)
library(yardstick)
library(reshape)
library(scales)
library(timetk)
library(plotly)
library(RColorBrewer)
library(Polychrome)

# Database loading
attach(datasets)

# Viewing the cross-validation plan conducted in the study

data_tbl <- datasets %>%
  filter(id == "arma") %>%
  select(id, Date, attendences, average_temperature, min, max,  sunday, monday, tuesday, wednesday, thursday, friday, saturday, Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec) %>%
  set_names(c("id", "date", "value","tempe_verage", "tempemin", "tempemax", "sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"))

data_tbl

# Displaying the data
data_tbl

# Setting up the cross-validation plan
splits <- data_tbl %>%
  filter(id == "arma") %>%
  time_series_cv(
    assess = "45 days",
    skip = "45 days",
    cumulative = TRUE,
    slice_limit = 5
  )

# Displaying the cross-validation plan
splits %>%
  tk_time_series_cv_plan()

# Visualizing the cross-validation plan
splits %>%
  tk_time_series_cv_plan() %>%
  plot_time_series_cv_plan(.title = "Cross-validation ARMA", .interactive = FALSE, date, value)

# ED visits time series
data_tbl <- datasets %>%
  select(id, Date, attendences) %>%
  set_names(c("id", "date", "value"))

# Displaying the data
data_tbl

# Plotting time series
data_tbl %>%
  group_by(id) %>%
  plot_time_series(
    date, value, .interactive = FALSE, .facet_ncol = 5, .title = ""
  )

# Another way of visualization to plot all series
# Plotting time series
ggplot(data = data_tbl, aes(x = Date, y = attendences, fill=id)) +
  geom_area() + labs(x = "Daily data", y = "Arrivals by hospitals")

ggplot(data = data_tbl, aes(x = Date, y = attendences, color=id)) +
  geom_area() + labs(x = "Daily data", y = "Arrivals by hospitals")

# Boxplot time series
ggplot(data = data_tbl, aes(x = Date, y = attendences, fill=id)) +
  geom_area() + labs(x = "Daily data", y = "Arrivals by hospitals")

p <- ggplot(data = data_tbl, aes(x = Date, y = attendences, color=id)) +
  geom_line() + scale_x_datetime(date_breaks = "1 year", date_labels = "%b %Y") + theme_minimal() + labs(color="EDs", x = "Daily data", y = "Arrivals by hospitals")
p + scale_fill_brewer(palette = "Set1")

p <- ggplot(data = data_tbl, aes(x = Date, y = attendences, color=id)) +
  geom_point() + labs(x = "Daily data", y = "Arrivals by hospitals")
p + scale_fill_brewer(palette ="Set1")

#top of form
p <- ggplot(data = data_tbl, aes(x = Date, y = attendences, color=id)) +
  geom_line() + scale_x_datetime(date_breaks = "1 year", date_labels = "%b %Y") + labs(color="EDs", x = "Daily data", y = "Arrivals by hospitals")
p + scale_colour_manual(
  values = c("aku" = "black", "amc" = "gold2", "antoniushove" = "gray45", "arma" ="dimgray", "bronovo"= "red", "davis" ="red1", "fre"="gray100", "fs"="red3", "joon"="red4", "kem" ="blue", "marina" ="green1", "pm"="blue2", "rg" ="blue3", "rph" ="blue4", "scg"="tan", "sjgm" ="tan1", "swan" ="tan2", "westeinde" ="tan3", "hcpa" ="tan4", "Iowa"="pink"))


# Inserting series names on X-axis
ggplot(data = data_tbl, aes(x = id, y = attendences, fill=id)) +
  geom_boxplot() + labs(fill="EDs", x = "Daily data", y = "Arrivals by hospitals")

## plot time series best form##

data_tbl %>%
  plot_time_series(Date, attendences, .color_var = id, .smooth = F, .y_lab ="Arrivals by hospitals", .x_lab = "Daily data", .title = "")

data_tbl %>%
  plot_time_series(Date, attendences, .color_var = id, .interactive = F, .smooth = F, .y_lab ="Arrivals by hospitals",  .x_lab = "Daily data", .title = "") 


data_tbl %>%
  plot_time_series(Date, attendences, .color_var = id, .interactive = F, .smooth = F, .y_lab ="Arrivals by hospitals",  .x_lab = "Daily data", .title = "", .line_size = 0.8) 

## This worked for me with a windows 10 laptop in German, where I wanted i.e. lubridate to return dates in English:
Sys.setlocale("LC_TIME", "English")

# Visualize seasonality box plot test in id time series
data_tbl %>% filter(id == "arma") %>%
  plot_seasonal_diagnostics(
    date, value, 
    .feature_set = c("wday.lbl", "month.lbl"),
    .interactive = FALSE, .title = "ARMA"
  )

data_tbl %>%
  plot_seasonal_diagnostics(
    date, value,
    .feature_set = c("week", "quarter"),
    .interactive = FALSE
  )


# Summary Diagnostics. Check for summary of timeseries
data_tbl <- datasets %>%
  select(id, Date, attendences, average_temperature) %>%
  set_names(c("id", "date", "value","temperature"))

# Displaying the data
data_tbl

data_tbl %>%
  group_by(id) %>%
  tk_summary_diagnostics()
