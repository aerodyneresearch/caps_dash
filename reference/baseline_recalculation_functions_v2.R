
require(tidyverse)
require(zoo)

baseline_recalc = function(df_caps_data,
                           status_col,
                           loss_col,
                           temperature_col,
                           pressure_col,
                           span_col,
                           species,
                           LED_time = 3) {
  # Function: Main baseline recalculation function
  # Input:
  #   df_caps_data - dataframe - standard caps dataframe
  #   LED_time - int - number of seconds to remove from the beginning of the baseline
  # Output:
  #   dataframe - CAPS dataframe contain recalculated and defined baselines
  
  rayleigh_col = paste("rayleigh", find_LED_color(df_caps_data, status_col), sep = "_")
  last_baseline_col_name = paste("LastBaseline",loss_col,"recalc", sep="_")
  print(rayleigh_col)
  
  df_baseline_recalc = assign_baseline_period(df_caps_data) %>%
                       assign_baseline_number() %>%
                       filter_bad_baselines(loss_col) %>% 
                       assign_baseline_number() %>% #This is needed to reassign baseline numbers after filtering bad baselines
                       assign_rayleigh(status_col) %>%
                       recalculate_baseline_loss(loss_col , rayleigh_col) %>%
                       recalculate_concentration(loss_col, span_col, rayleigh_col, temperature_col, pressure_col, species = species) %>%
                       baseline_interpolation(last_baseline_col_name) %>%
                       recalculate_concentration(loss_col, span_col, rayleigh_col, temperature_col, pressure_col, species = species, interp = TRUE)
  
  return(df_baseline_recalc)
}

#### Rayleigh Calculation ####

find_LED_color = function(df_caps_data, status_col) {
  status = df_caps_data[status_col][1,1]
  LED_color = switch(substring(status, nchar(status), nchar(status)),
                     "2" = 365,
                     "3" = 405,
                     "4" = 450,
                     "5" = 530,
                     "6" = 630,
                     "7" = 660,
                     "8" = 780)
  return(LED_color)
}

find_rayleigh_constant = function(LED_color) {
  switch( #This should be moved to its own area
    as.character(LED_color),
    "365" = 64.3,
    "405" = 42.4,
    "450" = 27.6,
    "530" = 14.1,
    "630" = 6.96,
    "660" = 5.98,
    "780" = 3.07
  )
}

assign_rayleigh = function(df_caps_data, status_col) {
  STP_TEMP = 273.15
  STP_PRES = 760
  LED_color = find_LED_color(df_caps_data, status_col)
  rayleigh_constant = find_rayleigh_constant(LED_color)
  name = paste("rayleigh", as.character(LED_color), sep = "_")
  print(paste("Assigning rayleigh to", name))
  output = df_caps_data %>% 
            mutate(!!name := rayleigh_constant * (Pressure/STP_PRES) * ((STP_TEMP)/Temperature))
  return(output)
} 

#### Baseline Functions ####

assign_baseline_period = function(df_caps_data) {
  # Function: Add the a column for baseline period to a CAPS dataframe
  # Input:
  #   df_caps_data - dataframe - standard caps dataframe
  # Output:
  #   dataframe - CAPS dataframe that contains a baseline period. 1 is a baseline
  #               and 0 is a measurement
  
  caps_output <- df_caps_data %>%
    rowwise() %>%
    mutate(
      baseline_period = ifelse(
        any(
        startsWith(as.character(Status), c("32"))), 1, 0
      )
    )
  
  if(sum(is.na(caps_output$baseline_period > 0))) {print("Baseline Period has NA's")}
  
  print(paste("Finished assigning baseline period, unique values are:", paste(unique(caps_output$baseline_period), collapse = ",")))
  return(caps_output)
  
}

assign_baseline_number = function(df_caps_data) {

  baseline_number = c(rep(1,nrow(df_caps_data)))
  j = 1
  
  if(sum(is.na(df_caps_data$baseline_period > 0))) {print("Baseline Period has NA's")}
  
  if(sum(df_caps_data$baseline_period == 1, na.rm = TRUE) == 0) {
    print("No useable baselines")
    return(df_caps_data)
  }
  
  for(i in which(df_caps_data$baseline_period == 1)[1]:nrow(df_caps_data)) {
    baseline_number[i] = j
    if(i == nrow(df_caps_data)) {next}
    if(df_caps_data$baseline_period[i] < df_caps_data$baseline_period[i+1]) {
      j = j + 1
    }
  }
  
  print("Finished assigning baseline number")
  df_caps_data$baseline_number = baseline_number
  
  return(df_caps_data)
  
}

filter_bad_baselines <- function(df_caps_data, loss_col) {
  output_col_name = paste("sd_loss_baseline",loss_col, sep="_")
  
  standard_deviation_filter = 0.3 #Mm-1
  
  output = df_caps_data %>%
    group_by(baseline_number) %>%
    filter(baseline_period == 1) %>% 
    summarise(
      !!output_col_name := sd(.data[[loss_col]], na.rm = T)) %>% 
    right_join(df_caps_data, by = "baseline_number") %>%
    mutate(
      baseline_period = ifelse(
        startsWith(as.character(Status), c("32")) & !is.na(.data[[output_col_name]]) & .data[[output_col_name]] < standard_deviation_filter, 1, 0)) 
     
  print(paste("Finished filtering points:", sum(output$baseline_period == 1, na.rm = TRUE), "points remain"))
  
  
  return(output)      
}

recalculate_baseline_loss <- function(df_caps_data, loss_col, rayleigh_col) {
  
  
  output_col_name = paste("LastBaseline",loss_col,"recalc", sep="_")
  
  baseline_output = df_caps_data %>%
    filter(baseline_period == 1) %>%
    group_by(baseline_number) %>%
    summarise(
      ave_loss = mean(.data[[loss_col]]),
      IQR = IQR(.data[[loss_col]], na.rm = T),
      min_threshold = quantile(.data[[loss_col]], prob = 0.25, na.rm = T) - 1.5 * IQR,
      max_threshold = quantile(.data[[loss_col]], prob = 0.75, na.rm = T) + 1.5 * IQR
    ) %>% glimpse() %>%
    right_join(df_caps_data) %>% 
    filter(between(.data[[loss_col]], min_threshold, max_threshold)) %>%
    group_by(baseline_number) %>%
    summarise(
      !!output_col_name := mean(.data[[loss_col]]) - mean(.data[[rayleigh_col]])
    ) %>% right_join(df_caps_data) %>%
    mutate(baseline_period = replace_na(baseline_period, 0))

  print("Finished recalculating baseline")
  return(baseline_output)
}

recalculate_concentration <- function(df_caps_data, 
                                      loss_col,
                                      span_col, 
                                      rayleigh_col,
                                      temperature_col, 
                                      pressure_col,
                                      species,
                                      interp = FALSE) {
  
  STP_PRES = 760
  STP_TEMP = (273.15)
  
  if (interp == TRUE) {
    name_val = "interp"
    name_val2 = "recalc_interp"
  } else {
    name_val = "recalc"
    name_val2 = "recalc"
  }
  
  name = paste("concentration",species,name_val, sep = "_")
  baseline_recalc_col = paste("LastBaseline",loss_col,name_val2, sep="_")
  
  df_caps_data[,name] <- 
    ((df_caps_data[loss_col] - df_caps_data[rayleigh_col]) - df_caps_data[baseline_recalc_col])/df_caps_data[span_col]*(STP_PRES/df_caps_data[pressure_col]*df_caps_data[temperature_col]/STP_TEMP)
  
  print("Finished recalculating concentration")
  return(df_caps_data)
  
  }

#### Interpolation Functions

baseline_interpolation <- function(df_caps_data,
                                   baseline_col) {
  
  output_col_name = paste(baseline_col,"interp", sep="_")
  print(output_col_name)
  
  output = df_caps_data %>%
    mutate(
      !!output_col_name := na.approx(
        ifelse((baseline_period == 1 | baseline_number %in% c(1, max(baseline_number))), get(baseline_col), NA), na.rm = FALSE)
    )
  
}

