
library(tidyverse)
source("./scripts/baseline_recalculation_functions_v2.R")

#### Inputs ####

ALL_DATA = TRUE

#### Determine Files ####

files <- fs::dir_ls("./data/Raw Data/") #faster list.files() function

# Determine what files need to be run #

if(ALL_DATA == F) {
  print("Processing only new data")
  corr_files <- fs::dir_ls("./Data/corrected_data/")
  corr_file_dates = sapply(corr_files, function(x) strsplit(x, split = "[/_.]")[[1]][6])
  
  files <- files[!grepl(paste(corr_file_dates, collapse = "|"), files)]
} else {
  print("Processing all data")
}


for (file in seq_along(files)) {
  date <- str_split(files[file], pattern = "\\.")[[1]][4]
  print(paste("Running", date))
  data <- data.table::fread(files[file], skip = "Igor") %>%
    rowwise() %>%
    mutate(datetime = as.POSIXct(Timestamp, format = "%Y/%m/%d %H:%M:%S", tz = "UTC"))
  
  ####Skip files without baseline periods ####
  if(!("32164" %in% unique(data$Status))) {
    print(paste("Skipping", date, "due to incomplete baselines"))
    next
  }
  
  #### Correct Data ####
  
  data_corr = baseline_recalc(data, "Status", "Loss_NO2", "Temperature", "Pressure", "Cal_NO2", "NO2")
  data_corr = baseline_recalc(data_corr, "Status2", "Loss_NOx", "Temperature", "Pressure", "Cal_NOx", "NOx")
  
  #### Plots ####
  
  #Concentration Plot
  
  # ggplot_uncorrected_data <- ggplot(data_corr %>% filter(baseline_period == 0), aes(x = datetime, y = Concentration_NOx, color = "NOx")) + 
  #   geom_line() +
  #   geom_line(aes(x = datetime, y = Concentration_NO2, color = "NO2")) + 
  #   geom_line(aes(x = datetime, y = Concentration_NO, color = "NO")) +
  #   labs(
  #     title = paste(date, "Raw Concentration"),
  #     x = "Time (UTC)",
  #     y = "Mixing Ratio (ppb)") +
  #   scale_color_manual(
  #     name = "Species",
  #     values = c("NOx" = "black",
  #                "NO2" = "red",
  #                "NO" = "blue")
  #   ) + 
  # theme_bw()
  # ggsave(filename = paste0("./data/concentrations/raw/",date,"_","raw_data.png"), plot = ggplot_uncorrected_data)
  # 
  #Baseline Plot
  
  # ggplot_uncorrected_baselines <- ggplot(data_corr %>% filter(startsWith(as.character(Status), c("32"))), aes(x = datetime, y = Loss_NOx, color = "NOx Loss")) + 
  #   geom_line() +
  #   geom_line(aes(x = datetime, y = Loss_NO2, color = "NO2 Loss")) +    
  #   labs(
  #     title = paste(date, "Baseline Loss"),
  #     x = "Time (UTC)",
  #     y = "Loss (Mm-1)") +
  #       scale_color_manual(
  #         name = "Species",
  #         values = c("NOx Loss" = "black",
  #                    "NO2 Loss" = "red"
  #       )) + theme_bw()
  # ggsave(filename = paste0("./data/Baseline Correction Plots/raw baselines/",date,"_","raw_data.png"), plot = ggplot_uncorrected_baselines)
  # 

  #Corrected Data Baseline Plot
#   ggplot_NO2_Baseline = ggplot(data_corr,
#                       aes(x = datetime)
#       ) +
#     geom_line(aes(x = datetime, y = Baseline_NO2 - rayleigh_450, color = "Baseline NO2")) +
#     geom_line(aes(x = datetime, y = LastBaseline_Loss_NO2_recalc, color = "Baseline Recalc")) +
#     geom_line(aes(x = datetime, y = LastBaseline_Loss_NO2_recalc_interp, color = "Baseline Interp")) +
#     labs(
#       title = paste(date, "NO2 Baseline Recalculation"),
#       x = "Time (UTC)",
#       y = "Loss (Mm-1)") +
#     scale_color_manual(
#       name = NULL,
#       values = c(
#         "Baseline NO2" = "black",
#         "Baseline Recalc" = "red",
#         "Baseline Interp" = "blue"
#       )) +
#     theme_bw()
#   ggsave(filename = paste0("./data/Baseline Correction Plots/NO2/",date,"_","NO2_Baseline_Correction.png"), plot = ggplot_NO2_Baseline)
#   
#   ggplot_NOx_Baseline = ggplot(data_corr,
#                                aes(x = datetime)
#   ) +
#     geom_line(aes(x = datetime, y = Baseline_NOx - rayleigh_405, color = "Baseline NOx")) +
#     geom_line(aes(x = datetime, y = LastBaseline_Loss_NOx_recalc, color = "Baseline Recalc")) +
#     geom_line(aes(x = datetime, y = LastBaseline_Loss_NOx_recalc_interp, color = "Baseline Interp")) +
#     labs(
#       title = paste(date, "NOx Baseline Recalculation"),
#       x = "Time (UTC)",
#       y = "Loss (Mm-1)") +
#     scale_color_manual(
#       name = NULL,
#       values = c(
#         "Baseline NOx" = "black",
#         "Baseline Recalc" = "red",
#         "Baseline Interp" = "blue"
#       )) +
#     theme_bw()
# ggsave(filename = paste0("./data/Baseline Correction Plots/NOx/",date,"_","NOx_Baseline_Correction.png"), plot = ggplot_NOx_Baseline)
#   

#### Raw vs Recalculated Concentration Comparison #####

names(data_corr)

data_corr = data_corr %>%
  mutate(concentration_NO_interp = concentration_NOx_interp - concentration_NO2_interp)

ggplot_interp_conc = ggplot(data_corr %>%
         filter(startsWith(as.character(Status), "10"))) + 
         geom_line(aes(x = datetime, y = concentration_NO2_interp, color = "NO2")) +
         geom_line(aes(x = datetime, y = concentration_NOx_interp, color = "NOx")) +
         geom_line(aes(x = datetime, y = concentration_NO_interp, color = "NO")) +
         labs(
           title = paste(date, "Baseline Interpolated Concentrations"),
           x = "Time (UTC)",
           y = "Mixing Ratio (ppb)"
         ) + 
          scale_color_manual(
            name = "Species",
            values = c(
              "NO2" = "black",
              "NOx" = "blue",
              "NO" = "red"
            )
          ) + theme_bw()
ggsave(filename = paste0("./data/concentrations/interpolated baselines/",date,"_","interpolated_concentrations.png"), ggplot_interp_conc)


  #Corrected Data output
  
  corrected_output_filename = paste0("./data/corrected_data/",date,"_corrected_data.csv")
  
  data.table::fwrite(data_corr, file = corrected_output_filename)
  
  #### Add processed file to file list ####
  
}
