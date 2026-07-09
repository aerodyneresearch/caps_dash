
require(tidyverse)

get_parameters = function(file) {
  
  # Function: Get the instrument metadata from the CAPS header
  # Input: 
  #   file - CAPS file path
  # Return:
  #   Character Vector - Parameters of CAPS input
  
  lines = (readLines(file))
  
  metadata = lines[grepl("%", lines)]
  
  last_header_i = which(grepl("Exact Sample Time", metadata))
  last_header_e = which(grepl("Igor", lines, ignore.case = T)) - 1
          
  parameters = metadata[(last_header_i+1): last_header_e] %>%
  gsub(pattern="%\\s+", replacement="") %>%
  strsplit(split="\\s+") %>%
  unlist()
  
  return(parameters)
}

LED_color <- function(status) {
  # Function: Return the color of the LED in the CAPS instrument based on the status code
  # Input: 
  #   status - Integer containing the CAPS status
  # Return:
  #   Int - Wavelength of the LED
  return(switch(substring(as.character(status[1]), 5, 5),
         "2" = 365,
         "3" = 405,
         "4" = 450,
         "5" = 530,
         "6" = 630,
         "7" = 660,
         "8" = 780))
}

rayleigh_constant <- function(LED_color) {
  # Function: Return the Rayleigh correction constant based on LED color
  # Input:
  #   LED_color - Int - Contains LED color
  # Return:
  #   Int - Rayleigh correction value
  switch(as.character(LED_color),
  "365" = 64.3,
  "405" = 42.4,
  "450" = 27.6,
  "530" = 14.1,
  "630" = 6.96,
  "660" = 5.98,
  "780" = 3.07)
}
