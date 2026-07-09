caps_read <- function(file) {
  output = data.table::fread(file) %>%
    as.data.frame() %>%
    mutate(Timestamp = as.POSIXct(Timestamp, format = "%Y/%m/%d %H:%M:%S"))
  return(output)
}

#' Calculate Rayleigh Scattering
#'
#' @param rayleigh_constant 
#' @param temperature 
#' @param pressure 
#'
#' @return A numeric vector giving Rayleigh scattering at a given temperature 
#' and pressure for a given scattering constant (which is based on wavelength)
#' @export
#'
#' @examples
calculate_rayleigh <- function(temperature, pressure, rayleigh_constant) {
  
  rayleigh_scattering <- rayleigh_constant * (pressure / PRESSURE_STP * TEMPERATURE_STP / temperature)
  return(rayleigh_scattering)

}

