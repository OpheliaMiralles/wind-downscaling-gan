# Title     : DOWNLOAD INLA
# Objective : Download r-inla
# Created by: opheliamiralles
# Created on: 14.06.21

setwd("/Users/opheliamiralles/Desktop/PhD/EPFL/WindDownscaling_EPFL_UNIBE/INLA")
local({ r <- getOption("repos")
  r["CRAN"] <- "http://cran.r-project.org"
  options(repos = r)
})
install.packages("INLA", repos = c(getOption("repos"), INLA = "https://inla.r-inla-download.org/R/testing"), dep = TRUE)