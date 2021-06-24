# Title     :  Multivariate regression model with additive / multiplicative bias depending on covariates
# Objective : Downscaling wind speed to specific points
# Created by: opheliamiralles
# Created on: 24.06.21

library(INLA)
library(splines)
inla.pardiso.check()

data.path <- "/Users/opheliamiralles/Desktop/PhD/EPFL/WindDownscaling_EPFL_UNIBE/data/point_prediction_files/dataframe_6months.csv"
out.data <- read.csv(data.path)

# Outputs
n <- nrow(out.data)
Y <- matrix(NA, n * 2, 2)
Y[1:n, 1] <- out.data$u10_hr
Y[(1 + n):(2 * n), 1] <- NA # faked observations
Y[(1 + n):(2 * n), 2] <- out.data$v10_hr   # actual observations
Y[1:n, 2] <- NA # faked observations
Y <- data.frame(Y)

# Inputs
x1 <- data.frame(a1 = 1,  u10 = out.data$u10, v10 = out.data$v10,
                fsr = out.data$fsr,
                z = out.data$z,
                blh = out.data$blh,
                hour = out.data$hour,
                month = out.data$month,
                station_id = out.data$station_id,
                tpi_500 = out.data$tpi_500,
                aspect = out.data$aspect,
                sp = out.data$sp,
                lon = out.data$lon,
                lat = out.data$lat)
x2 <- data.frame(a2 = 1,  u10 = out.data$u10, v10 = out.data$v10,
                fsr = out.data$fsr,
                z = out.data$z,
                blh = out.data$blh,
                hour = out.data$hour,
                month = out.data$month,
                station_id = out.data$station_id,
                tpi_500 = out.data$tpi_500,
                aspect = out.data$aspect,
                sp = out.data$sp,
                lon = out.data$lon,
                lat = out.data$lat)

# Creation of mesh, same locations for u10 and v10
loc1 <- cbind(x$lon, x$lat)
loc2 <- cbind(x$lon, x$lat)
bnd1 <- inla.nonconvex.hull(loc1, convex = 0.05, resolution = c(92, 43))
bnd2 <- inla.nonconvex.hull(loc1, convex = 0.25)
# triangulation of the space, to solve spde using numerical methods
mesh <- inla.mesh.2d(rbind(loc1, loc2), boundary = list(bnd1, bnd2),
                     max.edge = c(0.05, 0.2), cutoff = 0.005)
A1 <- inla.spde.make.A(mesh, loc1)
A2 <- inla.spde.make.A(mesh, loc2)
spde <- inla.spde2.pcmatern(mesh, alpha = 2,
                            prior.range = c(0.5, 0.01),
                            prior.sigma = c(1, 0.01))
hyper <- list(theta = list(prior = 'normal', param = c(0, 10)))
hyper.eps <- list(hyper = list(theta = list(prior = 'pc.prec',
  param = c(1, 0.01))))

formula1 <- wind ~ 0 +
  a1 +
  a2 +
  u10 +
  fsr +
  blh +
  z +
  sp +
  tpi_500 +
  aspect +
  ns(month, df = 10) +
  f(hour, model = "rw1", replicate = station_id) +
  f(s1, model = spde) +
  f(s2, model = spde) +
  f(s12, copy = "s1", fixed = FALSE, hyper = hyper)

# Creation of stack data for x and y
stk1 <- inla.stack(data = list(wind = cbind(as.vector(out.data$u10_hr), NA)),
                   A = list(A1, 1),
                   effect = list(s1 = 1:spde$n.spde, x1))
stk2 <- inla.stack(data = list(wind = cbind(NA, as.vector(out.data$v10_hr))),
                   A = list(A2, 1),
                   effect = list(list(s2 = 1:spde$n.spde, s12 = 1:spde$n.spde), x2))
stack <- inla.stack(stk1, stk2)
model2 <- inla(formula1, rep('gaussian', 2),
  data = inla.stack.data(stack),
  control.family = list(hyper.eps, hyper.eps),
  control.predictor = list(A = inla.stack.A(stack)),
  control.inla = list(int.strategy = 'eb'), verbose=TRUE)

