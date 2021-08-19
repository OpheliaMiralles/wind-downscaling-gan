# Title     :  Multivariate regression model with additive / multiplicative bias depending on covariates
# Objective : Downscaling wind speed to specific points
# Created by: opheliamiralles
# Created on: 24.06.21

library(INLA)
library(splines)
inla.pardiso.check()

data.path <- "/Users/Boubou/Documents/GitHub/WindDownscaling_EPFL_UNIBE/data/point_prediction_files/dataframe_6months.csv"
out.data <- read.csv(data.path)
out.data <- out.data[out.data$month %in% c(9,10),]
out.data <- na.omit(out.data)
# Outputs
n <- nrow(out.data)
Y <- matrix(NA, n * 2, 2)
Y[1:n, 1] <- out.data$u10
Y[(1 + n):(2 * n), 1] <- NA # faked observations
Y[(1 + n):(2 * n), 2] <- out.data$v10   # actual observations
Y[1:n, 2] <- NA # faked observations
Y <- data.frame(Y)

# Inputs
x1 <- data.frame(a1 = 1, windlr = -out.data$U_10M,
                hour = out.data$hour,
                month = out.data$month,
                tpi_500 = out.data$tpi_500,
                aspect = out.data$aspect,
                slope = out.data$slope,
                 sn_derivative = out.data$sn_derivative,
                 we_derivative = out.data$we_derivative,
                 ridge_index_norm = out.data$ridge_index_norm,
                 ridge_index_dir = out.data$ridge_index_dir,
                 station_id = out.data$station_id,
                lon = out.data$lon,
                lat = out.data$lat)
x2 <- data.frame(a2 = 1,  windlr = out.data$V_10M,
                hour = out.data$hour,
                month = out.data$month,
                tpi_500 = out.data$tpi_500,
                aspect = out.data$aspect,
                slope = out.data$slope,
                 sn_derivative = out.data$sn_derivative,
                 we_derivative = out.data$we_derivative,
                 ridge_index_norm = out.data$ridge_index_norm,
                 ridge_index_dir = out.data$ridge_index_dir,
                 station_id = out.data$station_id,
                lon = out.data$lon,
                lat = out.data$lat)

# Creation of mesh, same locations for u10 and v10
loc1 <- cbind(x1$lon, x1$lat)
loc2 <- cbind(x2$lon, x2$lat)
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
  windlr +
  tpi_500 +
  #ns(month, df = 10) +
  f(hour, model = "rw1", replicate = station_id) +
  f(s1, model = spde) +
  f(s2, model = spde)

# Creation of stack data for x and y
stk1 <- inla.stack(data = list(wind = cbind(as.vector(out.data$u10), NA)),
                   A = list(A1, 1),
                   effect = list(s1 = 1:spde$n.spde, x1))
stk2 <- inla.stack(data = list(wind = cbind(NA, as.vector(out.data$v10))),
                   A = list(A2, 1),
                   effect = list(list(s2 = 1:spde$n.spde, s12 = 1:spde$n.spde), x2))
stack <- inla.stack(stk1, stk2)
model2 <- inla(formula1, rep('gaussian', 2),
  data = inla.stack.data(stack),
  control.family = list(hyper.eps, hyper.eps),
  control.predictor = list(A = inla.stack.A(stack)),
  control.inla = list(int.strategy = 'eb'), verbose=TRUE)

save.image("/Users/Boubou/Documents/GitHub/WindDownscaling_EPFL_UNIBE/results/INLA/multivariate_regression_with_randomeffect.RData")

# Visualising random effect
gproj <- inla.mesh.projector( ## projector builder
  mesh, ## mesh used to define the model
  dims = c(500, 500)) ## grid dimension
## project the mean and the SD
g.mean <- inla.mesh.project(gproj, model2$summary.random$s2$mean)
g.sd <- inla.mesh.project(gproj, model2$summary.random$s2$sd)
par(mfrow = c(1, 2), mar = c(0, 0, 1, 0))
require(fields)
image.plot(g.mean, asp = 1, main = 'RF for U and V components: posterior mean', axes = FALSE, horizontal = TRUE)
image.plot(g.sd, asp = 1, main = 'RF for U and V components: posterior SD', axes = FALSE, horizontal = TRUE)

# Prediction
data.path.test <- "/Users/Boubou/Documents/GitHub/WindDownscaling_EPFL_UNIBE/data/point_prediction_files/dataframe_6months.csv"
in.data <- read.csv(data.path.test)
in.data <- in.data[in.data$month ==11,]

loc1.pred <- cbind(in.data$lon, in.data$lat)
loc2.pred <- cbind(in.data$lon, in.data$lat)
dim(A1.pred <- inla.spde.make.A(mesh = mesh, loc = loc1.pred))
dim(A2.pred <- inla.spde.make.A(mesh = mesh, loc = loc2.pred))
x1.pred <- data.frame(a1 = 1, windlr = -in.data$U_10M,
                hour = in.data$hour,
                month = in.data$month,
                tpi_500 = in.data$tpi_500,
                aspect = in.data$aspect,
                slope = in.data$slope,
                 sn_derivative = in.data$sn_derivative,
                 we_derivative = in.data$we_derivative,
                 ridge_index_norm = in.data$ridge_index_norm,
                 ridge_index_dir = in.data$ridge_index_dir,
                 station_id = in.data$station_id,
                lon = in.data$lon,
                lat = in.data$lat)
x2.pred <-  data.frame(a1 = 1, windlr = in.data$V_10M,
                hour = in.data$hour,
                month = in.data$month,
                tpi_500 = in.data$tpi_500,
                aspect = in.data$aspect,
                slope = in.data$slope,
                 sn_derivative = in.data$sn_derivative,
                 we_derivative = in.data$we_derivative,
                 ridge_index_norm = in.data$ridge_index_norm,
                 ridge_index_dir = in.data$ridge_index_dir,
                 station_id = in.data$station_id,
                lon = in.data$lon,
                lat = in.data$lat)
stk1.pred <- inla.stack(data = list(wind.1 = NA, wind.2 = NA),
                   A = list(A1.pred, 1),
                   tag='pred1',
                   effect = list(s1 = 1:spde$n.spde, x1.pred))
stk2.pred <- inla.stack(data = list(wind.1 = NA, wind.2 = NA),
                   A = list(A2.pred, 1),
                   tag='pred2',
                   effect = list(list(s2 = 1:spde$n.spde, s12 = 1:spde$n.spde), x2.pred))
stack.pred <- inla.stack(stk1.pred, stk2.pred)
stk.full <- inla.stack(stack, stack.pred) ## join both data


p.res <- inla(
  formula1, rep('gaussian', 2), data = inla.stack.data(stk.full), ## using the full data
  control.family = list(hyper.eps, hyper.eps),
  control.predictor = list(compute = TRUE, ## compute the predictor
                           A = inla.stack.A(stk.full)), ## from full data
  control.mode = list(theta = model2$mode$theta), verbose = TRUE) ## use the mode previously found

pred.ind <- inla.stack.index( ## stack index extractor function
  stk.full, ## the data stack to be considered
  tag = 'pred1' ## which part of the data to look at
)$data
pred.ind.red <- array(pred.ind[1:100]) ## which elements to collect
ypost <- p.res$marginals.fitted.values[pred.ind.red]
par(mfrow = c(1, 3), mar = c(3, 3, 2, 1), mgp = c(2, 1, 0))
observed <- in.data$u10
order <- order(p.res$summary.fitted.values$mean[pred.ind.red])
plot(observed[order], pch=0, cex=0.5, ylab='observed')
segments(1:length(observed), p.res$summary.fitted.val$`0.025quant`[pred.ind.red][order],
         1:length(observed), p.res$summary.fitted.val$`0.975quant`[pred.ind.red][order])