# Title     :  Simple regression model with additive / multiplicative bias depending on covariates
# Objective : Downscaling wind speed to specific points
# Created by: Boubou
# Created on: 17.06.21

library(INLA)
inla.pardiso.check()
data.path <- "/Users/Boubou/Documents/GitHub/WindDownscaling_EPFL_UNIBE/data/point_prediction_files/dataframe_6months.csv"
out.data <- read.csv(data.path)
loc <- cbind(out.data$lon, out.data$lat)
bnd1 <- inla.nonconvex.hull(loc, convex = 0.05)
bnd2 <- inla.nonconvex.hull(loc, convex = 0.25)
mesh <- inla.mesh.2d(loc, boundary = list(bnd1, bnd2),
                     max.edge = c(0.05, 0.2), cutoff = 0.005)
# triangulates the space, to solve spde using numerical methods
A <- inla.spde.make.A(mesh, loc)
# defines the mapping matrix between nodes for spde and initial locations
spde <- inla.spde2.matern(mesh, alpha = 2)
formula1 <- u10_hr ~ 0 + a0 + u10 + f(spatial, model = spde)
stk <- inla.stack(data = list(u10_hr = out.data$u10_hr), A = list(A, 1),
                  effect = list(spatial = 1:spde$n.spde,
                                data.frame(a0 = 1, u10 = out.data$u10)))
model1 <- inla(formula1, family = "gaussian",
               data = inla.stack.data(stk), control.predictor = list(A = inla.stack.A(stk)), verbose = TRUE)

# Visualising random effect
gproj <- inla.mesh.projector( ## projector builder
  mesh, ## mesh used to define the model
  dims = c(300, 300)) ## grid dimension
## project the mean and the SD
g.mean <- inla.mesh.project(gproj, model1$summary.random$spatial$mean)
g.sd <- inla.mesh.project(gproj, model1$summary.random$spatial$sd)
par(mfrow = c(1, 2), mar = c(0, 0, 1, 0))
require(fields)
image.plot(g.mean, asp = 1, main = 'RF posterior mean', axes = FALSE, horizontal = TRUE)
image.plot(g.sd, asp = 1, main = 'RF posterior SD', axes = FALSE, horizontal = TRUE)

# Prediction
data.path.test <- "/Users/Boubou/Documents/GitHub/WindDownscaling_EPFL_UNIBE/data/point_prediction_files/dataframe_6months_test.csv"
in.data <- read.csv(data.path.test)
loc.pred <- cbind(in.data$lon, in.data$lat)
dim(A.pred <- inla.spde.make.A(mesh = mesh, loc = loc.pred))
stk.pred <- inla.stack(data = list(u10_hr = NA), A = list(A.pred, 1),
                       tag = 'pred',
                       effect = list(spatial = 1:spde$n.spde,
                                     data.frame(a0 = 1, u10 = in.data$u10)))
stk.full <- inla.stack(stk, stk.pred) ## join both data

inla.setOption(pardiso.license = '~/licenses/pardiso.lic')
inla.pardiso()
inla.pardiso.check()
p.res <- inla(
  formula1, data = inla.stack.data(stk.full), ## using the full data
  control.predictor = list(compute = TRUE, ## compute the predictor
                           A = inla.stack.A(stk.full)), ## from full data
  control.mode = list(theta = model1$mode$theta), verbose = TRUE,
  tag = 'pred') ## use the mode previously found

pred.ind <- inla.stack.index( ## stack index extractor function
  stk.full, ## the data stack to be considered
  tag = 'pred' ## which part of the data to look at
)$data ## which elements to collect
ypost <- p.res$marginals.fitted.values[pred.ind]
xyl <- apply(Reduce('rbind', ypost), 2, range)
par(mfrow = c(1, 3), mar = c(3, 3, 2, 1), mgp = c(2, 1, 0))
for (j in 1:3)
  plot(ypost[[j]], type = 'l', xlim = xyl[, 1], ylim = xyl[, 2],
       xlab = 'y', ylab = 'density', main = paste0('y', j))