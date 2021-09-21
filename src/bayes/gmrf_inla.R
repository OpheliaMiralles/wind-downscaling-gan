# Title     :  Simple regression model with additive / multiplicative bias depending on covariates
# Objective : Downscaling wind speed to specific points
# Created by: Boubou
# Created on: 17.06.21

library(INLA)
inla.pardiso.check()
data.path <- "/Users/opheliamiralles/Desktop/PhD/EPFL/WindDownscaling_EPFL_UNIBE/data/point_prediction_files/dataframe_6months_test.csv"
out.data <- read.csv(data.path)

# Creation of spatial mesh with data fixed in time
t0.data <- out.data[out.data$time== 0,]
loc <- cbind(t0.data$lon, t0.data$lat)
#bnd1 <- inla.nonconvex.hull(loc, convex = 0.05, resolution=c(92, 43))
mesh <- inla.mesh.2d(loc,
                     max.edge = c(0.1, 0.2), offset = c(0.1, 0.2))

# Creation of temporal mesh with knots every 6 hours
loc0.data <- out.data[out.data$station=='ABO',]
mesh.t <- inla.mesh.1d(seq(0, max(loc0.data$time), by=6))
nb.groups <- mesh.t$m

# Creation of A matrix and spatial field indices per group
spde <- inla.spde2.pcmatern(mesh = mesh,
  prior.range = c(0.5, 0.01), # P(range < 0.05) = 0.01
  prior.sigma = c(10, 0.01)) # P(sigma > 1) = 0.01
iset <- inla.spde.make.index("spatial.field", spde$n.spde, n.group = nb.groups)
A <- inla.spde.make.A(mesh = mesh, loc = cbind(out.data$lon, out.data$lat),
  group = out.data$time, group.mesh = mesh.t)

# Stack all data
stk <- inla.stack(data = list(u10_hr = out.data$u10_hr),
                    A = list(A, 1),
                  effect = list(c(list(a0=1), iset),
                                list(u10 = out.data$u10)))
# Prior for AR1 effect, specifying that P(cor > 0) = 0.9
h.spec <- list(rho = list(prior = 'pc.cor1', param = c(0, 0.9)))
formula1 <- u10_hr ~ 0 + a0 + u10 + f(spatial.field,
                                      model = spde,
                                      group = spatial.field.group,
  control.group = list(model = 'ar1', hyper = h.spec))
# PC prior on the autoreg. param.
prec.prior <- list(prior = 'pc.prec', param = c(1, 0.01))
model1 <- inla(formula1, family = "gaussian",
               data = inla.stack.data(stk),
               control.predictor = list(A = inla.stack.A(stk)),
               control.family = list(hyper = list(prec = prec.prior)),
               control.inla = list(int.strategy = 'simplified.laplace', huge=TRUE),
               verbose = TRUE)

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
  control.mode = list(theta = model1$mode$theta), verbose = TRUE) ## use the mode previously found

pred.ind <- inla.stack.index( ## stack index extractor function
  stk.full, ## the data stack to be considered
  tag = 'pred' ## which part of the data to look at
)$data
pred.ind.red <- array(pred.ind[1:100]) ## which elements to collect
ypost <- p.res$marginals.fitted.values[pred.ind.red]
par(mfrow = c(1, 3), mar = c(3, 3, 2, 1), mgp = c(2, 1, 0))
observed <- in.data$u10_hr
order <- order(p.res$summary.fitted.values$mean[pred.ind.red])
plot(observed[order], pch=0, cex=0.5, ylab='observed')
segments(1:length(observed), p.res$summary.fitted.val$`0.025quant`[pred.ind.red][order],
         1:length(observed), p.res$summary.fitted.val$`0.975quant`[pred.ind.red][order])

ggplot_inla_residuals <- function(inla.model, observed, binwidth = NULL){
if(is.null(inla.model$marginals.fitted.values)) stop('No fitted values to plot')
if(any(is.na(inla.model$misc$linkfunctions$link))){
warning('Fitted values from the INLA model may have been returned on the linear, rather than link scale. Use `control.predictor = list(link = 1)` to make sure all fitted values are on the natural scale.')
}
predicted.p.value <- c()
n <- length(observed)
for(i in (1:n)){
predicted.p.value[i] <- INLA::inla.pmarginal(q = observed[i], marginal = inla.model$marginals.fitted.values[[i]])
}
df <- data.frame(predicted = inla.model$summary.fitted.values$mean[1:length(observed)],
observed = observed,
lower = inla.model$summary.fitted.values$`0.025quant`[1:length(observed)],
upper = inla.model$summary.fitted.values$`0.975quant`[1:length(observed)],
p.value = predicted.p.value)
min <- min(df[, c('lower', 'observed')])
max <- max(df[, c('upper', 'observed')])
plots <- list()
plots[[1]] <- ggplot2::ggplot(df, ggplot2::aes_string(x = 'predicted.p.value')) +
ggplot2::geom_histogram(binwidth = binwidth)
plots[[2]] <- ggplot2::ggplot(df, ggplot2::aes_string(x = 'predicted', y = 'observed')) +
ggplot2::geom_point() +
ggplot2::geom_abline(slope = 1, intercept = 0) +
ggplot2::labs(y = "Observed", x = "Fitted") +
ggplot2::lims(x = c(min, max), y = c(min, max))
return(invisible(plots))
}

plots <- ggplot_inla_residuals(p.res, observed)
save.image("/Users/Boubou/Documents/GitHub/WindDownscaling_EPFL_UNIBE/results/INLA/simple_regression_with_randomeffect.RData")