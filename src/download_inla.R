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

# Trial example

library(INLA)
data(Leuk)
loc <- cbind(Leuk$xcoord, Leuk$ycoord)
bnd1 <- inla.nonconvex.hull(loc, convex = 0.05)
bnd2 <- inla.nonconvex.hull(loc, convex = 0.25)
mesh <- inla.mesh.2d(loc, boundary = list(bnd1, bnd2),
                     max.edge = c(0.05, 0.2), cutoff = 0.005)
# triangulates the space, to solve spde using numerical methods
A <- inla.spde.make.A(mesh, loc)
# defines the mapping matrix between nodes for spde and initial locations
spde <- inla.spde2.matern(mesh, alpha = 2)
# gets the approximate solution for a Matern field on this graph
# B0, B1 and B2 are basis functions for "phi0, phi1 and phi2"?
# I guess then M0, M1 and M2 are precision matrices for "phi0, phi1 and phi2"
# Oh ok so maybe phi1 and phi2 are <psi, psi> and <delta psi, delta psi>
# B.tau, B.kappa are parameters used to adapt to non-stationary cases https://rdrr.io/github/andrewzm/INLA/man/inla.spde2.matern.html
# theta 1 and theta 2 are tau and kappa estimates + precision matrix (diagonal as they are assumed independent in this case)
# mesh in dim 3 as a default; 3rd component null as data in 2D?
# basis functions (sum phi_k for k in range (n)) are 1 for k and 0 for other vertices, linear over triangles
formula <- inla.surv(time, cens) ~ 0 +
  a0 +
  sex +
  age +
  wbc +
  tpi +
  f(spatial, model = spde)
stk <- inla.stack(data = list(time = Leuk$time, cens = Leuk$cens), A = list(A, 1),
                  effect = list(list(spatial = 1:spde$n.spde),
                                data.frame(a0 = 1, Leuk[, -c(1:4)])), remove.unused = FALSE)
r <- inla(formula, family = "weibull", data = inla.stack.data(stk),
          control.predictor = list(A = inla.stack.A(stk)), verbose = TRUE)


# Trial example 2
formula1=inla.surv(time,cens)~sex+

         age +

         wbc +

         f(inla.group(tpi,n=50),model="rw2")

model1 = inla(formula1, family="coxph",

  data=Leuk, verbose=TRUE,

  control.hazard=list(model="rw1",n.intervals=20,param=c(1,0.001)))