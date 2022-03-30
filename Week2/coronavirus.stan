#include quantile_functions.stan
data { /* these are known and passed as a named list from R */
  int<lower = 0> n;                          // number of tests
  int<lower = 0, upper = n> y;               // number of positives
  real m;
  real<lower = 0> IQR;
  real<lower = -1, upper = 1> asymmetry;
  real<lower =  0, upper = 1> steepness;
}
parameters { /* these are unknowns whose posterior distribution is sought */
  real<lower = 0, upper = 1> p;             // CDF of positive test probability
}
transformed parameters { /* deterministic unknowns that get stored in RAM */
  real VE = GLD_qf(p, m, IQR, asymmetry, steepness); // positive probability
} // this function ^^^ is defined in the quantile_functions.stan file
model { /* log-kernel of Bayes' Rule that essentially returns "target" */
  target += binomial_lpmf(y | n, (VE - 1) / (VE - 2)); // log-likelihood
} // implicit: p ~ uniform(0, 1) <=> theta ~ GLD(m, IQR, asymmetry, steepness)
