#include quantile_functions.stan
data {
  int<lower = 0> N; // number of observations
  int<lower = 0> K; // number of predictors
  matrix[N, K] X;   // matrix of predictors
  vector[N] y;      // outcomes
  int<lower = 0, upper = 1> prior_only;   // ignore data?
  vector[K + 2] m;                        // prior medians
  vector<lower = 0>[K + 2] r;             // prior IQRs
  vector<lower = -1, upper = 1>[K + 2] a; // prior asymmetry
  vector<lower =  0, upper = 1>[K + 2] s; // prior steepness
}
parameters {
  vector<lower = 0, upper = 1>[K + 2] p;  // CDF values
}
transformed parameters {
  real alpha = gld_qf(p[K + 1], m[K + 1], r[K + 1], a[K + 1], s[K + 1]);
  vector[K] beta; // as yet undefined
  real<lower = 0> sigma = gld_qf(p[K + 2], m[K + 2], r[K + 2], a[K + 2], s[K + 2]);
  for (k in 1:K) beta[k] = gld_qf(p[k], m[k], r[k], a[k], s[k]); // now defined
}
model { // log likelihood, equivalent to target += normal_lpdf(y | alpha + X * beta, sigma)
  if (!prior_only) target += normal_id_glm_lpdf(y | X, alpha, beta, sigma);
} // implicit: p ~ uniform(0, 1)
