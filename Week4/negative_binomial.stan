#include quantile_functions.stan
data {
  int<lower = 0> N;     // number of observations
  vector[N] offset;     // a predictor with a coefficient of 1
  int<lower = 0> K;     // number of other predictors
  matrix[N, K] X;       // matrix of other predictors
  int<lower = 0> y[N];  // outcomes
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
  real phi   = gld_qf(p[K + 2], m[K + 2], r[K + 2], a[K + 2], s[K + 2]);
  vector[K] beta; // as yet undefined
  for (k in 1:K) beta[k] = gld_qf(p[k], m[k], r[k], a[k], s[k]); // now defined
}
model {
  if (!prior_only) 
    target += neg_binomial_2_log_glm_lpmf(y | X, alpha + offset, beta, phi);
} // implicit: p ~ uniform(0, 1)
generated quantities {
  vector[N] log_lik;
  int y_rep[N];
  {
    vector[N] eta = alpha + offset + X * beta;
    for (n in 1:N) {
      log_lik[n] = neg_binomial_2_log_lpmf(y[n] | eta[n], phi);
      y_rep[n] = neg_binomial_2_log_rng(eta[n], phi);
    }
  }
}
