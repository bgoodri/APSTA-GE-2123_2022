data {              // saved as "beta.stan"
  int<lower = 0> N; // number of observations
  int<lower = 0> K; // number of predictors
  matrix[N, K] X;   // matrix of predictors
  vector<lower = 0, upper = 1>[N] y;      // outcomes
  int<lower = 1> J;                       // number of groups
  int<lower = 1, upper = J> group[N];     // group membership
  int<lower = 0, upper = 1> prior_only;   // ignore data?
  vector[K + 1] m;                        // prior means
  vector<lower = 0>[K + 1] scale;         // prior scales
  vector<lower = 0>[2] rate;              // prior rates
}
parameters {
  vector[K] beta;
  real mu_alpha;
  real<lower = 0> sigma;
  vector[J] alpha;
  real<lower = 0> kappa;
}
model {
  if (!prior_only) {
    vector[N] mu = inv_logit(alpha[group] + X * beta);
    target += beta_proportion_lpdf(y | mu, kappa);
  }
  target += normal_lpdf(beta  | m[1:K],   scale[1:K]);
  target += normal_lpdf(mu_alpha | m[K + 1], scale[K + 1]);
  target += normal_lpdf(alpha | mu_alpha, sigma);
  target += exponential_lpdf(sigma | rate[1]);
  target += exponential_lpdf(kappa | rate[2]);
}
generated quantities {
  vector[N] y_rep;
  {
    vector[N] mu = inv_logit(alpha[group] + X * beta);
    for (n in 1:N) y_rep[n] = beta_proportion_rng(mu[n], kappa);
  }
}
