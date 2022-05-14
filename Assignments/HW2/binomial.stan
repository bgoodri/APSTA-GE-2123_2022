data {              // saved as "binomial.stan"
  int<lower = 0> N; // number of observations
  int<lower = 0> K; // number of predictors
  matrix[N, K] X;   // matrix of predictors
  int<lower = 0> y[N];                    // outcomes
  int<lower = 0> pop[N];                  // population
  int<lower = 1> J;                       // number of groups
  int<lower = 1, upper = J> group[N];     // group membership
  int<lower = 0, upper = 1> prior_only;   // ignore data?
  vector[K + 1] m;                        // prior means
  vector<lower = 0>[K + 1] scale;         // prior scales
  real<lower = 0> rate;                   // prior rates
}
parameters {
  vector[K] beta;
  real mu_alpha;
  real<lower = 0> sigma;
  vector[J] alpha;
}
model {
  if (!prior_only) target += binomial_logit_lpmf(y | pop, alpha[group] + X * beta);
  target += normal_lpdf(beta  | m[1:K],   scale[1:K]);
  target += normal_lpdf(mu_alpha | m[K + 1], scale[K + 1]);
  target += normal_lpdf(alpha | mu_alpha, sigma);
  target += exponential_lpdf(sigma | rate);
}
generated quantities {
  vector[N] y_rep;
  {
    vector[N] mu = inv_logit(alpha[group] + X * beta);
    for (n in 1:N) {
      y_rep[n] = binomial_rng(pop[n], mu[n]);
      y_rep[n] /= pop[n];
    }
  }
}
