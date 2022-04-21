data {
  int<lower = 0> N; // number of observations
  int<lower = 1> J; // number of disciplines
  int<lower = 1, upper = J> discipline[N];
  vector<lower = 0, upper = 1>[N] female;
  int<lower = 0> applications[N];
  int<lower = 0, upper = max(applications)> awards[N];
  
  int<lower = 0, upper = 1> prior_only;
  real m;            // prior me{di}an
  real<lower = 0> s; // prior standard deviation
}
parameters {
  vector[J] alpha;
  real beta;
}
model {
  if (!prior_only) {
    vector[N] eta = alpha[discipline] + beta * female;
    target += binomial_logit_lpmf(awards | applications, eta);
  }
  target += normal_lpdf(alpha | m, s);
  target += normal_lpdf(beta  | m, s);
}
