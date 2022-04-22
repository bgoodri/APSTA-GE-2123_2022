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
  real<lower = 0> r; // prior rate
}
parameters {
  vector[2] mu;
  vector<lower = 0>[2] sigma;
  real<lower = -1, upper = 1> rho;
  
  vector[J] a;
  vector[J] b;
}
transformed parameters {
  vector[J] alpha = mu[1] + sigma[1] * a; // implies alpha ~ normal(mu[1], sigma[2])
  vector[J] beta  = mu[2] + sigma[2] / sigma[1] * rho * (alpha - mu[1]) +
                    sigma[2] * sqrt((1 + rho) * (1 - rho)) * b; 
                    // implies beta | alpha ~ normal(...)
} // alpha[j] and beta[j] are jointly biviariate normal under the prior implied by a and b
model {
  if (!prior_only) {
    vector[N] eta = alpha[discipline] + beta[discipline] .* female; // elementwise multiplication
    target += binomial_logit_lpmf(awards | applications, eta);
  }
  target += std_normal_lpdf(a);
  target += std_normal_lpdf(b);
  target += normal_lpdf(mu | m, s);
  target += exponential_lpdf(sigma | r);
  // implicit: rho ~ uniform(-1, 1)
}
