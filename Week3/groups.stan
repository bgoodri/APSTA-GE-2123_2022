data {              // saved as "groups.stan"
  int<lower = 0> N; // number of observations
  int<lower = 0> K; // number of predictors
  matrix[N, K] X;   // matrix of predictors
  vector[N] y;      // outcomes
  int<lower = 1> J; // number of groups
  int<lower = 1, upper = J> group[N];     // group membership
  int<lower = 0, upper = 1> prior_only;   // ignore data?
  vector[K + 2] m;                        // prior means
  vector<lower = 0>[K + 2] scale;         // prior scales
}
parameters {
  vector[K] beta;
  vector[J] alpha;
  real<lower = 0> sigma;
}
model {
  if (!prior_only) target += normal_id_glm_lpdf(y | X, alpha[group], beta, sigma);
  target += normal_lpdf(beta  | m[1:K],   scale[1:K]); // ^ important
  target += normal_lpdf(alpha | m[K + 1], scale[K + 1]);
  target += normal_lpdf(sigma | m[K + 2], scale[K + 2]); // actually half normal
}
generated quantities {
  vector[N] log_lik;
  {
    vector[N] mu = alpha[group] + X * beta;
    for (n in 1:N) log_lik[n] = normal_lpdf(y[n] | mu[n], sigma);
  }
}
