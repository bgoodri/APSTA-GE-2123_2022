data {              // saved as "generated_quantities.stan"
  int<lower = 0> N; // number of observations
  int<lower = 0> K; // number of predictors
  matrix[N, K] X;   // matrix of predictors
  vector[N] y;      // outcomes
  /* prior hyperparameters are not needed anymore */
}
parameters {
  real alpha;
  vector[K] beta;
  real<lower = 0> sigma;
}
generated quantities {
  vector[N] log_lik;
  { // mu is not stored because it is in this local block
    vector[N] mu = alpha + X * beta;
    for (n in 1:N) log_lik[n] = normal_lpdf(y[n] | mu[n], sigma);
  }
}
