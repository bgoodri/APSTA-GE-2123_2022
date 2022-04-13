data {
  int<lower = 0> N;        // number of observations
  int<lower = 0> y[N];     // outcomes
  int<lower = 0> n[N];     // trials
  int<lower = 1> J[5];     // number of levels of each predictor
  int<lower = 1, upper = J[1]> Region[N];
  int<lower = 1, upper = J[2]> Gender[N];
  int<lower = 1, upper = J[3]> Urban_Density[N];
  int<lower = 1, upper = J[4]> Age[N];
  int<lower = 1, upper = J[5]> Income[N];
  int<lower = 0, upper = 1> prior_only;   // ignore data?
  vector[5] m;                            // prior me{di}ans
  vector<lower = 0>[5] rate;              // prior rates
}
parameters {
  vector<lower = 0>[5] sigma;
  vector[J[1]] beta_Region_;
  vector[J[2]] beta_Gender_;
  vector[J[3]] beta_Urban_Density_;
  vector[J[4]] beta_Age_;
  vector[J[5]] beta_Income_;
}
transformed parameters {
  vector[J[1]] beta_Region = m[1] + sigma[1] * beta_Region_;
  vector[J[2]] beta_Gender = m[2] + sigma[2] * beta_Gender_;
  vector[J[3]] beta_Urban_Density = m[3] + sigma[3] * beta_Urban_Density_;
  vector[J[4]] beta_Age = m[4] + sigma[4] * beta_Age_;
  vector[J[5]] beta_Income = m[5] + sigma[5] * beta_Income_;
}
model {
  if (!prior_only) {
    vector[N] eta = beta_Region[Region] + beta_Gender[Gender] +
      beta_Urban_Density[Urban_Density] + beta_Age[Age] + beta_Income[Income];
    target += binomial_logit_lpmf(y | n, eta);
  }
  
  target += std_normal_lpdf(beta_Region_);
  target += std_normal_lpdf(beta_Gender_);
  target += std_normal_lpdf(beta_Urban_Density);
  target += std_normal_lpdf(beta_Age_);
  target += std_normal_lpdf(beta_Income_);
  
  target += exponential_lpdf(sigma | rate);
} 
