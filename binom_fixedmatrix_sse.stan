
data {
  int<lower=0> n_obs;                             // Number of observations (total samples across all sites):
  int<lower=1> n_pred;                            // Number of predictor variables in matrix u
  int<lower=1> n_taxa;                            // Number of taxa in matrix pa:
  int<lower=1> n_site;                            // Number of sites:
  matrix[n_obs,n_pred] u;                         // Matrix of predictor variables for each sample (2 per site):
  array[n_obs,n_taxa] int<lower=0,upper=1> pa;    // Presence-absence matrix pa for all taxa:
  array[n_obs,n_taxa] real s;                     // Subsample proportion for each sample (all coarsepicks = 1):
  array[n_obs] int site;                          // Site number for each sample:
}
parameters {
  vector[n_pred] mu_gamma;                        // Means of fixed-effect beta parameter hyperdistributions
  matrix[n_pred,n_taxa] gamma;                    // Individual beta parameters of fixed effects in u
  matrix[n_site,n_taxa] a_site_raw;               // Raw coefficient of random site effect
  matrix<upper=5>[n_obs,n_taxa] eps_raw;          // Raw residual error parameter
  vector<lower=0>[n_taxa] sigma_site;             // SD of hyperdistribution of a_site among taxa
  vector<lower=0>[n_taxa] sd_lam;                 // SD of hyperdistribution of eps among taxa
  corr_matrix[n_pred] Omega;                      // Regularising hyperprior for correlation matrix among taxa
  vector<lower=0>[n_pred] tau;                    // Prior scale for correlation matrix Omega
}
transformed parameters {
  matrix[n_obs,n_taxa] alpha;                     // linear probability of occurrence matrix 
  matrix[n_obs,n_taxa] eps;                       // Coefficient of residual error
  matrix[n_site, n_taxa] a_site;                  // Coefficient of random site effect
  for (j in 1:n_taxa){
    eps[,j] = sd_lam[j] * eps_raw[,j];            // Non-centered error term (eps)
    a_site[,j] = sigma_site[j] * a_site_raw[,j];  // Non-centered random site effect (a_site)
  }
// Non-centered parameterisation for hierarchical models fit with Hamiltonian Monte Carlo
// (e.g.) a_site_raw ~ std_normal() implies a_site ~ normal(0, sigma_site)
// See Neal's Funnel: https://mc-stan.org/docs/stan-users-guide/reparameterization.html
  for (i in 1:n_obs){
    for (j in 1:n_taxa){
      // Model for linear predictor (alpha)
      alpha[i,j] = a_site[site[i],j] + u[i,] * gamma[,j] + eps[i,j];
    }
  }
}
model {
  // Priors
  mu_gamma ~ normal(0, 5);
  to_vector(a_site_raw) ~ std_normal();
  to_vector(eps_raw) ~ normal(0, 1);
  sigma_site ~ normal(0, 2);
  sd_lam ~ normal(1,1);
  tau ~ exponential(1);
  Omega ~ lkj_corr(2);

  // Estimation of correlated beta-coefficient parameters in gamma:
  for (j in 1:n_taxa){
    target += multi_normal_prec_lpdf(gamma[,j] | mu_gamma, quad_form_diag(Omega, tau));
  }
  
  // Estimated taxon occurrence probability  
  for (i in 1:n_obs){
    for (j in 1:n_taxa){
      target += binomial_logit_lpmf(pa[i,j] | 1, alpha[i,j] + log(s[i,j]));
      
  // This parameterization adds the marginal log-binomial-probability 
  // resulting from subsampling error to the marginal inverse-logit 
  // probability of the linear model.  It is equivalent to a (50 times) slower 
  // parameterization modelling the marginal log-binomial and inverse-logit
  // probabilities separately, by looping through all feasible total counts
  // given each count and subsample proportion.
    }
  }
}
generated quantities {
  // log-likelihood was only used for model comparisons during model development.
  matrix[n_obs,n_taxa] log_lik;
  for (i in 1:n_obs){
    for (j in 1:n_taxa){
      log_lik[i,j] = binomial_logit_lpmf(pa[i,j] | 1, alpha[i,j] + log(s[i,j]));
    }
  }
}
