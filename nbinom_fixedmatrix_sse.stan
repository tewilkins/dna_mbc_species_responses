
data {
  int<lower=0> n_obs;                             // Number of observations (total samples across all sites):
  int<lower=1> n_pred;                            // Number of predictor variables in matrix u
  int<lower=1> n_taxa;                            // Number of taxa in matrix c:
  int<lower=1> n_site;                            // Number of sites:
  matrix[n_obs,n_pred] u;                         // Matrix of predictor variables for each sample (2 per site):
  array[n_obs,n_taxa] int c;                      // Count of each taxon in each sample:
  array[n_obs,n_taxa] real s;                     // Subsample proportion for each sample (all coarsepicks = 1):
  array[n_obs] int site;                          // Site number for each sample:
}
parameters {
  vector[n_pred] mu_gamma;                        // Means of fixed-effect beta parameter hyperdistributions
  matrix[n_pred,n_taxa] gamma;                    // Individual beta parameters of fixed effects in u
  matrix[n_site,n_taxa] a_site_raw;               // Raw coefficient of random site effect
  vector<lower=0>[n_taxa] sigma_site;             // SD of hyperdistribution of a_site among taxa
  corr_matrix[n_pred] Omega;                      // Regularising hyperprior for correlation matrix among taxa
  vector<lower=0>[n_pred] tau;                    // Prior scale for correlation matrix Omega
  vector<lower=0>[n_taxa] inv_phi;
  real<lower=0> mu_inv_phi;
  real<lower=0> sd_inv_phi;
}
transformed parameters {
  matrix[n_obs, n_taxa] log_lambda;               // Log total count
  matrix[n_site, n_taxa] a_site;                  // Coefficient of random site effect
  vector<lower=0>[n_taxa] phi;                    // Dispersion parameter for each taxon
  for (j in 1:n_taxa){
    a_site[,j] = sigma_site[j] * a_site_raw[,j];  // Non-centered random site effect (a_site)

  }
// Non-centered parameterisation for hierarchical models fit with Hamiltonian Monte Carlo
// (e.g.) a_site_raw ~ std_normal() implies a_site ~ normal(0, sigma_site)
// See Neal's Funnel: https://mc-stan.org/docs/stan-users-guide/reparameterization.html

  phi = pow(1/inv_phi,2);
// See https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations
// And https://statmodeling.stat.columbia.edu/2018/04/03/justify-my-love/

  for (i in 1:n_obs){
    for (j in 1:n_taxa){
      // Model for linear predictor (log_lambda):
      log_lambda[i,j] = a_site[site[i],j] + u[i,] * gamma[,j];
    }
  }
}
model {
  // Priors
   mu_gamma ~ normal(0 , 5);
   to_vector(a_site_raw) ~ std_normal();
   sigma_site ~ normal(0, 2);
   inv_phi ~ normal(mu_inv_phi, sd_inv_phi);
   sd_inv_phi ~ normal(0, 1);
   mu_inv_phi ~ normal(0, 1);
   tau ~ exponential(1);
   Omega ~ lkj_corr(2);
   
  // Estimation of correlated beta-coefficient parameters in gamma:
   for (j in 1:n_taxa){
     target += multi_normal_prec_lpdf(gamma[,j] | mu_gamma, quad_form_diag(Omega, tau));
   }
   
  // Estimated taxon abundance  
   for (i in 1:n_obs){
     for (j in 1:n_taxa){
       target += neg_binomial_2_log_lpmf(c[i,j] | log_lambda[i,j] + log(s[i,j]), phi[j]);
       
  // This parameterization adds the marginal log-binomial-probability 
  // resulting from subsampling error to the marginal negative-binomial 
  // probability of the linear model.  It is equivalent to a (50 times) slower 
  // parameterization modelling the marginal binomial and negative-binomial
  // probabilities separately, by looping through all feasible total counts
  // given each count and subsample proportion.
     }
   }
}
generated quantities {
  // log-likelihood was only used for model comparisons during model development.
  matrix[n_obs,n_taxa] log_lik; // Estimate of true total abandance in each sample
  for (i in 1:n_obs){
    for (j in 1:n_taxa){
      log_lik[i,j] = neg_binomial_2_log_lpmf(c[i,j] | log_lambda[i,j] + log(s[i,j]), phi[j]);
    }
  }
}
