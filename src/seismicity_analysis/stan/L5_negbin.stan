// Full Bayesian L5 Model — Negative Binomial (Binned)
// Uses L5 physics for expected counts with NegBin likelihood.
// Var = mu + mu^2/phi; Poisson recovered as phi -> infinity.
//
// Reference: Stafford (2026), ONR T983 Report, Section 7

functions {
    real ml2mw(real ml) {
        return 0.0376 * square(ml) + 0.646 * ml + 0.53;
    }

    real mw2ml(real mw) {
        real a = 0.0376;
        real b = 0.646;
        real c = 0.53 - mw;
        real disc = max(square(b) - 4 * a * c, 0.0);
        return (-b + sqrt(disc)) / (2 * a);
    }

    real grunthal_slope(real ml) {
        return 2 * 0.0376 * ml + 0.646;
    }

    real sigma_conv_mw(real ml) {
        real var_mw = (0.97 * pow(ml, 4) - 12.4 * pow(ml, 3) +
                       58.4 * square(ml) - 120.0 * ml + 921.0) * 1e-4;
        real sigma = sqrt(fmax(var_mw, 0.0));
        return fmin(fmax(sigma, 0.29), 0.34);
    }

    real compute_sigma_total(real ml, real sigma_ml_i, real sigma_round) {
        real sigma_conv = sigma_conv_mw(ml) / fabs(grunthal_slope(ml));
        return sqrt(square(sigma_conv) + square(sigma_ml_i) + square(sigma_round));
    }

    // Expected count in ML bin [ml_lo, ml_hi] from GR integration
    real compute_bin_expected(real beta, real lambda_floor,
                               real ml_lo, real ml_hi,
                               real mw_floor, real mw_max,
                               real sigma_ml_avg, real sigma_round,
                               int n_comp_bins, vector mw_comp_lo,
                               vector mw_comp_hi, vector t_obs,
                               int n_quad) {
        real dmw = (mw_max - mw_floor) / n_quad;
        real total = 0.0;
        real norm = 1.0 - exp(-beta * (mw_max - mw_floor));

        for (k in 1:n_quad) {
            real mw = mw_floor + (k - 0.5) * dmw;
            real f_gr = beta * exp(-beta * (mw - mw_floor)) / norm;

            // Observation time (scatter-IN region uses T_1)
            real T_k = t_obs[1];
            for (j in 1:n_comp_bins) {
                if (mw >= mw_comp_lo[j] && mw < mw_comp_hi[j]) {
                    T_k = t_obs[j];
                    break;
                }
            }

            // P(ml in [ml_lo, ml_hi] | mw)
            real ml_true = mw2ml(mw);
            real sigma_total = compute_sigma_total(ml_true, sigma_ml_avg, sigma_round);
            real p_bin = Phi((ml_hi - ml_true) / sigma_total) -
                         Phi((ml_lo - ml_true) / sigma_total);

            total += lambda_floor * f_gr * T_k * p_bin * dmw;
        }
        return total;
    }
}

data {
    int<lower=1> n_bins;
    vector[n_bins] ml_lo;
    vector[n_bins] ml_hi;
    array[n_bins] int<lower=0> counts;
    real<lower=0> sigma_ml_avg;
    real<lower=0> sigma_round;
    real mw_floor;
    real mw_min;
    real mw_max;
    int<lower=1> n_comp_bins;
    vector[n_comp_bins] mw_comp_lo;
    vector[n_comp_bins] mw_comp_hi;
    vector<lower=0>[n_comp_bins] t_obs;
    int<lower=1> n_quad;
    // Configurable priors
    real beta_prior_mean;
    real<lower=0> beta_prior_sd;
    real lambda_prior_mean;
    real<lower=0> lambda_prior_sd;
}

transformed data {
    real LOG10 = log(10.0);
    int N_total = sum(counts);
}

parameters {
    real<lower=0.5, upper=5.0> beta;
    real<lower=0.001, upper=1000.0> lambda_floor;
    real<lower=0.1> phi;    // Dispersion: Var = mu + mu^2/phi
}

model {
    // Priors
    beta ~ normal(beta_prior_mean, beta_prior_sd);
    lambda_floor ~ lognormal(lambda_prior_mean, lambda_prior_sd);
    phi ~ gamma(2.0, 0.1);

    // NegBin likelihood on binned counts
    for (j in 1:n_bins) {
        real mu_j = compute_bin_expected(beta, lambda_floor,
                                          ml_lo[j], ml_hi[j],
                                          mw_floor, mw_max,
                                          sigma_ml_avg, sigma_round,
                                          n_comp_bins, mw_comp_lo,
                                          mw_comp_hi, t_obs, n_quad);
        mu_j = fmax(mu_j, 1e-10);
        counts[j] ~ neg_binomial_2(mu_j, phi);
    }
}

generated quantities {
    real b = beta / LOG10;
    real frac_above_mw_min = (1 - exp(-beta * (mw_max - mw_min))) /
                              (1 - exp(-beta * (mw_max - mw_floor)));
    real lambda_mw_min = lambda_floor * frac_above_mw_min;
    real mean_overdispersion;
    {
        real weighted_sum = 0.0;
        real weight_sum = 0.0;
        for (j in 1:n_bins) {
            real mu_j = compute_bin_expected(beta, lambda_floor,
                                              ml_lo[j], ml_hi[j],
                                              mw_floor, mw_max,
                                              sigma_ml_avg, sigma_round,
                                              n_comp_bins, mw_comp_lo,
                                              mw_comp_hi, t_obs, n_quad);
            mu_j = fmax(mu_j, 1e-10);
            weighted_sum += mu_j * (1.0 + mu_j / phi);
            weight_sum += mu_j;
        }
        mean_overdispersion = weighted_sum / fmax(weight_sum, 1e-10);
    }
}
