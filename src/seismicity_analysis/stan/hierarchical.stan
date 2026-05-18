// Hierarchical Multi-Zone GR Model
// Non-centred parameterisation for zone-specific beta values.
// Population hyperpriors on mu_beta and sigma_beta.
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
}

data {
    int<lower=1> n_zones;
    int<lower=1> n_events_total;
    array[n_zones] int<lower=0> n_events;      // Events per zone
    array[n_zones] int<lower=1> zone_start;     // Start index per zone
    vector[n_events_total] ml_reported;
    vector<lower=0>[n_events_total] sigma_ml;
    real<lower=0> sigma_round;
    real mw_min;
    real mw_max;
    real mw_floor;
    array[n_zones] real<lower=0> t_obs;
    // Hyperprior settings
    real mu_b_prior_mean;
    real<lower=0> mu_b_prior_sd;
    real<lower=0> sigma_b_prior_scale;
}

transformed data {
    real LOG10 = log(10.0);
    real ml_lower = mw2ml(mw_floor) - 2.0;
    real ml_upper = mw2ml(mw_max) + 3.0;
}

parameters {
    real mu_beta;
    real<lower=0.01, upper=0.5> sigma_beta;
    vector[n_zones] beta_raw;                    // Non-centred
    vector<lower=0>[n_zones] lambda_floor;
    vector<lower=ml_lower, upper=ml_upper>[n_events_total] ml_true;
}

transformed parameters {
    vector[n_zones] beta;
    for (z in 1:n_zones) {
        beta[z] = mu_beta + sigma_beta * beta_raw[z];
    }
}

model {
    // Hyperpriors
    mu_beta ~ normal(mu_b_prior_mean * LOG10, mu_b_prior_sd * LOG10);
    sigma_beta ~ normal(0, sigma_b_prior_scale * LOG10);
    beta_raw ~ std_normal();
    lambda_floor ~ gamma(2.0, 0.5);

    // Per-zone likelihood
    for (z in 1:n_zones) {
        int idx_start = zone_start[z];
        int n_z = n_events[z];

        // GR prior on latent Mw
        for (i in 1:n_z) {
            int idx = idx_start + i - 1;
            real mw_i = ml2mw(ml_true[idx]);
            target += log(beta[z]) - beta[z] * (mw_i - mw_floor)
                      - log(1.0 - exp(-beta[z] * (mw_max - mw_floor)));
        }

        // Observation model
        for (i in 1:n_z) {
            int idx = idx_start + i - 1;
            real sigma_total = compute_sigma_total(ml_true[idx],
                                                    sigma_ml[idx], sigma_round);
            ml_reported[idx] ~ normal(ml_true[idx], sigma_total);
        }

        // Poisson rate
        target += n_z * log(lambda_floor[z]) - lambda_floor[z] * t_obs[z];
    }
}

generated quantities {
    real mu_b = mu_beta / LOG10;
    real sigma_b = sigma_beta / LOG10;
    vector[n_zones] b;
    vector[n_zones] lambda_mw_min;
    vector[n_zones] shrinkage;

    for (z in 1:n_zones) {
        b[z] = beta[z] / LOG10;
        real frac = (1 - exp(-beta[z] * (mw_max - mw_min))) /
                    (1 - exp(-beta[z] * (mw_max - mw_floor)));
        lambda_mw_min[z] = lambda_floor[z] * frac;

        // Shrinkage: ratio of population variance to total
        real n_eff = n_events[z] * 1.0;
        shrinkage[z] = square(sigma_beta) /
                       (square(sigma_beta) + 1.0 / fmax(n_eff, 1.0));
    }
}
