// L5 Marginalised Model — Constant Completeness
// Simplified version for single observation period with uniform sigma_ml.
// All data-dependent quantities precomputed in transformed data.
//
// Reference: Stafford (2026), ONR T983 Report, Section 6

data {
    int<lower=1> N;
    vector[N] ml_reported;
    real<lower=0> sigma_ml;
    real<lower=0> sigma_round;
    real mw_min;
    real mw_max;
    real mw_floor;
    real ml_threshold;
    real<lower=0> t_obs;
    int<lower=1> n_quad;
    // Configurable priors
    real beta_prior_mean;
    real<lower=0> beta_prior_sd;
    real lambda_prior_mean;
    real<lower=0> lambda_prior_sd;
}

transformed data {
    real LOG10 = log(10.0);
    real dmw = (mw_max - mw_floor) / n_quad;
    real inv_sqrt2pi = 1.0 / sqrt(2 * pi());

    // Quadrature grid — all quantities independent of parameters
    vector[n_quad] mw_grid;
    vector[n_quad] p_sel_grid;
    matrix[N, n_quad] obs_kernel;   // p(ml_i | mw_q)

    for (q in 1:n_quad) {
        real mw_q = mw_floor + (q - 0.5) * dmw;
        mw_grid[q] = mw_q;

        // mw2ml (Grünthal inverse)
        real a = 0.0376;
        real b_coeff = 0.646;
        real c = 0.53 - mw_q;
        real disc = fmax(square(b_coeff) - 4 * a * c, 0.0);
        real ml_q = (-b_coeff + sqrt(disc)) / (2 * a);

        // Conversion uncertainty in Mw space
        real slope = 2 * a * ml_q + b_coeff;
        real var_conv = (0.97 * pow(ml_q, 4) - 12.4 * pow(ml_q, 3) +
                         58.4 * square(ml_q) - 120.0 * ml_q + 921.0) * 1e-4;
        real s_conv = fmin(fmax(sqrt(fmax(var_conv, 0.0)), 0.29), 0.34) / abs(slope);
        real st = sqrt(square(sigma_ml) + square(s_conv) + square(sigma_round));

        // Selection probability
        p_sel_grid[q] = Phi((ml_q - ml_threshold) / st);

        // Per-event observation kernel
        for (i in 1:N) {
            obs_kernel[i, q] = inv_sqrt2pi / st *
                               exp(-0.5 * square((ml_reported[i] - ml_q) / st));
        }
    }
}

parameters {
    real<lower=0.5, upper=5.0> beta;
    real<lower=0.001, upper=1000.0> lambda_floor;
}

model {
    beta ~ normal(beta_prior_mean, beta_prior_sd);
    lambda_floor ~ lognormal(lambda_prior_mean, lambda_prior_sd);

    real gr_norm = 1.0 - exp(-beta * (mw_max - mw_floor));
    vector[n_quad] gr_wt;
    for (q in 1:n_quad) {
        gr_wt[q] = beta / gr_norm * exp(-beta * (mw_grid[q] - mw_floor)) * dmw;
    }

    // Expected count
    real expected_count = lambda_floor * t_obs * dot_product(p_sel_grid, gr_wt);

    // Per-event log-intensity via matrix-vector product
    target += sum(log(obs_kernel * gr_wt)) + N * log(lambda_floor) - expected_count;
}

generated quantities {
    real b = beta / LOG10;
    real frac_above_mw_min = (exp(-beta * (mw_min - mw_floor)) -
                               exp(-beta * (mw_max - mw_floor))) /
                              (1 - exp(-beta * (mw_max - mw_floor)));
    real lambda_mw_min = lambda_floor * frac_above_mw_min;
}
