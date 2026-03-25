// =============================================================================
// GOLD STANDARD L5: Marginalised Likelihood for Thinned Poisson Process
// =============================================================================
// Integrates out latent true magnitudes analytically (no latent mw_true).
// Properly weights by T(mw) for variable completeness.
//
// All data-dependent quantities (mw2ml, sigma_conv, Phi, T(mw)) are
// precomputed in transformed data.  The model block reduces to a
// matrix-vector product plus a dot product — only exp()/log() on the
// autodiff tape.
//
// The likelihood is:
//   L = exp(-Λ) × ∏_i λ_floor × I_i
// where:
//   Λ = λ_floor × Σ_q T(mw_q) × f_GR(mw_q) × p_sel(mw_q) × Δmw
//   I_i = Σ_q T(mw_q) × f_GR(mw_q) × p(ml_i|mw_q) × Δmw
//
// Reference: Stafford (2026), ONR T983 Report, Section 6

functions {
    // Retained for generated quantities
    real mw2ml(real mw) {
        real a = 0.0376;
        real b = 0.646;
        real c = 0.53 - mw;
        real disc = fmax(square(b) - 4 * a * c, 0.0);
        return (-b + sqrt(disc)) / (2 * a);
    }
}

data {
    int<lower=1> N;
    vector[N] ml_reported;
    vector<lower=0>[N] sigma_ml;
    real<lower=0> sigma_round;
    real mw_min;
    real mw_max;
    real mw_floor;
    real ml_threshold;
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
    real sigma_ml_avg = mean(sigma_ml);
    real dmw = (mw_max - mw_floor) / n_quad;
    real inv_sqrt2pi = 1.0 / sqrt(2 * pi());

    // Quadrature grid — all quantities independent of parameters
    vector[n_quad] mw_grid;
    vector[n_quad] T_grid;          // observation time at each grid point
    vector[n_quad] T_psel_grid;     // T(mw_q) × p_sel(mw_q) for expected count
    matrix[N, n_quad] obs_kernel;   // T(mw_q) × p(ml_i | mw_q) for per-event intensity

    for (q in 1:n_quad) {
        real mw_q = mw_floor + (q - 0.5) * dmw;
        mw_grid[q] = mw_q;

        // ── mw2ml (Grünthal inverse) ──
        real a = 0.0376;
        real b_coeff = 0.646;
        real c = 0.53 - mw_q;
        real disc = fmax(square(b_coeff) - 4 * a * c, 0.0);
        real ml_q = (-b_coeff + sqrt(disc)) / (2 * a);

        // ── Conversion uncertainty in Mw space ──
        real slope = 2 * a * ml_q + b_coeff;
        real var_conv = (0.97 * pow(ml_q, 4) - 12.4 * pow(ml_q, 3) +
                         58.4 * square(ml_q) - 120.0 * ml_q + 921.0) * 1e-4;
        real s_conv = fmin(fmax(sqrt(fmax(var_conv, 0.0)), 0.29), 0.34) / abs(slope);
        real s_conv_sq = square(s_conv) + square(sigma_round);

        // ── T(mw) from completeness bins ──
        real T_q;
        if (mw_q < mw_comp_lo[1]) {
            T_q = t_obs[1];           // scatter-IN region
        } else {
            T_q = t_obs[n_comp_bins]; // above last bin edge
            for (j in 1:n_comp_bins) {
                if (mw_q >= mw_comp_lo[j] && mw_q < mw_comp_hi[j]) {
                    T_q = t_obs[j];
                }
            }
        }
        T_grid[q] = T_q;

        // ── Selection probability (using sigma_ml_avg) ──
        real st_avg = sqrt(square(sigma_ml_avg) + s_conv_sq);
        real p_sel_q = Phi((ml_q - ml_threshold) / st_avg);
        T_psel_grid[q] = T_q * p_sel_q;

        // ── Per-event observation kernel: T(mw_q) × N(ml_i; ml_q, sigma_total_i) ──
        for (i in 1:N) {
            real st_i = sqrt(square(sigma_ml[i]) + s_conv_sq);
            obs_kernel[i, q] = T_q * inv_sqrt2pi / st_i *
                               exp(-0.5 * square((ml_reported[i] - ml_q) / st_i));
        }
    }

    // Precompute p_sel at mw_min for generated quantities
    real p_sel_at_mw_min_precomp;
    {
        real ml_at_min = mw2ml(mw_min);
        real a2 = 0.0376;
        real b2 = 0.646;
        real slope2 = 2 * a2 * ml_at_min + b2;
        real var_conv2 = (0.97 * pow(ml_at_min, 4) - 12.4 * pow(ml_at_min, 3) +
                          58.4 * square(ml_at_min) - 120.0 * ml_at_min + 921.0) * 1e-4;
        real sc2 = fmin(fmax(sqrt(fmax(var_conv2, 0.0)), 0.29), 0.34) / abs(slope2);
        real st2 = sqrt(square(sigma_ml_avg) + square(sc2) + square(sigma_round));
        p_sel_at_mw_min_precomp = Phi((ml_at_min - ml_threshold) / st2);
    }
}

parameters {
    real<lower=0.5, upper=5.0> beta;
    real<lower=0.001, upper=1000.0> lambda_floor;
}

model {
    // Priors
    beta ~ normal(beta_prior_mean, beta_prior_sd);
    lambda_floor ~ lognormal(lambda_prior_mean, lambda_prior_sd);

    // GR weights at quadrature points (only parameter-dependent part)
    real gr_norm = 1.0 - exp(-beta * (mw_max - mw_floor));
    vector[n_quad] gr_wt;
    for (q in 1:n_quad) {
        gr_wt[q] = beta / gr_norm * exp(-beta * (mw_grid[q] - mw_floor)) * dmw;
    }

    // Expected count: Λ = λ_floor × Σ_q T(mw_q) × p_sel(mw_q) × f_GR(mw_q) × Δmw
    real expected_count = lambda_floor * dot_product(T_psel_grid, gr_wt);

    // Per-event marginalised log-intensity via matrix-vector product
    //   I_i = Σ_q obs_kernel[i,q] × gr_wt[q]
    target += sum(log(obs_kernel * gr_wt)) + N * log(lambda_floor) - expected_count;
}

generated quantities {
    real b = beta / LOG10;
    real frac_above_mw_min = (exp(-beta * (mw_min - mw_floor)) -
                               exp(-beta * (mw_max - mw_floor))) /
                              (1 - exp(-beta * (mw_max - mw_floor)));
    real lambda_mw_min = lambda_floor * frac_above_mw_min;
    real p_sel_at_mw_min = p_sel_at_mw_min_precomp;
}
