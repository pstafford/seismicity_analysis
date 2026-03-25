"""
Logic tree discretization for probabilistic seismic hazard analysis.

Implements several schemes for discretizing continuous posterior distributions
into weighted logic tree branches, preserving correlation structure.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class DiscretizationScheme:
    """
    Named discretization scheme with z-values and weights.
    The 3-point schemes place nodes at (μ - z σ, μ, μ + z σ) with weights (w₁, w₂, w₃).
    """
    name: str
    z: float
    weights: list


# Miller-Rice (1983): z = √3, w = 1/6. Optimal for matching mean and variance.
MILLER_RICE = DiscretizationScheme("Miller-Rice", np.sqrt(3), [1/6, 2/3, 1/6])

# Heavy-Tail: z = 1.034, w = 0.468. Better captures regulatory fractiles (P84).
HEAVY_TAIL = DiscretizationScheme("Heavy-Tail", 1.034, [0.468, 0.064, 0.468])

# ESM (Equal-Spaced Model): z = 1.282 (P10/P90), w = 0.3.
ESM = DiscretizationScheme("ESM", 1.282, [0.3, 0.4, 0.3])

# EPT (Extended Percentile Triplet): z = 1.645 (P5/P95), w = 0.185.
EPT = DiscretizationScheme("EPT", 1.645, [0.185, 0.630, 0.185])

# Equal weight: z = √(3/2), w = 1/3.
EQUAL_WEIGHT = DiscretizationScheme("Equal", np.sqrt(1.5), [1/3, 1/3, 1/3])


def discretize_marginal(mu, sigma, scheme=None):
    """
    Discretize a univariate Gaussian into 3 weighted nodes.

    Returns dict with 'nodes' and 'weights' where nodes = [μ - zσ, μ, μ + zσ].
    """
    if scheme is None:
        scheme = HEAVY_TAIL
    nodes = np.array([mu - scheme.z * sigma, mu, mu + scheme.z * sigma])
    return {"nodes": nodes, "weights": np.array(scheme.weights)}


def discretize_joint(mu, Sigma, scheme=None):
    """
    Discretize a bivariate Gaussian preserving correlation via conditional distributions.

    For parameters (x₁, x₂) with correlation ρ:
    - Marginal nodes for x₁: μ₁ ± z σ₁
    - Conditional nodes for x₂|x₁: μ₂|₁ ± z σ₂|₁

    Returns dict with 'nodes_x1', 'nodes_x2', 'weights' as arrays of length 9.
    """
    if scheme is None:
        scheme = HEAVY_TAIL

    mu = np.asarray(mu)
    Sigma = np.asarray(Sigma)
    if len(mu) != 2:
        raise ValueError("Only bivariate discretization supported")

    mu1, mu2 = mu
    sigma1 = np.sqrt(Sigma[0, 0])
    sigma2 = np.sqrt(Sigma[1, 1])
    rho = Sigma[0, 1] / (sigma1 * sigma2)

    sigma2_cond = sigma2 * np.sqrt(max(1 - rho**2, 0.0))

    z = scheme.z
    w = scheme.weights
    z_nodes = [-z, 0.0, z]

    nodes_x1 = []
    nodes_x2 = []
    weights = []

    for i, z1 in enumerate(z_nodes):
        x1 = mu1 + sigma1 * z1
        mu2_cond = mu2 + rho * (sigma2 / sigma1) * (x1 - mu1)

        for j, z2 in enumerate(z_nodes):
            x2 = mu2_cond + sigma2_cond * z2
            nodes_x1.append(x1)
            nodes_x2.append(x2)
            weights.append(w[i] * w[j])

    return {
        "nodes_x1": np.array(nodes_x1),
        "nodes_x2": np.array(nodes_x2),
        "weights": np.array(weights),
    }


def build_logic_tree(result, scheme=None):
    """
    Build a logic tree from a GRResult, preserving the (λ, b) correlation.

    Returns a list of dicts with keys 'lambda_n', 'b', 'weight'.
    """
    if scheme is None:
        scheme = HEAVY_TAIL

    mu = np.array([np.log(result.lambda_mean), result.b_mean])
    sigma_lnlam = result.lambda_sd / result.lambda_mean  # delta method
    sigma_b = result.b_sd
    rho = result.rho_lb

    Sigma = np.array([
        [sigma_lnlam**2, rho * sigma_lnlam * sigma_b],
        [rho * sigma_lnlam * sigma_b, sigma_b**2],
    ])

    d = discretize_joint(mu, Sigma, scheme=scheme)

    branches = []
    for i in range(len(d["weights"])):
        branches.append({
            "lambda_n": np.exp(d["nodes_x1"][i]),
            "b": d["nodes_x2"][i],
            "weight": d["weights"][i],
        })

    return branches


def discretize_five_point(mu, sigma):
    """
    5-point discretization matching the first 4 moments of a Gaussian.
    Nodes at {μ ± 2.86σ, μ ± 1.36σ, μ} with weights {0.011, 0.222, 0.534, 0.222, 0.011}.
    """
    z = [2.856, 1.356]
    w = [0.0113, 0.2221, 0.5332, 0.2221, 0.0113]
    nodes = np.array([mu - z[0]*sigma, mu - z[1]*sigma, mu,
                      mu + z[1]*sigma, mu + z[0]*sigma])
    return {"nodes": nodes, "weights": np.array(w)}
