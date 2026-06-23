#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AMM Liquidity Provider Risk Simulator

Simulates LP economics in a constant-product AMM (Uniswap V2 style).
Compares LP performance vs HODL across different market conditions.

@author: viktorvantsev
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({"font.size": 12})


# 1. STATIC AMM MECHANICS

def pool_reserves(k, price):
    """Reserves of constant-product pool at given external price.
    Solves x*y=k and y/x=P, giving x=sqrt(k/P), y=sqrt(k*P).
    """
    x = np.sqrt(k / price)
    y = np.sqrt(k * price)
    return x, y


def lp_value(k, price):
    """Value of full LP position at given price (before fees)."""
    x, y = pool_reserves(k, price)
    return x * price + y


def hodl_value(x0, y0, price):
    """Value of passive HODL position with initial quantities (x0, y0)."""
    return x0 * price + y0


def impermanent_loss(r):
    """IL relative to HODL as function of price ratio r = P_T/P_0.
    Formula: 2*sqrt(r)/(1+r) - 1. Symmetric: IL(r) = IL(1/r).
    """
    return 2 * np.sqrt(r) / (1 + r) - 1


# Sanity checks
def run_sanity_checks(P0=2000, x0=1.0, y0=2000.0):
    k = x0 * y0
    x, y = pool_reserves(k, P0)
    assert np.isclose(x, x0) and np.isclose(y, y0), "Initial reserves mismatch"
    assert np.isclose(lp_value(k, P0), hodl_value(x0, y0, P0)), \
        "LP and HODL should match at initial price"
    assert np.isclose(impermanent_loss(0.5), impermanent_loss(2.0)), \
        "IL symmetry failed"
    print("Sanity checks passed.")


# 2. PRICE PROCESSES

def simulate_gbm_path(P0, mu, sigma, T, n_steps, seed=None):
    """Geometric Brownian Motion price path.
    dP/P = mu*dt + sigma*dW. Always positive.
    """
    if seed is not None:
        np.random.seed(seed)
    dt = T / n_steps
    Z = np.random.normal(0, 1, size=n_steps)
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_prices = np.log(P0) + np.cumsum(log_returns)
    return np.concatenate([[P0], np.exp(log_prices)])


def simulate_log_ou_path(P0, mu_long, kappa, sigma, T, n_steps, seed=None):
    """Mean-reverting price path via log-OU (Schwartz model).
    d(log P) = kappa*(log(mu_long) - log P)*dt + sigma*dW.
    Always positive by construction (works in log space).
    """
    if seed is not None:
        np.random.seed(seed)
    dt = T / n_steps
    Z = np.random.normal(0, 1, size=n_steps)
    log_mu = np.log(mu_long)
    log_prices = np.zeros(n_steps + 1)
    log_prices[0] = np.log(P0)
    for t in range(n_steps):
        log_prices[t+1] = (log_prices[t]
                           + kappa * (log_mu - log_prices[t]) * dt
                           + sigma * np.sqrt(dt) * Z[t])
    return np.exp(log_prices)


def generate_price_path(P0, sigma, T, n_steps, model="gbm",
                        mu=0.0, mu_long=None, kappa=4.0, seed=None):
    """Wrapper that picks the right process."""
    if model == "gbm":
        return simulate_gbm_path(P0, mu, sigma, T, n_steps, seed)
    elif model == "ou":
        if mu_long is None:
            mu_long = P0
        return simulate_log_ou_path(P0, mu_long, kappa, sigma, T, n_steps, seed)
    else:
        raise ValueError(f"Unknown model: {model}")



# 3. LP PATH SIMULATION

def simulate_lp_path(prices, x0, y0, fee_rate, T, noise_volume_daily=0.0):
    """Walk along a price path, accounting for arbitrage, fees, and LVR.

    At each step:
    - Arbitrage moves pool to new equilibrium reserves
    - Fees accumulate based on traded volume (arbitrage + noise)
    - LVR tracks value leaked to arbitrageurs (vs fair-price rebalancing)

    Note: fees are added as external income, not reinvested into k.
    This is a simplification — in real Uniswap V2 fees stay in the pool.
    For our purpose (comparing fee income vs IL), it's adequate.

    noise_volume_daily : fraction of TVL traded daily by non-arbitrage flow.
                         E.g. 0.3 means 30% TVL/day exogenous volume.

    Returns dict with all trajectories and totals.
    """
    n_steps = len(prices) - 1
    days_per_step = 365.0 * T / n_steps
    k = x0 * y0

    # Trajectories
    pool_x = np.zeros(n_steps + 1)
    pool_y = np.zeros(n_steps + 1)
    hodl = np.zeros(n_steps + 1)
    lp_no_fees = np.zeros(n_steps + 1)
    lp_with_fees = np.zeros(n_steps + 1)
    cum_fees_arb = np.zeros(n_steps + 1)
    cum_fees_noise = np.zeros(n_steps + 1)
    cum_lvr = np.zeros(n_steps + 1)

    # State
    x, y = x0, y0
    pool_x[0], pool_y[0] = x, y
    hodl[0] = hodl_value(x0, y0, prices[0])
    lp_no_fees[0] = x * prices[0] + y
    lp_with_fees[0] = lp_no_fees[0]

    for t in range(1, n_steps + 1):
        P_new = prices[t]
        x_new, y_new = pool_reserves(k, P_new)

        # Arbitrage trade volume (in quote currency = USDC)
        delta_x = x - x_new       # ETH leaving the pool (if positive)
        delta_y = y_new - y       # USDC entering the pool (if positive)
        arb_volume = abs(delta_y)
        arb_fee = fee_rate * arb_volume

        # Noise volume: independent of price moves, doesn't change reserves
        tvl = 2 * y_new
        noise_volume = noise_volume_daily * tvl * days_per_step
        noise_fee = fee_rate * noise_volume

        # LVR step: value arbitrageur extracted vs fair price rebalancing
        lvr_step = delta_x * P_new - delta_y

        cum_fees_arb[t] = cum_fees_arb[t-1] + arb_fee
        cum_fees_noise[t] = cum_fees_noise[t-1] + noise_fee
        cum_lvr[t] = cum_lvr[t-1] + lvr_step

        # Update state
        x, y = x_new, y_new
        pool_x[t], pool_y[t] = x, y
        hodl[t] = hodl_value(x0, y0, P_new)
        lp_no_fees[t] = x * P_new + y
        lp_with_fees[t] = lp_no_fees[t] + cum_fees_arb[t] + cum_fees_noise[t]

    cum_fees_total = cum_fees_arb + cum_fees_noise

    return {
        "prices": prices,
        "pool_x": pool_x,
        "pool_y": pool_y,
        "hodl": hodl,
        "lp_no_fees": lp_no_fees,
        "lp_with_fees": lp_with_fees,
        "cum_fees_arb": cum_fees_arb,
        "cum_fees_noise": cum_fees_noise,
        "cum_fees": cum_fees_total,
        "cum_lvr": cum_lvr,
        "fees_total": cum_fees_total[-1],
        "lvr_total": cum_lvr[-1],
        "V_hodl_final": hodl[-1],
        "V_lp_final": lp_with_fees[-1],
        "r_final": prices[-1] / prices[0],
        "lp_vs_hodl_pct": (lp_with_fees[-1] / hodl[-1] - 1) * 100,
    }



# 4. MONTE CARLO AND REGIME MAPS

def generate_paths_batch(P0, sigma, T, n_steps, n_sim, model="gbm",
                        mu=0.0, mu_long=None, kappa=4.0, base_seed=0):
    """Generate many price paths at once, vectorized.
    Much faster than calling the single-path function in a loop.
    """
    rng = np.random.default_rng(base_seed)
    dt = T / n_steps
    Z = rng.standard_normal((n_sim, n_steps))
    prices = np.zeros((n_sim, n_steps + 1))
    prices[:, 0] = P0

    if model == "gbm":
        log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        prices[:, 1:] = P0 * np.exp(np.cumsum(log_returns, axis=1))
    elif model == "ou":
        if mu_long is None:
            mu_long = P0
        log_prices = np.zeros_like(prices)
        log_prices[:, 0] = np.log(P0)
        log_mu = np.log(mu_long)
        for t in range(n_steps):
            log_prices[:, t+1] = (log_prices[:, t]
                                  + kappa * (log_mu - log_prices[:, t]) * dt
                                  + sigma * np.sqrt(dt) * Z[:, t])
        prices = np.exp(log_prices)
    else:
        raise ValueError(f"Unknown model: {model}")

    return prices


def run_monte_carlo(P0, x0, y0, sigma, T, n_steps, fee_rate, n_sim,
                    model="gbm", mu=0.0, mu_long=None, kappa=4.0,
                    noise_volume_daily=0.0, base_seed=0):
    """Vectorized Monte Carlo across many paths.

    Loops only over time steps, not over simulations. Returns dict of arrays.
    """
    prices = generate_paths_batch(P0, sigma, T, n_steps, n_sim, model,
                                   mu, mu_long, kappa, base_seed)
    k = x0 * y0
    days_per_step = 365.0 * T / n_steps

    # State per simulation
    x = np.full(n_sim, x0)
    y = np.full(n_sim, y0)
    fees_total = np.zeros(n_sim)
    lvr_total = np.zeros(n_sim)

    for t in range(1, n_steps + 1):
        P_new = prices[:, t]
        x_new = np.sqrt(k / P_new)
        y_new = np.sqrt(k * P_new)

        delta_x = x - x_new
        delta_y = y_new - y
        arb_volume = np.abs(delta_y)
        arb_fee = fee_rate * arb_volume

        tvl = 2 * y_new
        noise_fee = fee_rate * noise_volume_daily * tvl * days_per_step

        fees_total += arb_fee + noise_fee
        lvr_total += delta_x * P_new - delta_y

        x, y = x_new, y_new

    P_final = prices[:, -1]
    V_hodl = x0 * P_final + y0
    V_lp_no_fees = x * P_final + y
    V_lp = V_lp_no_fees + fees_total
    diff_pct = (V_lp / V_hodl - 1) * 100

    return {
        "diff_pct": diff_pct,
        "V_hodl": V_hodl,
        "V_lp": V_lp,
        "fees_total": fees_total,
        "lvr_total": lvr_total,
        "r_final": P_final / P0,
        "mean": np.mean(diff_pct),
        "median": np.median(diff_pct),
        "winrate": np.mean(diff_pct > 0) * 100,
    }


def compute_regime_map(P0, x0, y0, T, n_steps, sigma_grid, fee_grid, n_sim,
                       model="gbm", noise_volume_daily=0.0, **kwargs):
    """Build a regime map: winrate and mean diff over (sigma, fee) grid.
    Different seed per cell to avoid correlated samples.
    """
    winrate = np.zeros((len(fee_grid), len(sigma_grid)))
    mean_diff = np.zeros((len(fee_grid), len(sigma_grid)))

    for i, fee in enumerate(fee_grid):
        for j, sigma in enumerate(sigma_grid):
            seed = 10_000 * i + 1_000 * j
            result = run_monte_carlo(
                P0, x0, y0, sigma, T, n_steps, fee, n_sim,
                model=model, noise_volume_daily=noise_volume_daily,
                base_seed=seed, **kwargs
            )
            winrate[i, j] = result["winrate"]
            mean_diff[i, j] = result["mean"]

    return {"sigma_grid": sigma_grid, "fee_grid": fee_grid,
            "winrate": winrate, "mean_diff": mean_diff}



# 5. PLOTTING

def save_or_show(fig, output_dir, filename, show):
    """Helper: save figure to disk and/or show it."""
    fig.tight_layout()
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / filename, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_il_curve(output_dir=None, show=True):
    """Plot universal IL curve."""
    r = np.linspace(0.1, 5.0, 500)
    il = impermanent_loss(r) * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(r, il, linewidth=2.5, color="crimson")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.axvline(1, color="black", linewidth=0.7, linestyle="--")

    for ratio in [0.5, 2.0, 3.0, 4.0]:
        y_val = impermanent_loss(ratio) * 100
        ax.plot(ratio, y_val, "o", color="black", markersize=5)
        ax.annotate(f"{y_val:.1f}%", xy=(ratio, y_val),
                    xytext=(8, -14), textcoords="offset points")

    ax.set_xlabel("Price ratio $r = P_T / P_0$")
    ax.set_ylabel("Impermanent loss vs HODL (%)")
    ax.set_title("Impermanent Loss in a Constant-Product AMM")
    ax.grid(alpha=0.3)
    save_or_show(fig, output_dir, "01_il_curve.png", show)


def plot_sample_paths(P0, sigma, T, n_steps, model="gbm", n_paths=5,
                     output_dir=None, show=True, **kwargs):
    """Plot a few sample price paths."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for seed in range(n_paths):
        path = generate_price_path(P0, sigma, T, n_steps, model=model,
                                    seed=seed, **kwargs)
        ax.plot(path, alpha=0.75, linewidth=1.2)

    ax.axhline(P0, color="black", linestyle="--", linewidth=0.8,
               label="Initial price")
    title = "GBM" if model == "gbm" else "Mean-reverting log-OU"
    ax.set_title(f"{title} price paths ({n_paths} simulations)")
    ax.set_xlabel("Step (day)")
    ax.set_ylabel("Price (USDC)")
    ax.legend()
    ax.grid(alpha=0.3)
    save_or_show(fig, output_dir, f"02_{model}_paths.png", show)


def plot_single_path_lp_vs_hodl(result, output_dir=None, show=True):
    """Two panels: price and the three strategies."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    axes[0].plot(result["prices"], color="gray", linewidth=1.3, label="ETH price")
    axes[0].axhline(result["prices"][0], color="black", linestyle="--",
                    linewidth=0.7, label="Initial price")
    axes[0].set_ylabel("ETH price (USDC)")
    axes[0].set_title("One simulated price path")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(result["hodl"], color="steelblue", linewidth=2, label="HODL")
    axes[1].plot(result["lp_no_fees"], color="crimson", linewidth=2,
                 label="LP (no fees)")
    axes[1].plot(result["lp_with_fees"], color="darkgreen", linewidth=2,
                 label="LP (with fees)")
    axes[1].axhline(result["hodl"][0], color="black", linestyle="--",
                    linewidth=0.7, label="Initial value")
    axes[1].set_xlabel("Step (day)")
    axes[1].set_ylabel("Portfolio value (USDC)")
    axes[1].set_title("LP vs HODL on one path")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    save_or_show(fig, output_dir, "03_single_path.png", show)


def plot_fee_decomposition(result, output_dir=None, show=True):
    """Show IL and fees as differences from HODL."""
    gap_no_fees = result["lp_no_fees"] - result["hodl"]
    gap_with_fees = result["lp_with_fees"] - result["hodl"]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(gap_no_fees, color="crimson", linewidth=2,
            label="LP (no fees) − HODL = IL in USDC")
    ax.plot(gap_with_fees, color="darkgreen", linewidth=2,
            label="LP (with fees) − HODL")
    ax.fill_between(np.arange(len(gap_no_fees)), gap_no_fees, gap_with_fees,
                     color="lightgreen", alpha=0.35, label="Accumulated fees")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Step (day)")
    ax.set_ylabel("Difference vs HODL (USDC)")
    ax.set_title("LP Decomposition: IL vs Fees")
    ax.legend()
    ax.grid(alpha=0.3)
    save_or_show(fig, output_dir, "04_fee_decomposition.png", show)


def plot_distribution(result, title, filename, output_dir=None, show=True):
    """Histogram of LP vs HODL outcomes."""
    diff = result["diff_pct"]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.hist(diff, bins=50, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.1, label="LP = HODL")
    ax.axvline(result["mean"], color="crimson", linewidth=2,
               label=f"Mean = {result['mean']:+.2f}%")
    ax.axvline(result["median"], color="darkgreen", linewidth=2,
               label=f"Median = {result['median']:+.2f}%")
    ax.axvspan(diff.min() - 1, 0, alpha=0.08, color="red")
    ax.axvspan(0, diff.max() + 1, alpha=0.08, color="green")
    ax.set_xlabel("(LP with fees − HODL) / HODL (%)")
    ax.set_ylabel("Number of simulations")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    save_or_show(fig, output_dir, filename, show)


def plot_volatility_sweep(P0, x0, y0, T, n_steps, fee_rate, sigma_values, n_sim,
                          output_dir=None, show=True):
    """Three panels: mean diff, winrate, mean fees, all vs sigma."""
    mean_diffs, winrates, mean_fees = [], [], []
    for i, sigma in enumerate(sigma_values):
        result = run_monte_carlo(P0, x0, y0, sigma, T, n_steps, fee_rate, n_sim,
                                 base_seed=100_000 + 1_000 * i)
        mean_diffs.append(result["mean"])
        winrates.append(result["winrate"])
        mean_fees.append(np.mean(result["fees_total"]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
    axes[0].plot(sigma_values, mean_diffs, marker="o", color="crimson", linewidth=2)
    axes[0].axhline(0, color="black", linestyle="--", linewidth=0.7)
    axes[0].set_title("Mean LP − HODL")
    axes[0].set_xlabel("Annual volatility σ")
    axes[0].set_ylabel("Mean LP − HODL (%)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(sigma_values, winrates, marker="o", color="steelblue", linewidth=2)
    axes[1].axhline(50, color="black", linestyle="--", linewidth=0.7, label="50%")
    axes[1].set_title("LP winrate")
    axes[1].set_xlabel("Annual volatility σ")
    axes[1].set_ylabel("P(LP > HODL) (%)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(sigma_values, mean_fees, marker="o", color="darkgreen", linewidth=2)
    axes[2].set_title("Mean fee income")
    axes[2].set_xlabel("Annual volatility σ")
    axes[2].set_ylabel("Mean fees (USDC)")
    axes[2].grid(alpha=0.3)

    fig.suptitle(f"Volatility sensitivity (fee={fee_rate:.2%}, T={T}y, n={n_sim})",
                 y=1.02)
    save_or_show(fig, output_dir, "06_volatility_sweep.png", show)


def plot_regime_map(regime, title, filename, output_dir=None, show=True):
    """Two panels: winrate heatmap and mean-diff heatmap."""
    sigma_grid = regime["sigma_grid"]
    fee_grid = regime["fee_grid"]
    extent = [sigma_grid[0], sigma_grid[-1], fee_grid[0]*100, fee_grid[-1]*100]
    vmax = max(abs(regime["mean_diff"].min()),
               abs(regime["mean_diff"].max()), 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    im1 = axes[0].imshow(regime["winrate"], aspect="auto", origin="lower",
                          cmap="RdYlGn", vmin=0, vmax=100, extent=extent)
    axes[0].set_title("LP winrate")
    axes[0].set_xlabel("Annual volatility σ")
    axes[0].set_ylabel("Fee rate (%)")
    fig.colorbar(im1, ax=axes[0], label="P(LP > HODL) (%)")

    im2 = axes[1].imshow(regime["mean_diff"], aspect="auto", origin="lower",
                          cmap="RdYlGn", vmin=-vmax, vmax=vmax, extent=extent)
    axes[1].set_title("Mean LP − HODL")
    axes[1].set_xlabel("Annual volatility σ")
    axes[1].set_ylabel("Fee rate (%)")
    fig.colorbar(im2, ax=axes[1], label="Mean diff (%)")

    fig.suptitle(title, y=1.02)
    save_or_show(fig, output_dir, filename, show)


def plot_two_regime_maps(map_a, map_b, label_a, label_b, output_dir=None,
                         show=True, filename="08_regime_comparison.png"):
    """Compare two regime maps side by side, plus difference."""
    sigma_grid = map_a["sigma_grid"]
    fee_grid = map_a["fee_grid"]
    extent = [sigma_grid[0], sigma_grid[-1], fee_grid[0]*100, fee_grid[-1]*100]
    gain = map_b["winrate"] - map_a["winrate"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im1 = axes[0].imshow(map_a["winrate"], aspect="auto", origin="lower",
                          cmap="RdYlGn", vmin=0, vmax=100, extent=extent)
    axes[0].set_title(label_a)
    axes[0].set_xlabel("Annual volatility σ")
    axes[0].set_ylabel("Fee rate (%)")
    fig.colorbar(im1, ax=axes[0], label="Winrate (%)")

    im2 = axes[1].imshow(map_b["winrate"], aspect="auto", origin="lower",
                          cmap="RdYlGn", vmin=0, vmax=100, extent=extent)
    axes[1].set_title(label_b)
    axes[1].set_xlabel("Annual volatility σ")
    axes[1].set_ylabel("Fee rate (%)")
    fig.colorbar(im2, ax=axes[1], label="Winrate (%)")

    im3 = axes[2].imshow(gain, aspect="auto", origin="lower",
                          cmap="Greens", vmin=0, vmax=max(gain.max(), 1.0),
                          extent=extent)
    axes[2].set_title("Difference (p.p.)")
    axes[2].set_xlabel("Annual volatility σ")
    axes[2].set_ylabel("Fee rate (%)")
    fig.colorbar(im3, ax=axes[2], label="Δ Winrate (p.p.)")

    save_or_show(fig, output_dir, filename, show)


def plot_two_distributions(result_a, result_b, label_a, label_b,
                            title, filename, output_dir=None, show=True):
    """Compare two LP outcome distributions on one histogram."""
    diff_a = result_a["diff_pct"]
    diff_b = result_b["diff_pct"]
    bins = np.linspace(min(diff_a.min(), diff_b.min()) - 1,
                       max(diff_a.max(), diff_b.max()) + 1, 60)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.hist(diff_a, bins=bins, alpha=0.6, color="steelblue", edgecolor="white",
            label=f"{label_a} (mean={result_a['mean']:+.2f}%, "
                  f"winrate={result_a['winrate']:.0f}%)")
    ax.hist(diff_b, bins=bins, alpha=0.6, color="darkgreen", edgecolor="white",
            label=f"{label_b} (mean={result_b['mean']:+.2f}%, "
                  f"winrate={result_b['winrate']:.0f}%)")
    ax.axvline(0, color="black", linestyle="--", linewidth=1.1, label="LP = HODL")
    ax.set_xlabel("(LP with fees − HODL) / HODL (%)")
    ax.set_ylabel("Number of simulations")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    save_or_show(fig, output_dir, filename, show)


def plot_lvr_path(result, output_dir=None, show=True):
    """Two panels: price and cumulative LVR vs fees."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    axes[0].plot(result["prices"], color="gray", linewidth=1.3)
    axes[0].axhline(result["prices"][0], color="black", linestyle="--",
                    linewidth=0.7)
    axes[0].set_ylabel("ETH price (USDC)")
    axes[0].set_title("One price path")
    axes[0].grid(alpha=0.3)

    axes[1].plot(result["cum_lvr"], color="crimson", linewidth=2,
                 label=f"Cumulative LVR = {result['lvr_total']:.1f} USDC")
    axes[1].plot(result["cum_fees"], color="darkgreen", linewidth=2,
                 label=f"Cumulative fees = {result['fees_total']:.1f} USDC")
    axes[1].fill_between(np.arange(len(result["cum_lvr"])),
                          result["cum_lvr"], result["cum_fees"],
                          where=result["cum_fees"] >= result["cum_lvr"],
                          color="lightgreen", alpha=0.35, label="Fees > LVR")
    axes[1].fill_between(np.arange(len(result["cum_lvr"])),
                          result["cum_lvr"], result["cum_fees"],
                          where=result["cum_fees"] < result["cum_lvr"],
                          color="lightcoral", alpha=0.35, label="LVR > Fees")
    axes[1].set_xlabel("Step (day)")
    axes[1].set_ylabel("Cumulative value (USDC)")
    axes[1].set_title("LVR vs Fees on one path")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    save_or_show(fig, output_dir, "12_lvr_path.png", show)


def plot_lvr_vs_sigma(P0, x0, y0, T, n_steps, fee_rate, sigma_values, n_sim,
                      output_dir=None, show=True):
    """LVR and fees as functions of volatility."""
    mean_lvr, mean_fees = [], []
    for i, sigma in enumerate(sigma_values):
        result = run_monte_carlo(P0, x0, y0, sigma, T, n_steps, fee_rate, n_sim,
                                 base_seed=900_000 + 1_000 * i)
        mean_lvr.append(np.mean(result["lvr_total"]))
        mean_fees.append(np.mean(result["fees_total"]))
    mean_lvr = np.array(mean_lvr)
    mean_fees = np.array(mean_fees)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(sigma_values, mean_lvr, marker="o", color="crimson", linewidth=2,
            label="Mean LVR (leaked to arbitrageurs)")
    ax.plot(sigma_values, mean_fees, marker="s", color="darkgreen", linewidth=2,
            label="Mean fees (LP income)")
    ax.fill_between(sigma_values, mean_lvr, mean_fees,
                     where=mean_fees >= mean_lvr, color="lightgreen", alpha=0.35)
    ax.fill_between(sigma_values, mean_lvr, mean_fees,
                     where=mean_fees < mean_lvr, color="lightcoral", alpha=0.35)
    ax.set_xlabel("Annual volatility σ")
    ax.set_ylabel("USDC over the horizon")
    ax.set_title(f"LVR vs Fees (GBM, fee={fee_rate:.2%}, T={T}y)")
    ax.legend()
    ax.grid(alpha=0.3)
    save_or_show(fig, output_dir, "13_lvr_vs_sigma.png", show)



# 6. MAIN WORKFLOW

def print_mc_summary(result, label):
    """Pretty-print Monte Carlo summary."""
    diff = result["diff_pct"]
    print(f"\n{label}")
    print("-" * len(label))
    print(f"Mean LP − HODL:   {result['mean']:+.2f}%")
    print(f"Median LP − HODL: {result['median']:+.2f}%")
    print(f"LP winrate:       {result['winrate']:.1f}%")
    print(f"5th percentile:   {np.percentile(diff, 5):+.2f}%")
    print(f"95th percentile:  {np.percentile(diff, 95):+.2f}%")
    print(f"Mean fees:        {np.mean(result['fees_total']):.2f} USDC")
    print(f"Mean LVR:         {np.mean(result['lvr_total']):.2f} USDC")


def main(mode="quick", output_dir="figures", show=False):
    """Run the whole project end-to-end."""
    # Baseline parameters: ETH/USDC pool, 1 ETH + 2000 USDC at P0=2000.
    P0 = 2000
    x0, y0 = 1.0, 2000.0
    T = 1.0
    n_steps = 365
    sigma_base = 0.6
    fee_base = 0.003
    mu_long = P0
    kappa = 4.0

    # Monte Carlo/regime grid sizes
    if mode == "quick":
        n_baseline, n_sweep, n_grid = 300, 150, 100
    else:  
        n_baseline, n_sweep, n_grid = 1000, 500, 300

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Running {mode} analysis. Figures → {out.resolve()}")

    run_sanity_checks(P0, x0, y0)

    # Chapter 1: Static AMM mechanics
    plot_il_curve(out, show)

    # Chapter 2: One GBM path, LP accounting
    plot_sample_paths(P0, sigma_base, T, n_steps, model="gbm",
                      output_dir=out, show=show)
    prices = generate_price_path(P0, sigma_base, T, n_steps, model="gbm", seed=42)
    baseline_path = simulate_lp_path(prices, x0, y0, fee_base, T)
    plot_single_path_lp_vs_hodl(baseline_path, out, show)
    plot_fee_decomposition(baseline_path, out, show)

    # Chapter 3: Monte Carlo baseline
    baseline_mc = run_monte_carlo(P0, x0, y0, sigma_base, T, n_steps, fee_base,
                                   n_baseline, model="gbm")
    print_mc_summary(baseline_mc, "Baseline GBM outcome")
    plot_distribution(baseline_mc, "Distribution of LP vs HODL (baseline)",
                     "05_distribution_baseline.png", out, show)

    # Chapter 4: Volatility sweep & regime map
    sigma_values = np.linspace(0.1, 1.5, 15)
    plot_volatility_sweep(P0, x0, y0, T, n_steps, fee_base, sigma_values,
                          n_sweep, out, show)

    sigma_grid = np.linspace(0.1, 1.5, 12)
    fee_grid = np.array([0.0005, 0.001, 0.003, 0.005, 0.01, 0.015, 0.02])
    gbm_regime = compute_regime_map(P0, x0, y0, T, n_steps, sigma_grid,
                                     fee_grid, n_grid, model="gbm")
    plot_regime_map(gbm_regime, "Market regime map (GBM)",
                    "07_regime_gbm.png", out, show)

    # Chapter 5: Noise traders
    noise_regime = compute_regime_map(P0, x0, y0, T, n_steps, sigma_grid,
                                       fee_grid, n_grid, model="gbm",
                                       noise_volume_daily=0.3)
    plot_two_regime_maps(gbm_regime, noise_regime,
                         "GBM, arbitrage only", "GBM + noise (30% TVL/day)",
                         out, show, "08_noise_effect.png")

    # Chapter 6: Mean-reverting OU 
    plot_sample_paths(P0, sigma_base, T, n_steps, model="ou", n_paths=5,
                      output_dir=out, show=show, mu_long=mu_long, kappa=kappa)
    ou_mc = run_monte_carlo(P0, x0, y0, sigma_base, T, n_steps, fee_base,
                             n_baseline, model="ou", mu_long=mu_long, kappa=kappa,
                             base_seed=250_000)
    print_mc_summary(ou_mc, "OU (mean-reverting) outcome")
    plot_two_distributions(baseline_mc, ou_mc, "GBM", "OU",
                           "GBM vs OU: LP outcome distribution",
                           "10_gbm_vs_ou_distribution.png", out, show)
    ou_regime = compute_regime_map(P0, x0, y0, T, n_steps, sigma_grid,
                                    fee_grid, n_grid, model="ou",
                                    mu_long=mu_long, kappa=kappa)
    plot_two_regime_maps(gbm_regime, ou_regime, "GBM (random walk)",
                         "Log-OU (mean-reverting)",
                         out, show, "11_ou_vs_gbm_regimes.png")

    # Chapter 7: LVR decomposition
    plot_lvr_path(baseline_path, out, show)
    plot_lvr_vs_sigma(P0, x0, y0, T, n_steps, fee_base,
                      np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5]),
                      n_sweep, out, show)

    print("\nDone.")


if __name__ == "__main__":
    main(mode="quick", output_dir="figures", show=False)