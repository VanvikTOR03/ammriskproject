# AMM Liquidity Provider Risk Simulator

## Overview

This project studies the economics of providing liquidity to a **constant-product automated market maker (AMM)**, using a simplified ETH/USDC-style pool as the running example.

The central question is straightforward:

> When can trading fees compensate a liquidity provider for the structural cost of AMM rebalancing, and when is passive holding still the better strategy?

The simulator starts from the mechanics of a full-range constant-product pool, then adds stochastic price paths, arbitrage-driven reserve changes, trading fees, Monte Carlo analysis, market-regime maps, and a discrete Loss-Versus-Rebalancing (LVR) proxy.

The project is designed as a transparent analytical benchmark. It is not a forecast of realised ETH/USDC returns and it does not attempt to reproduce every detail of Uniswap execution.

The accompanying report (`Report_Liquidity_Provision_in_Constant-Product_AMMs.pdf`) presents the full analysis, including the theoretical background, methodology, and interpretation of all results.

---

## Main idea

A constant-product AMM satisfies

$$
x \cdot y = k,
$$

where:

- $x$ is the base-asset reserve, for example ETH;
- $y$ is the quote-asset reserve, for example USDC;
- $k$ is the invariant.

Once arbitrage has aligned the AMM price with the external market price $P$, the pool reserves satisfy

$$
P = \frac{y}{x}, \qquad
x = \sqrt{\frac{k}{P}}, \qquad
y = \sqrt{kP}.
$$

This means that the pool automatically changes its inventory as the external price moves:

- when ETH rises, the pool ends up holding less ETH and more USDC;
- when ETH falls, the pool ends up holding more ETH and less USDC.

The LP is therefore not simply holding a fixed 50/50 portfolio. The AMM continuously rebalances it through arbitrage. The LP earns fees for providing this liquidity, but also gives up value when the AMM executes against a stale pool price.

A key architectural detail is that the AMM is a deterministic on-chain smart contract. It cannot query external price feeds and only updates its reserves when transactions are submitted to it. As a result, the pool always trades at a slightly stale price relative to the external market, and arbitrage is the sole mechanism through which external price information enters the pool. The structural cost this imposes on the LP is what the project analyses.

---

## What the project models

The script is organised as one connected storyline.

### 1. Static AMM mechanics

The first chapter derives the reserve composition of a constant-product pool at any external price. It then compares two benchmark positions:

- **HODL:** keep the initial ETH and USDC quantities unchanged;
- **LP before fees:** provide the same initial assets to the AMM and allow the pool to rebalance.

The static difference between the two is impermanent loss:

$$
\mathrm{IL}(r) = \frac{2\sqrt{r}}{1+r} - 1,
$$

where $r = P_T/P_0$ is the final price ratio.

The formula has two useful properties:

- impermanent loss is zero only if the final price equals the initial price;
- a price doubling and a price halving produce the same percentage loss relative to HODL.

### 2. Price dynamics

The project uses two stylised price processes.

**Geometric Brownian Motion (GBM)** is the baseline process. It represents a price that evolves through random percentage shocks and can drift away from its initial level. It is the appropriate stylised model for assets without a structural price anchor, such as ETH/USDC, BTC/USDC, or most fiat-quoted crypto pairs.

**Log-Ornstein-Uhlenbeck (log-OU)** is a mean-reverting alternative, applied in log-price space to preserve positivity. It is useful as a regime comparison for relative-price pairs with a stable long-run anchor, such as stablecoin pairs (USDC/USDT), liquid staking derivative pairs (ETH/stETH), or wrapped-token pairs (wBTC/BTC).

### 3. LP accounting over a path

For every price step, the script:

1. moves the AMM reserves to the new arbitrage-consistent state;
2. calculates arbitrage-related volume in USDC;
3. credits the LP with fees from this volume;
4. optionally adds independent fee-generating noise volume;
5. compares LP value with the initial HODL benchmark;
6. tracks a step-wise LVR proxy.

This allows the project to separate the two sides of liquidity provision:

$$
\text{LP outcome} = \text{inventory value after rebalancing} + \text{fee income}.
$$

### 4. Monte Carlo analysis

One price path is only an illustration. The project therefore simulates many independent paths and reports:

- mean LP performance relative to HODL;
- median LP performance;
- LP win rate, meaning $P(\text{LP} > \text{HODL})$;
- 5th and 95th percentiles;
- mean fee income;
- mean discrete LVR proxy.

### 5. Regime maps

The script then varies two economically important inputs:

- annual volatility $\sigma$;
- fee rate.

The resulting heatmaps show where the model finds favourable or unfavourable LP conditions. This makes the project less about one ETH/USDC example and more about the broader relationship between volatility, fee income, and AMM inventory risk.

### 6. Extensions

The project adds three extensions to the baseline model.

**Noise-trader volume** adds exogenous fee-generating flow, specified as a daily fraction of pool TVL. In this extension, noise volume does not move the pool price or reserves; it is intentionally modelled as an isolated source of fee income.

**Mean-reverting prices** compare the GBM benchmark with a log-OU process. This helps show that volatility alone is not enough to describe LP economics: the persistence of price divergence from the initial level also matters.

**Loss-Versus-Rebalancing (LVR)** tracks a discrete proxy for the value lost when the AMM rebalances against an out-of-date price. At each price change, the AMM exchanges assets through its curve while the external market has already moved. The proxy measures the gap between the value transferred through that AMM rebalance and the value that would have been obtained at the new external price. This isolates the adverse-selection component of AMM liquidity provision, following the framework of Milionis et al. (2022). In the model, fees are the compensation mechanism; LVR is the structural cost proxy.

---

## Terminology

| Term | Meaning in this project |
|---|---|
| **AMM** | Automated market maker: a trading mechanism that prices assets from reserves rather than an order book. |
| **Constant-product AMM** | An AMM with invariant $x \cdot y = k$, similar in spirit to a full-range Uniswap V2 pool. |
| **LP** | Liquidity provider: someone who deposits both assets into the pool and earns trading fees. |
| **HODL** | Passive benchmark: keep the original quantities of ETH and USDC without rebalancing. |
| **Impermanent loss (IL)** | LP underperformance relative to HODL caused by automatic reserve rebalancing after a price move. |
| **Arbitrage** | Trading that brings the AMM price back in line with the external market price. |
| **GBM** | Geometric Brownian Motion: a random-walk-style process for positive prices. |
| **Log-OU** | Mean-reverting process applied to log prices, keeping simulated prices positive. |
| **Noise volume** | Additional fee-generating volume assumed to be independent of price changes. |
| **LVR proxy** | A discrete approximation of value leakage through stale-price AMM rebalancing. |
| **Adverse selection** | Systematic value transfer from a less-informed counterparty (the AMM trading at a stale price) to a better-informed one (arbitrageurs trading on fresh external prices). |
| **Win rate** | Share of simulations in which LP with fees finishes above HODL. |

---

## Baseline setup

The default baseline in `main()` uses:

| Parameter | Baseline value |
|---|---:|
| Initial ETH price | 2,000 USDC |
| Initial pool reserves | 1 ETH and 2,000 USDC |
| Initial LP / HODL value | 4,000 USDC |
| Time horizon | 1 year |
| Time steps | 365 daily steps |
| Baseline annual volatility | 60% |
| Baseline fee rate | 0.30% |
| GBM drift | 0% |
| OU long-run price | 2,000 USDC |
| OU speed of mean reversion | $\kappa = 4$ per year |

These values are inputs for scenario analysis. They are not calibrated estimates or forecasts.

---

## Key outputs

The script creates a `figures/` directory next to the Python file and saves all charts as PNG files.

The figures, in the order they appear in the accompanying report:

1. **`01_il_curve.png`** — Impermanent loss curve as a function of terminal price ratio.
2. **`02_gbm_paths.png`** — Five sample GBM price paths.
3. **`02_ou_paths.png`** — Five sample log-OU price paths.
4. **`03_single_path.png`** — LP, LP-with-fees, and HODL values along one GBM path.
5. **`04_fee_decomposition.png`** — Impermanent loss and accumulated fees on the same path.
6. **`05_distribution_baseline.png`** — Monte Carlo distribution of LP outcomes at baseline.
7. **`06_volatility_sweep.png`** — Three-panel volatility sensitivity (mean LP-HODL, win rate, mean fees).
8. **`07_regime_gbm.png`** — Market-regime map across volatility and fee rate (GBM).
9. **`08_noise_effect.png`** — Regime map with and without noise volume, plus win-rate gain.
10. **`10_gbm_vs_ou_distribution.png`** — Distribution comparison of LP outcomes under GBM and log-OU.
11. **`11_ou_vs_gbm_regimes.png`** — Side-by-side regime maps for the two price processes.
12. **`12_lvr_path.png`** — Cumulative LVR proxy versus cumulative fees on one path.
13. **`13_lvr_vs_sigma.png`** — Mean LVR and fees across volatility levels.

Note: figure index `09` is intentionally skipped; figure numbering preserves the order used in the report.

---

## Installation

The project only requires NumPy and Matplotlib.

```bash
pip install numpy matplotlib
```

---

## How to run

Place the script and `README.md` in the same repository folder.

Run the script with:

```bash
python amm_lp_risk_simulator.py
```

The current default configuration at the bottom of the file is:

```python
main(mode="quick", output_dir="figures", show=False)
```

It saves figures to the `figures/` folder but does not open windows.

For publication-quality figures with more Monte Carlo simulations, change the last line to:

```python
main(mode="full", output_dir="figures", show=False)
```

For visual inspection in Spyder, VS Code, or a local Python environment, use:

```python
main(mode="full", output_dir="figures", show=True)
```

The `full` mode uses more Monte Carlo simulations and produces smoother, more stable output. It is the recommended mode for the figures used in the report.

---

## Project structure

```text
amm-lp-risk-simulator/
├── amm_lp_risk_simulator.py
├── Report_Liquidity_Provision_in_Constant-Product_AMMs.pdf
├── README.md
├── requirements.txt
└── figures/
    ├── 01_il_curve.png
    ├── 02_gbm_paths.png
    ├── 02_ou_paths.png
    ├── 03_single_path.png
    ├── 04_fee_decomposition.png
    ├── 05_distribution_baseline.png
    ├── 06_volatility_sweep.png
    ├── 07_regime_gbm.png
    ├── 08_noise_effect.png
    ├── 10_gbm_vs_ou_distribution.png
    ├── 11_ou_vs_gbm_regimes.png
    ├── 12_lvr_path.png
    └── 13_lvr_vs_sigma.png
```

---

## Assumptions and limitations

The purpose of this project is clarity, not a complete replication of a live DEX.

The model assumes:

- a full-range constant-product pool;
- exogenous external prices;
- immediate arbitrage after every time step;
- a fixed LP share of 100% of the simulated pool;
- fees added as external LP income rather than reinvested into the invariant $k$;
- noise volume that generates fees but does not change reserves or price;
- no gas costs, slippage beyond the AMM curve, MEV, liquidity migration, or dynamic fee changes;
- no concentrated liquidity, active range management, or Uniswap V3 position management;
- no empirical calibration to realised ETH/USDC volume or on-chain data.

The LVR quantity in this script is a **discrete step-wise proxy**. It is intended to make the adverse-selection mechanism visible in a transparent simulation. It should not be read as an exact replication of a continuous-time theoretical LVR result.

For the same reason, the model should not be used as investment advice or as a forecast of realised liquidity-provider returns.

---

## How to interpret the results

The right conclusion is not that LP is always bad or that a specific pool must always lose money.

The model instead makes the trade-off explicit:

- stronger and more persistent price moves increase the cost of automatic AMM rebalancing;
- higher fees and additional trading volume improve LP compensation;
- the price process matters, not only annual volatility;
- mean-reverting relative-price regimes can look very different from directional random-walk-style regimes;
- LP profitability should be analysed as a balance between inventory risk, arbitrage leakage, and fee income.

A useful question for each scenario is therefore:

> Do the fees earned by the LP compensate the economic cost of keeping liquidity available at the AMM curve?

The accompanying report develops this question across the three extensions and concludes that fee compensation holds in three regimes: low volatility combined with high fees, mean-reverting price dynamics, and pools with substantial non-arbitrage volume. The illustrative ETH/USDC-style baseline at a 0.30% fee falls outside these regimes under the model’s assumptions.

---

## Possible extensions

Natural next steps would include:

- empirical calibration to on-chain ETH/USDC data;
- reserve-level fee compounding;
- concentrated liquidity and active range management (Uniswap V3);
- gas costs and rebalancing costs;
- dynamic fee tiers;
- liquidity competition between multiple LPs;
- explicit MEV and sandwich-attack modelling;
- comparison with alternative AMM curves;
- stochastic volatility (Heston-style) and jump processes (Merton-style).

---

## Author

Viktor Vantsev  
Master of Economics, University of St. Gallen  
Independent FinTech / DeFi analytics project, 2026
