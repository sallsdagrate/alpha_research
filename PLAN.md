# Crypto Perpetual Futures Alpha Research Project — Source of Truth

## Project Purpose

This project is a serious alpha research project intended for a quant trading / quant research CV. The goal is not to build a toy trading bot or generic crypto price predictor. The goal is to demonstrate a rigorous end-to-end research process for discovering, validating, and stress-testing potential alpha signals in crypto perpetual futures markets using freely available data.

The project should show competence in:

* market microstructure reasoning
* data cleaning and alignment
* feature engineering
* statistical signal testing
* time-series validation
* realistic backtesting assumptions
* execution-cost awareness
* research communication

The final project should look like something a junior quant trader, desk quant, or quant researcher might discuss in an interview.

---

## Asset Class

The chosen asset class is **crypto perpetual futures**, with a focus on liquid USD/USDT-margined perpetual contracts.

The main reasons for choosing this asset class are:

1. **Free data availability**
   Crypto exchanges publish substantial public market data, including trades, klines, funding rates, open interest, mark prices, and sometimes order-book snapshots or live book updates.

2. **Non-equity focus**
   This avoids the very common retail-style equities projects that many candidates use.

3. **Rich microstructure**
   Crypto markets have fragmented venues, spot/perp relationships, funding mechanisms, high retail participation, and possible short-horizon inefficiencies.

4. **Good fit for the user’s background**
   The user has software engineering, Rust, trading systems, market data, and quantitative research interests. This asset class allows the project to combine research and engineering.

---

## Primary Market Universe

Initial research should focus on highly liquid instruments only.

Core instruments:

```text
BTCUSDT
ETHUSDT
```

Expansion instruments:

```text
SOLUSDT
XRPUSDT
BNBUSDT
DOGEUSDT
```

Further expansion can include the top 10–20 liquid perpetual contracts by volume, but only after the core research pipeline is robust.

The primary exchange should be Binance because of public data availability and liquidity. Bybit, Coinbase, or other venues may be added later for cross-venue lead-lag research.

---

## Core Research Question

The central research question is:

> Can spot/perp order flow, funding/basis dislocations, order-book imbalance, and cross-venue lead-lag effects predict short-horizon returns in liquid crypto perpetual futures after realistic costs?

The project should avoid vague claims like “predicting Bitcoin price”. The framing should be about measurable alpha hypotheses, signal decay, costs, and robustness.

---

## Four Main Alpha Hypotheses

The project is organized around four research hypotheses.

---

## Hypothesis 1: Spot Leads Perp

### Idea

Spot markets may sometimes incorporate information before perpetual futures markets. Aggressive buying or selling in spot may lead short-horizon movements in the corresponding perp contract.

### Research Question

> Does spot order flow or spot price movement predict future perp returns?

### Example Relationships

```text
BTCUSDT spot → BTCUSDT perp
ETHUSDT spot → ETHUSDT perp
SOLUSDT spot → SOLUSDT perp
```

### Potential Features

* spot return over 1s, 5s, 30s, 1m
* spot signed volume imbalance
* spot trade intensity
* spot volatility burst
* spot VWAP deviation
* spot-perp price divergence
* spot-perp basis change

### Potential Labels

* future perp return over 5s, 30s, 1m, 5m
* future perp return sign
* future perp return after estimated taker costs

### Key Validation Concern

Avoid fake alpha from timestamp misalignment. Spot and perp data must be aligned carefully, and features should only use information available at or before the prediction timestamp.

---

## Hypothesis 2: Funding / Basis Mean Reversion

### Idea

Perpetual futures have funding payments and can trade at a premium or discount to spot/index prices. Extreme funding or basis may indicate crowded positioning and future mean reversion.

### Research Question

> Do funding rates, perp basis, and open-interest changes predict future perp returns or basis compression?

### Potential Features

* funding rate
* funding rate rolling z-score
* mark-index basis
* spot-perp basis
* basis rolling z-score
* open-interest change
* open-interest rolling z-score
* price return × open-interest change
* funding × momentum interaction
* funding × basis interaction
* time to next funding event

### Example Interpretations

```text
price up + open interest up + high positive funding
= crowded leveraged longs

price down + open interest up + negative funding
= crowded leveraged shorts

extreme basis + falling open interest
= possible unwind / basis compression
```

### Potential Labels

* future return over 15m, 1h, 4h, 8h
* return until next funding event
* future basis compression
* future return after funding adjustment

### Key Validation Concern

Funding/basis signals are lower-frequency than order-flow signals. They should not be evaluated only on ultra-short horizons.

---

## Hypothesis 3: Order-Flow Imbalance / Microprice Alpha

### Idea

Short-horizon price movement may be predictable from trade aggressor imbalance, order-book imbalance, spread, microprice, and liquidity skew.

### Research Question

> Does order-book imbalance or aggressive trade flow predict very short-horizon perp returns?

### Potential Features

* bid/ask spread
* spread in basis points
* L1 order-book imbalance
* L2 depth imbalance
* microprice
* microprice deviation from mid
* aggressive buy volume
* aggressive sell volume
* signed trade imbalance
* trade count imbalance
* trade intensity
* quote replenishment rate, if L2 data is available
* liquidity skew

### Potential Labels

* future mid-price return over 1s, 5s, 10s, 30s
* next-tick direction
* probability price moves up before moving down
* future return net of spread and taker fee

### Key Validation Concern

This is the most sensitive hypothesis to costs, latency, and timestamp quality. Apparent alpha may disappear once execution assumptions are made realistic.

---

## Hypothesis 4: Cross-Venue Lead-Lag

### Idea

Crypto markets are fragmented across venues. One venue may sometimes lead another due to differences in liquidity, trader composition, latency, or market structure.

### Research Question

> Do price or order-flow changes on one venue predict short-horizon returns on another venue?

### Initial Relationships

```text
Binance spot → Binance perp
Binance perp → Bybit perp
Coinbase spot → Binance perp
BTC perp → major alt perps
ETH perp → major alt perps
```

### Potential Features

* leader venue return
* follower venue return
* cross-venue price spread
* lagged leader returns
* lagged leader signed volume imbalance
* relative volume
* relative volatility
* relative spread
* delayed leader features at 100ms, 500ms, 1s, 5s

### Potential Labels

* follower venue future return
* follower venue next-tick direction
* follower venue return after assumed latency delay

### Key Validation Concern

Lead-lag alpha is especially vulnerable to fake results from inconsistent timestamps, clock drift, and using data that would not have been available in real time.

---

## Data Philosophy

The project should use public/free data where possible.

Priority data types:

1. trades
2. klines / bars
3. funding rates
4. mark price and index price
5. open interest
6. order-book snapshots or live book updates
7. cross-venue data, if available

Infrastructure should only download and preserve original source files. Each
research notebook should perform the cleaning and construct the dataset needed
for its hypothesis. Notebook-built tables should make these fields explicit:

* timestamp
* exchange
* symbol
* market type: spot or perp
* relevant price/volume fields
* clear timezone treatment, ideally UTC

The project should prefer reproducible public historical datasets where possible. Live collection can be added later, but the research should not depend entirely on personally collected live data.

---

## Research Philosophy

The project should follow a proper alpha research workflow:

```text
hypothesis → data → features → labels → signal tests → validation → backtest → report
```

The agent should not jump straight to ML models. First, establish whether individual features have predictive value through simple statistical tests.

Preferred order:

1. data quality checks
2. feature sanity checks
3. forward-return label construction
4. information coefficient analysis
5. signal decay analysis
6. simple threshold strategies
7. linear/logistic models
8. tree-based models
9. backtesting with costs
10. robustness analysis

Simple models with strong validation are preferred over complex models with weak validation.

---

## Modelling Philosophy

The project should start with interpretable baselines.

Useful baseline models:

* naive momentum
* naive reversal
* basis z-score mean reversion
* order-flow imbalance threshold model
* funding crowding threshold model
* linear regression
* ridge regression
* logistic regression

Only after baselines should the project use:

* random forest
* LightGBM
* XGBoost
* calibrated classifiers

Deep learning should not be a priority unless the simpler research already shows strong signal and the data volume justifies it.

---

## Validation Standards

Validation is one of the most important parts of the project.

The project should include:

* chronological train/test split
* walk-forward validation
* out-of-sample by date
* out-of-sample by asset
* out-of-sample by venue, if multiple venues are used
* latency sensitivity
* cost sensitivity
* feature IC decay
* robustness across market regimes

Important metrics:

```text
IC
rank IC
hit rate
AUC
mean return by prediction bucket
average trade return
average trade return after costs
Sharpe
Sortino
max drawdown
turnover
PnL by symbol
PnL by regime
```

The project should distinguish between:

* a signal that predicts returns statistically
* a signal that survives costs
* a signal that is realistically tradeable

---

## Backtesting Philosophy

The backtester does not need to be institutional-grade, but it must avoid common retail errors.

It must not:

* trade at the same close used to generate the signal
* ignore spread
* ignore fees
* ignore slippage
* ignore funding payments for perps
* optimize directly on the test set
* assume perfect fills at mid price

Basic execution assumptions:

* market/taker buy executes at next available ask
* market/taker sell executes at next available bid
* taker fees are applied
* configurable slippage is applied
* configurable latency delay is applied
* funding PnL is applied when holding perp positions across funding events
* position limits are enforced

The goal is not to prove a live-tradeable strategy immediately. The goal is to show honest research under increasingly realistic assumptions.

---

## Non-Goals

The project is not intended to be:

* a live trading bot
* a crypto dashboard
* a generic Bitcoin prediction model
* a reinforcement learning trading project
* a technical-analysis indicator project
* a project optimized for impressive in-sample PnL
* a black-box ML project with no market intuition

The agent should avoid turning the project into any of these.

---

## Expected Final Output

The final project should ideally produce:

1. notebook code that turns raw market data into an analysis-ready table
2. reusable feature-generation code
3. forward-return labels
4. signal IC analysis
5. alpha decay plots
6. baseline model results
7. cost-aware backtest results
8. latency sensitivity results
9. research report
10. clean README
11. credible CV bullet

The most important final deliverable is a research report explaining:

* what was tested
* why it was tested
* what worked
* what failed
* whether the signal survived costs
* what assumptions were made
* what should be tested next

---

## Communication Style for the Agent

When helping with this project, the agent should act like a quant research assistant.

The agent should:

* preserve the four-hypothesis structure
* keep research claims grounded
* flag lookahead bias risks
* flag unrealistic execution assumptions
* prefer incremental research steps
* separate hypothesis testing from backtesting
* explain design decisions in quant/trading terms
* avoid unnecessary complexity
* keep the repo CV/interview-oriented
* produce code that is clean, modular, and testable when asked

The agent should not assume the goal is to build everything at once. The source of truth is the research plan above. Individual implementation tasks should be derived from this plan one step at a time.

---

## Candidate CV Framing

The project should eventually support a CV bullet similar to:

> Built a crypto perpetual futures alpha research platform using public tick, funding, basis and open-interest data; engineered order-flow, cross-venue lead-lag and funding-crowding signals; evaluated signal decay and out-of-sample performance using walk-forward validation and a cost-aware simulator with spread, fees, slippage, latency and funding costs.

This is the target standard for the project.
