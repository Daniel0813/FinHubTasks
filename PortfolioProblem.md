# FinHub RA Problem Set: Portfolio Construction
Prof. Ing-Haw Cheng, inghaw.cheng@rotman.utoronto.ca

## Objective
Compare monthly return performance of equal-weighted (EW) versus value-weighted (VW) portfolios for a fixed universe of ten stocks, starting January 2022.

## Deliverables
- A **private GitHub repository** containing:
  - A `portfolio.py` script (or similarly named) that runs end-to-end without errors.
  - An `environment.yml` or `requirements.txt` specifying all dependencies.
  - A `README.md` with clear setup, run instructions, and your AI attestation.
- The script should create an organized directory structure and save all intermediate data for reproducibility.
- Share the GitHub repository link (private, with access).

If you are completing this in conjunction with other FinHub RA Problem Set problems, you should deliver a **single repository** that with a **single unified directory structure, unified README.md, and unified dependency file (either environment.yml or requirements.txt).**

## Best Practices and AI Usage Statement
#### Code Quality Best Practices
- Modular Design
  - Break your solution into well-defined functions or classes.
  - Each module should have a single responsibility and clear inputs/outputs.
- Thorough Commenting
  - Every function/class must include a docstring describing purpose, inputs, outputs, and any side effects.
  - Inline comments should explain “why,” not just “what.”
- Readability & Style
  - Follow a consistent style guide (e.g. PEP 8 for Python).
  - Use meaningful variable and function names.
- Adopt a `fail fast` approach: Exceptions should terminate your script with an explicit error statement.

#### Allowable AI Usage
- Permitted:
  - Code-completion tools (e.g., GitHub Copilot, TabNine, VS Code IntelliSense) that assist you as you type.
  - Syntax & snippet suggestions to speed up boilerplate or standard patterns.
  - Documentation look-ups via AI-powered search (provided you validate accuracy).
- Not Permitted:
  - Agentic or autonomous AI assistants (e.g., Claude Code, Cursor agents) that independently generate, orchestrate, or execute multi-step workflows on your behalf.
  - “Black-box” AI pipelines that you cannot fully inspect, version-control, or explain.
- Responsibility & Attribution
  - Treat AI suggestions as draft code: review, test, and adapt them before committing.
  - Your `README.md` must contain a clear attestation of how you used AI to complete this assignment.

## Data Organization & Reproducibility

Your script should create a well-organized directory structure to store all intermediate data, results, and outputs. This ensures transparency, reproducibility, and makes it easy to inspect each stage of the pipeline.

### Required Directory Structure

```
project_root/
├── logs/
│   └── run_portfolio.log                         # Execution logs
├── rawdata/
│   ├── {TICKER}_prices_{enddate}.csv             # Downloaded data
├── data/
│   ├── portfolio_returns_{enddate}.parquet       # Clean dataset of stock and portfolio returns
└── figures/
    └── portfolio_cumulret.svg                    # Cumulative return plot
```

---

## Problem Statement

### 1. Data Collection
- Fetch monthly **Adjusted Close** prices via Yahoo Finance (`yfinance`) to account for dividends.
- Date range: December 2021 through the end of June 2025.
- Tickers: MSFT, JNJ, WMT, KO, MCD, HD, PG, INTC, V, XOM

### 2. Calculations

1. **Stock Returns**  
   - Compute each stock’s monthly return.
   - For this exercise, focus on simple returns.
   - The first month's return should be January 2022.

2. **Equal‑Weighted (EW) Portfolio Return**  
   - Your portfolio should rebalance to equal weights ($1/n$) at the start of each calendar year.

3. **Value‑Weighted (VW) Portfolio Return**  
   - Conceptually, your portfolio should weight stock returns by each stock's market cap at the end of the previous month.
   - For this problem, you should weight stock returns by the stock price itself as a substitute for market cap due to data limitations.

4. **Calculate and graph portfolio performance**
   - Calculate the average, standard deviation, and maximum drawdown of monthly returns.
   - Produce a single plot that plots the cumulative return of $1 invested in each portfolio at the start of January 2022 (equivalent to end of December 2021) through the end of June 2025.

5. **(Optional, for extra credit) Build your own monthly series**
    - As an optional path, you can extract daily data from Yahoo Finance and construct your own monthly series.
    - Your month-end price should be the last available `adjclose` price each month as determined by the `pandas_market_calendars` NYSE trading calendar.
    - Once you have constructed your monthly dataset, proceed as before.
    
### 3. Output & Logging  
- **Script:** `portfolio.py` runs end‑to‑end.  
- **Log file:** `run_portfolio.log` capturing:
  - Date range and tickers  
  - Number of observations (months)  
  - Portfolio summary statistics:
    - Mean monthly return  
    - Monthly volatility (standard deviation)  
    - Maximum drawdown
- **Figure:** A single **SVG** (Scalable Vector Graphics) file for your figure
- **Data file:** `portfolio_returns.parquet` with columns `[date, ret_{tickers}, ret_ew, ret_vw]`.
- **Date format:** Where applicable, adopt YYYYMMDD as your date convention.
---

## Contact
Ing-Haw Cheng
inghaw.cheng@rotman.utoronto.ca
