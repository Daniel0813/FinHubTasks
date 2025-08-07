## Project Overview

The project analyzes monthly return performance of two portfolio construction strategies:
- **Equal-Weighted (EW) Portfolio**: Rebalanced to equal weights (1/n) at the start of each calendar year
- **Value-Weighted (VW) Portfolio**: Weighted by stock prices as a proxy for market capitalization

**Analysis Period**: January 2022 - June 2025  
**Stock Universe**: 10 large-cap stocks (MSFT, JNJ, WMT, KO, MCD, HD, PG, INTC, V, XOM)

**Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Dependencies
The project requires the following Python packages:
- `yfinance`: Yahoo Finance data download
- `pandas`: Data manipulation and analysis
- `matplotlib`: Plotting and visualization
- `pandas_market_calendars`: Market calendar functionality
- `numpy`: Numerical computations
- `pyarrow`: Parquet file support

## Usage

### Basic Execution
Run the complete portfolio analysis pipeline:

```bash
python portfolio.py
```

### Data Collection
- **Source**: Yahoo Finance via `yfinance` library
- **Data Type**: Monthly Adjusted Close prices (accounts for dividends and splits)
- **Date Range**: December 2021 through June 2025
- **Return Calculation**: Simple monthly returns: (P_t / P_{t-1}) - 1

### Portfolio Construction

#### Equal-Weighted Portfolio
- **Rebalancing**: Annual rebalancing in January
- **Weights**: w_i,t = 1/n for each stock
- **Formula**: ret_ew = Σ(w_i,t × ret_i,t)

#### Value-Weighted Portfolio
- **Weights**: Based on previous month's stock prices as market cap proxy
- **Calculation**: w_i,t = Price_i,t-1 / Σ(Price_j,t-1)
- **Formula**: ret_vw = Σ(w_i,t × ret_i,t)

### Performance Metrics
- **Mean Monthly Return**: Average of monthly portfolio returns
- **Volatility**: Standard deviation of monthly returns
- **Maximum Drawdown**: Largest peak-to-trough decline in cumulative returns

## Results

### Portfolio Performance Summary (Jan 2022 - Jun 2025)

| Portfolio Type | Mean Monthly Return | Volatility | Max Drawdown |
|----------------|-------------------|------------|--------------|
| Equal-Weighted | 0.74% | 4.16% | -14.84% |
| Value-Weighted | 0.65% | 4.16% | -16.84% |

### Key Findings
- Both portfolios showed positive average monthly returns over the analysis period
- Similar volatility levels between the two strategies
- Equal-weighted portfolio slightly outperformed with higher average returns and lower maximum drawdown
- Both portfolios experienced their maximum drawdowns during market stress periods

## Output Files

### Data Files
- **`portfolio_returns_20250630.parquet`**: Complete dataset with individual stock returns and portfolio returns
- **Raw price data**: Individual CSV files for each ticker in `rawdata/` directory

### Visualizations
- **`portfolio_cumulret.svg`**: Cumulative return comparison showing $1 invested growth trajectory

### Logs
- **`run_portfolio.log`**: Detailed execution log with performance metrics and run information

## AI Usage Attestation

### AI Tools Used
I used GitHub Copilot and VS Code IntelliSense during the development of this project for:
- **Code completion**: Auto-completion of variable names, function calls, and standard Python patterns
- **Syntax assistance**: Help with pandas DataFrame operations and matplotlib plotting syntax
- **Documentation lookup**: Quick reference for function parameters and usage examples

### AI Usage Guidelines Followed
- **Permitted Usage**: Used only code-completion tools and syntax suggestions
- **Code Review**: All AI-generated suggestions were reviewed, tested, and adapted before implementation
- **Understanding**: Every line of code in this project can be fully explained and was validated through testing
- **No Autonomous AI**: Did not use any autonomous AI assistants or black-box AI pipelines

### Personal Contribution
The overall architecture, algorithm logic, problem-solving approach, and validation methodology were designed and implemented by me. AI tools served only as coding assistants to improve development efficiency and code quality.



#### 1. Summarize which preprocessing & pooling choices worked best

- **Pooling Strategy**: Mean pooling generally performed better than CLS token pooling
- **Dimensionality**: PCA reduction to 5 components improved performance over full 768-dimensional embeddings

#### 2. Principal Component Analysis

The PCA analysis on the best-performing configuration (`raw_mean`) revealed:

- **PC1**: Captures 28.7% of variance - likely represents general financial sentiment
- **PC2**: Captures 21.6% of variance - possibly company-specific news themes
- **PC3**: Captures 18.5% of variance - potentially market context and timing
- **PC4**: Captures 14.4% of variance - specific event types or sector influences
- **PC5**: Captures 5.0% of variance - residual semantic patterns

**Total Variance Explained**: 88.1% - indicating the 5 components capture most semantic information while reducing noise.

#### 3. Limitations and Challenges

**Sample Size**: With only 54 article-return pairs for modeling, the dataset is limited for robust statistical inference on 768-dimensional embeddings.

**Model Performance**: All models show negative R² values, indicating predictions perform worse than a simple mean baseline. This suggests:
- Financial return prediction from news is inherently challenging
- The next-day prediction window may be too short for news impact
- Additional features (market context, sentiment scores) might be needed

**Model Size**: DistilBERT, while efficient, is a relatively small transformer. Larger models (BERT-Large, RoBERTa) might capture more nuanced financial language patterns.

**Market Noise**: Financial markets are inherently noisy, and low predictive power is expected for individual stock returns, especially at daily frequency.

#### 4. Value of Organized Data Structure

The structured approach provides significant benefits:

**Reproducibility**: All intermediate data is saved with meaningful timestamps and configuration identifiers, enabling exact replication of results.

**Debugging**: Separate storage of embeddings, models, and datasets allows inspection of each pipeline stage.

**Extensibility**: The modular structure supports easy addition of new:
- Text preprocessing methods
- Embedding models
- Feature engineering techniques
- Evaluation metrics

**Analysis Flexibility**: Saved embeddings can be reused for different modeling approaches without recomputing expensive transformer forward passes.

#### 5. Additional Analyses Possible

The comprehensive data structure enables several extended analyses:

**Temporal Analysis**: 
- Examine embedding patterns across different market regimes
- Analyze correlation between news sentiment and market volatility

**Cross-Sectional Studies**:
- Apply the same pipeline to multiple tickers
- Study sector-specific news impact patterns

**Advanced Modeling**:
- Ensemble methods combining multiple configurations
- Time-series models incorporating embedding sequences
- Multi-task learning for return magnitude and direction

**Semantic Analysis**:
- Cluster analysis of embeddings to identify news themes
- Attention visualization to understand model focus areas
- Topic modeling on high-dimensional embeddings

**Feature Engineering**:
- Combine embeddings with traditional financial indicators
- Engineer features from embedding time series (momentum, volatility)
- Sentiment analysis using specialized financial language models

### Recommendations for Improvement

1. **Data Enhancement**: Increase sample size by extending time period or including multiple tickers
2. **Model Architecture**: Experiment with larger transformer models or financial-domain pretrained models
3. **Feature Engineering**: Incorporate market context, sentiment scores, and technical indicators
4. **Temporal Modeling**: Use sequence models to capture temporal dependencies in news flow
5. **Ensemble Methods**: Combine predictions from multiple configurations

## AI Usage Attestation

This project was completed with the assistance of AI coding tools in accordance with the FinHub RA Problem Set guidelines:

### Permitted AI Usage:
- **GitHub Copilot**: Used for code completion, boilerplate generation, and syntax suggestions
- **VS Code IntelliSense**: Provided autocomplete and function signature assistance
- **Documentation Assistance**: AI-powered search for API documentation and best practices

### AI Contribution Details:
- **Code Structure**: AI assisted in designing the modular class architecture and pipeline organization
- **Implementation**: AI provided syntax suggestions and common patterns for data processing, ML pipelines, and logging
- **Documentation**: AI helped with docstring generation and README structure
- **Debugging**: AI suggested solutions for common issues (e.g., Unicode encoding, package compatibility)

### Human Oversight:
- **Design Decisions**: All architectural and methodological choices were made by human judgment
- **Code Review**: Every AI-generated code snippet was reviewed, tested, and adapted as needed
- **Results Interpretation**: All analysis and discussion points are based on human understanding of the results
- **Problem Solving**: Complex debugging and pipeline integration was done through human reasoning

### What AI Did NOT Do:
- No autonomous code execution or multi-step workflow orchestration
- No black-box model training or hyperparameter optimization
- No independent data analysis or result interpretation
- No automated git commits or repository management

All AI suggestions were treated as draft code that required human validation, testing, and integration into the broader project architecture.

## Dependencies

Key packages and their purposes:

### Core ML/Data Science
- `pandas` (2.0+): Data manipulation and analysis
- `numpy` (1.24+): Numerical computations
- `scikit-learn` (1.3+): Machine learning models and evaluation
- `transformers` (4.30+): HuggingFace transformer models
- `torch` (2.0+): PyTorch for deep learning

### Data Collection
- `yfinance` (0.2+): Stock price data
- `requests` (2.31+): HTTP requests for web scraping
- `beautifulsoup4` (4.12+): HTML parsing
- `newspaper3k` (0.2+): Article content extraction

### Text Processing
- `nltk` (3.8+): Natural language processing
- `lxml` (4.9+): XML/HTML processing
- `lxml_html_clean` (0.1+): HTML cleaning for newspaper3k

### Data Storage/Visualization
- `pyarrow` (12.0+): Parquet file format
- `matplotlib` (3.7+): Plotting and visualization

### Utility
- Standard library modules: `os`, `sys`, `logging`, `json`, `pickle`, `datetime`, `traceback`

