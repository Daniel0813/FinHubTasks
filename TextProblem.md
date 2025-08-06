# FinHub RA Problem Set: Embeddings → Returns

## Objective
Write a runnable Python script that uses a small HuggingFace model to extract text embeddings from financial news (headlines + body) and predicts next-day stock returns via regression. The script should produce a log file capturing key outputs.

## Deliverables
- A **private GitHub repository** containing:
  - A `textanalysis.py` script (or similarly named) that runs end-to-end without errors.
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

---

## Data Organization & Reproducibility

Your script should create a well-organized directory structure to store all intermediate data, results, and outputs. This ensures transparency, reproducibility, and makes it easy to inspect each stage of the pipeline.

### Required Directory Structure
```
project_root/
├── logs/
│   └── run.log                              # Execution logs
├── rawdata/
│   ├── {TICKER}_articles_{dates}.csv        # Scraped news articles
│   ├── {TICKER}_prices_{dates}.csv          # Stock price data
│   └── {TICKER}_returns_{dates}.csv         # Calculated return series
├── data/
│   ├── {TICKER}_embeddings_*.npy            # Saved embeddings for each configuration
│   ├── {TICKER}_pca_model_*.pkl             # Fitted PCA models (if used)
│   ├── {TICKER}_regression_dataset_*.parquet # Final dataset for modeling
│   └── {TICKER}_results_*.json              # Structured results
└── figures/
    └── {TICKER}_predictions_*.svg           # Prediction plots - use SVG format
```

### Data Saving Requirements
1. **Raw Data**: Save the original scraped articles and price data in CSV format
2. **Embeddings**: Store computed embeddings as NumPy arrays for each text processing configuration
3. **Models**: Save any fitted transformers (like PCA) for reproducibility
4. **Final Dataset**: Store the complete feature matrix with all embedding variants in Parquet format
5. **Results**: Save model performance metrics in both human-readable and structured formats
6. **Visualizations**: Generate and save comparison plots

---
## Problem Statement

### 1. Data Collection
1. **Ticker selection**: We will look at AAPL.
2. **News scraping**:  
   - Query the Yahoo Finance news page for your ticker (e.g. `https://finance.yahoo.com/quote/AAPL/news?p=AAPL`).  
   - Grab at least 80 articles (headline, URL, and date) from the past 6 months.  
   - Use `requests` + `BeautifulSoup` or `newspaper3k` to fetch each article’s **body text**.  
3. **Price data**:  
   - Fetch daily adjusted close prices via `yfinance` for the same date range.  
   - Compute **next-day log return**:  
     $$r_{t+1} = \ln(P_{t+1}/P_t)\,. $$

---

### 2. Text Preprocessing
For each article:
1. **Combine** the headline + full body into one text blob.  
2. **Two preprocessing pipelines** (so you can compare):  
   - **Raw text**: minimal cleanup (lowercase, strip extra whitespace).  
   - **Stop-word filtered**: remove English stop-words using NLTK’s list *before* tokenization.  
3. **Tokenization**:  
   - Use a small Transformer tokenizer (e.g. `distilbert-base-uncased`’s `AutoTokenizer`).  
   - **Truncate** to the first **256 tokens**.

---

### 3. Embedding Extraction
1. Load a small HF model (e.g. `distilbert-base-uncased`):
   ```python
   from transformers import AutoTokenizer, AutoModel
   tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
   model = AutoModel.from_pretrained("distilbert-base-uncased")
   ```
2. For each text blob (raw vs. stop-word filtered), compute hidden states and compare two pooling schemes:
   - **Mean-pooling** over tokens:  
     ```python
     embedding = outputs.last_hidden_state.mean(dim=1)
     ```
   - **[CLS]-token**:  
     ```python
     embedding = outputs.last_hidden_state[:, 0, :]
     ```

---

### 4. Feature Engineering & Dimensionality Reduction
With your embeddings `(N_articles × D=768)`:
1. **Direct regression** on embeddings, or  
2. **PCA** to reduce to top 5 principal components.  
3. Assemble a DataFrame linking each article’s date, embedding features (or PCs), and next-day return.

---

### 5. Return Prediction (Regression)
1. **Chronological split**: use the first 70% of unique dates for training and the last 30% for testing.  
2. Fit a simple model (e.g., Linear Regression or Ridge):
   ```python
   from sklearn.linear_model import Ridge
   model = Ridge(alpha=1.0)
   model.fit(X_train, y_train)
   ```  
3. **Evaluate** on the test set:
   - MSE, R², and correlation between predicted vs. actual returns.  
   - Plot predictions vs. actual with a y=x line.  
4. **Compare** results across:
   - Raw vs. stop-word–filtered text  
   - Mean-pooling vs. CLS-token  
   - With vs. without PCA

---

### 6. Logging & Data Persistence
- Use Python's `logging` module to record:
  - Ticker chosen and date range.
  - Number of articles scraped.
  - Shapes of data arrays at each processing stage.
  - File paths where data is saved.
  - Regression metrics (MSE, R², Corr) for all configurations.
  - Summary of which approach performed best.
- Log file should be saved to `logs/run.log`.
- Ensure all intermediate data is saved with meaningful filenames that include:
  - Ticker symbol
  - Date range (YYYYMMDD format)
  - Configuration details (e.g., text processing type, pooling method)
- Save logs both human-readable format (text files) and structured format (JSON) for programmatic access.
---

### 7. Discussion
- Summarize which preprocessing & pooling choices worked best (refer to your saved results files).  
- Interpret key coefficients or PC loadings using the saved model artifacts.  
- Note limitations (sample size, model size, noise).
- Discuss the value of the organized data structure for reproducibility and further analysis.
- Comment on what additional analyses could be performed using the saved embeddings and datasets.

---

### Packages Allowed
```bash
pip install transformers yfinance pandas numpy scikit-learn matplotlib nltk newspaper3k requests beautifulsoup4 lxml lxml_html_clean pyarrow
```

**Note**: You may need to install `lxml_html_clean` separately to resolve compatibility issues with `newspaper3k`:
```bash
pip install lxml_html_clean
```
---

## Contact
Ing-Haw Cheng
inghaw.cheng@rotman.utoronto.ca
