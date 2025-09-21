"# Stock_hackathon" 

What We Need to Do Summary

- Design an investment strategy

- Use the provided datasets (~9GB returns + ~30GB text filings).

- Predict next month’s stock returns (or fundamentals like earnings ratios).

- Build a long–short global portfolio with 100–250 stocks.

- Rebalance at least semi-annually (every 6 months), though monthly is preferred.

- Evaluate performance out-of-sample (2015–2025)

- Use ML/LLM approaches

- Start from the provided Python templates (penalized_linear_hackathon.py, portfolio_analysis_hackathon.py).

- You can use simple models (Lasso, Ridge, Elastic Net) or get creative with neural nets, transformers, or LLMs on text

- Tune hyperparameters on validation sets; avoid using forward-looking information.

- Backtest and evaluate

- Report portfolio metrics vs S&P500: returns, volatility, alpha, Sharpe, info ratio, drawdowns, turnover

- Compute out-of-sample R² to show statistical predictability.

What to Submit

- Your Final Submission Package must include

- Presentation Deck (PDF)

- Max 5 pages + up to 5 pages appendix.

    Must cover:

    - Executive summary & strategy outline.

    - Portfolio methodology, data, and ML/LLM models.

    - Performance vs S&P500.

    - Key signals driving performance & discussion of results.

- Python Code (Zipped folder)

- All scripts you used, zipped into one file.

    - Clearly label the main run file.

    - Follow Google Python Style Guide and comment thoroughly.

    - Example: main.py, data_cleaning.py, portfolio_analysis.py.

- CVs of all team members for registration and recruiter visibility.