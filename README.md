# ![CI logo](https://codeinstitute.s3.amazonaws.com/fullstack/ci_logo_small.png)

# Insider Trading Analysis and Prediction Platform

## Table of Contents
- [Dataset Content](#dataset-content)
  - [1. Insider Trading Data Set](#1-insider-trading-data-set)
  - [2. Financial Data (2014-2018)](#2-financial-data-2014-2018)
  - [3. Stock Price Data](#3-stock-price-data)
- [Business Requirements](#business-requirements)
- [Hypotheses and How to Validate](#hypotheses-and-how-to-validate)
- [The Rationale to Map the Business Requirements to the Data Visualisations and ML Tasks](#the-rationale-to-map-the-business-requirements-to-the-data-visualisations-and-ml-tasks)
- [ML Business Case](#ml-business-case)
- [Dashboard Design](#dashboard-design)
- [Unfixed Bugs](#unfixed-bugs)
- [Deployment](#deployment)
- [Main Data Analysis and Machine Learning Libraries](#main-data-analysis-and-machine-learning-libraries)
- [Commit Messages Prefixes](#commit-messages-prefixes)
- [Credits](#credits)
- [Acknowledgements](#acknowledgements)

---

## Dataset Content

The Insider Trading Analysis and Prediction Platform utilises three primary datasets to analyse and predict stock price movements based on insider trading activities and financial indicators. The datasets have been carefully collected and processed to ensure they are of reasonable size, optimising model training time and adhering to repository size limitations.

### 1. Insider Trading Data Set
- **Source:** U.S. Securities and Exchange Commission (SEC) EDGAR Database.
- **Description:** This dataset contains detailed records of insider transactions filed with the SEC. Each record represents a transaction made by a company insider, such as executives, directors, or significant shareholders.

#### Data Source:
You can access the insider transactions dataset from the SEC at the following link: [SEC Insider Transactions Dataset](https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets)

#### Focus on Non-Derivative Transactions:
The project focuses on non-derivative insider buys which alignes with the hypothesis of the project and offers a more straightforward and robust signals about stock price impact.
This project emphasizes non-derivative insider buys for the following key reasons:

1. **Clarity in Data**:  
   Non-derivative transactions avoid complexities such as options and conversion rates, resulting in cleaner data for analysis. They directly reflect changes in insider ownership, simplifying the modeling of buying activity.

2. **Market Interpretation**:  
   The market generally interprets non-derivative insider buying (e.g., purchasing common shares) as a stronger and clearer bullish signal compared to derivative transactions, which may serve as hedges or have limited impact on actual share prices.

3. **Direct Ownership Signal**:  
   Non-derivative transactions indicate direct purchases or sales of company shares, showcasing an insider’s immediate and vested interest in the stock's performance. For instance, buying company stock directly signals confidence in the company’s future.

4. **Past Studies and Predictive Insights**:  
   Historical research often shows a stronger correlation between non-derivative insider buying and future price movements, as it explicitly expresses confidence from company leaders.

#### Data Structure and Variables:
The insider trading data is organized into quarterly folders, each containing multiple TSV (Tab-Separated Values) files corresponding to different tables from the SEC filings.

**Primary Tables and Variables:**
- **Trade Date:**
  - **Table:** NONDERIV_TRANS
  - **Variable:** TRANS_DATE
  - **Description:** Transaction date in DD-MON-YYYY format.
  
- **Ticker:**
  - **Table:** SUBMISSION
  - **Variable:** ISSUERTRADINGSYMBOL
  - **Description:** Issuer's trading symbol.

- **Company Name:**
  - **Table:** SUBMISSION
  - **Variable:** ISSUERNAME
  - **Description:** Name of the issuer.

- **Insider Name:**
  - **Table:** REPORTINGOWNER
  - **Variable:** RPTOWNERNAME
  - **Description:** Name of the reporting owner.

- **Insider Position:**
  - **Table:** REPORTINGOWNER
  - **Variables:** RPTOWNER_RELATIONSHIP, RPTOWNER_TITLE
  - **Description:** Role of the insider, detailing if they are an OFFICER, DIRECTOR, TENPERCENTOWNER, or OTHER.

- **Transaction Type:**
  - **Table:** NONDERIV_TRANS
  - **Variable:** TRANS_CODE
  - **Description:** Transaction code representing the type of transaction (e.g., Purchase, Sale).

- **Price:**
  - **Table:** NONDERIV_TRANS
  - **Variable:** TRANS_PRICEPERSHARE
  - **Description:** Price per share for the transaction.

- **Quantity:**
  - **Table:** NONDERIV_TRANS
  - **Variable:** TRANS_SHARES
  - **Description:** Number of shares traded.

- **Owned Shares:**
  - **Table:** NONDERIV_TRANS and NONDERIV_HOLDING
  - **Variable:** SHRS_OWND_FOLWNG_TRANS
  - **Description:** Shares owned following the reported transaction(s).

- **Value:**
  - **Table:** NONDERIV_TRANS
  - **Variable:** VALU_OWND_FOLWNG_TRANS
  - **Description:** Value owned following the reported transaction(s), typically calculated as Price × Quantity for transactions.

#### Key Variables and Data Types:

| Variable Name       | Data Type | Description                                              |
|---------------------|-----------|----------------------------------------------------------|
| Trade Date          | Date      | Date when the transaction occurred.                      |
| Ticker              | String    | Stock ticker symbol of the company.                      |
| Company Name        | String    | Name of the company.                                    |
| Insider Name        | String    | Name of the insider conducting the transaction.         |
| Insider Position     | String    | Role of the insider within the company.                  |
| Transaction Type    | Categorical | Type of transaction (e.g., Purchase, Sale) represented by codes. |
| Price               | Float     | Transaction price per share.                             |
| Quantity            | Integer   | Number of shares traded.                                 |
| Owned Shares        | Integer   | Shares owned after the transaction.                      |
| Value               | Float     | Total value of the transaction (Price × Quantity).      |

#### Data Collection and Processing:
- **Data Retrieval:**
  - Downloaded quarterly filings from the SEC EDGAR database.
  - Extracted relevant tables (NONDERIV_TRANS, SUBMISSION, REPORTINGOWNER) from TSV files.
  
- **Data Cleaning:**
  - Merged tables using common identifiers to consolidate data.
  - Handled missing or null values appropriately.
  - Standardised date formats and decoded transaction codes.

- **Data Storage:**
  - Organised in CSV files for efficient querying and analysis.

### 2. Financial Data (2014-2018)
- **Source:** Collected from company financial statements and public records.
- **Description:** This dataset includes key financial metrics and ratios for companies from 2014 to 2018, stored in five CSV files corresponding to each year.

#### Data Structure and Variables:
Each CSV file contains financial data for multiple companies for a specific year.

#### Data Source:
You can access the insider transactions dataset from kaggle at the following link: [200+ Financial Indicators of US stocks (2014-2018)](https://www.kaggle.com/datasets/cnic92/200-financial-indicators-of-us-stocks-20142018)

**Key Variables and Corresponding Columns:**
- **Ticker:**
  - **Column:** Unnamed: 0
  - **Description:** Contains the stock ticker symbol.

- **Fiscal Date Ending:**
  - **Column:** Not explicitly present. May use the file's year or other available data as a proxy.

- **Gross Profit:**
  - **Column:** Gross Profit
  - **Description:** Total revenue minus the cost of goods sold.

- **Total Revenue:**
  - **Column:** Revenue
  - **Description:** Total income generated from sales.

- **Operating Income:**
  - **Column:** Operating Income
  - **Description:** Profit after deducting operating expenses.

- **Net Income:**
  - **Column:** Net Income
  - **Description:** Total earnings after all expenses.

- **Earnings Per Share (EPS):**
  - **Column:** EPS
  - **Description:** Net income divided by outstanding shares.

- **Price-to-Earnings Ratio (P/E):**
  - **Column:** priceEarningsRatio or PE ratio
  - **Description:** Market value per share divided by EPS.

- **Return on Equity (ROE):**
  - **Column:** returnOnEquity or ROE
  - **Description:** Net income divided by shareholders’ equity.

- **Debt-to-Equity Ratio:**
  - **Column:** debtEquityRatio
  - **Description:** Total liabilities divided by shareholders’ equity.

- **Current Ratio:**
  - **Column:** currentRatio
  - **Description:** Current assets divided by current liabilities.

- **Dividend Yield:**
  - **Column:** dividendYield
  - **Description:** Dividend per share divided by price per share.

#### Data Collection and Processing:
- **Data Retrieval:**
  - Data collected from annual reports and financial databases.
  - Compiled into CSV files for each year from 2014 to 2018.

- **Data Cleaning:**
  - Verified data accuracy by cross-referencing sources.
  - Handled missing data by imputation or exclusion.

- **Data Storage:**
  - Stored in CSV format, with each file representing a year.

### 3. Stock Price Data
- **Source:** Historical stock price data collected for each company.
- **Description:** Contains daily stock price information for each company, stored in individual .txt files per company.

#### Data Source:
You can access the insider transactions dataset from kaggle at the following link: [Huge Stock Market Dataset](https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs)

#### Data Structure and Variables:
Each .txt file represents a company's stock price data over a period.

**Key Variables and Corresponding Columns:**
- **Date:**
  - **Column:** Date
  - **Description:** Trading date.

- **Ticker:**
  - **Column:** Not present in the sample; needs to be added based on file name or additional column.

- **Open:**
  - **Column:** Open
  - **Description:** Opening price of the stock.

- **High:**
  - **Column:** High
  - **Description:** Highest price during the trading day.

- **Low:**
  - **Column:** Low
  - **Description:** Lowest price during the trading day.

- **Close:**
  - **Column:** Close
  - **Description:** Closing price of the stock.

- **Adjusted Close:**
  - **Column:** Not present; may need to be calculated or retrieved from additional sources.

- **Volume:**
  - **Column:** Volume
  - **Description:** Number of shares traded during the day.

#### Data Collection and Processing:
- **Data Retrieval:**
  - Downloaded historical price data for each company.

- **Data Cleaning:**
  - Added a Ticker column to each file, inferred from the file name.
  - Ensured dates are formatted consistently.
  - Removed Adjusted Close.

- **Data Storage:**
  - Stored as .txt or CSV files, one per company.

---

## Business Requirements
The project aims to address the following business requirements:

1. **Analyse the Impact of Insider Trading on Stock Prices**
   - Understand how insider buying and selling activities influence stock price movements.
   - Provide insights into the correlation between insider transactions and subsequent stock performance.

2. **Predict Stock Price Movements Using Combined Data**
   - Develop predictive models that integrate insider trading data with key financial indicators.
   - Enhance investment decision-making by forecasting future stock prices with improved accuracy.

3. **Identify Patterns in Insider Trading Activities**
   - Detect and analyse specific patterns or trends within insider trading data.
   - Uncover potential signals for stock performance based on insider behavior.

4. **Deliver an Interactive User Interface**
   - Create a user-friendly dashboard for investors and analysts.
   - Allow users to input parameters, view analyses, and receive real-time predictions.

---

## Hypotheses and How to Validate

1. **Hypothesis 1:** Significant insider buying activity positively impacts future stock prices.
   - **Validation Method:**
     - Merge insider buying records with subsequent stock price data.
     - Perform regression analysis to model the relationship between insider buying and stock returns.
     - Use statistical tests (e.g., t-tests) to determine the significance of the relationship.

2. **Hypothesis 2:** Specific patterns in insider trading can predict stock price volatility.
   - **Validation Method:**
     - Apply clustering algorithms to insider trading data to identify patterns.
     - Analyse stock price volatility following these patterns.
     - Conduct statistical tests (e.g., ANOVA) to assess if patterns are predictive of volatility changes.

3. **Hypothesis 3:** Combining insider trading data with financial indicators improves stock price prediction accuracy.
   - **Validation Method:**
     - Develop predictive models using insider trading data alone and compare them with models that include financial indicators.
     - Evaluate models using metrics like R², Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).
     - Use cross-validation to ensure robustness.

---

## The Rationale to Map the Business Requirements to the Data Visualisations and ML Tasks

- **Business Requirement 1:** Analyse the impact of insider trading on stock prices.
  - **Data Visualisations:**
    - Time-series plots showing stock prices with insider trading events marked.
    - Scatter plots of transaction values versus subsequent stock returns.
  - **ML Tasks:**
    - Regression analysis to quantify the impact.
    - Correlation studies to assess relationships.

- **Business Requirement 2:** Predict stock price movements using combined data.
  - **Data Visualisations:**
    - Actual vs. predicted stock prices.
    - Feature importance charts.
  - **ML Tasks:**
    - Develop regression models integrating both insider trading and financial data.
    - Hyperparameter tuning for model optimization.

- **Business Requirement 3:** Identify patterns in insider trading activities.
  - **Data Visualisations:**
    - Cluster plots showing different insider trading behaviors.
    - Heatmaps indicating frequency and timing of trades.
  - **ML Tasks:**
    - Clustering algorithms to detect patterns.
    - Feature engineering to create relevant variables.

- **Business Requirement 4:** Deliver an interactive user interface.
  - **Data Visualisations:**
    - Interactive dashboards with customisable charts.
  - **ML Tasks:**
    - Implement real-time predictions and user input handling.

---

## ML Business Case

- **Aim:** To enhance investment strategies by providing insights and predictions based on insider trading activities and financial indicators.
  
- **Learning Methods:**
  - **Supervised Learning:**
    - Regression models (e.g., Linear Regression, Random Forest Regressor).
  - **Unsupervised Learning:**
    - Clustering algorithms (e.g., K-Means) for pattern recognition.

- **Ideal Outcomes:**
  - Accurate stock price predictions.
  - Identification of significant insider trading patterns.

- **Success Metrics:**
  - **Regression Models:** High R² scores, low MSE/RMSE.
  - **Clustering:** Meaningful clusters with high silhouette scores.

- **Model Output and Relevance:**
  - **Predictive Insights:** Assist investors in making informed decisions.
  - **Pattern Detection:** Provide signals that could indicate market movements.

- **Heuristics and Training Data:**
  - Utilise historical data spanning multiple years.
  - Incorporate lag variables for delayed effects.
  - Normalise and scale features for consistency.

---

## Dashboard Design

The dashboard will consist of seven main pages:

1. **Project Summary**
   - **Content:**
     - Introduction to the project, objectives, and scope.
     - Explanation of key terms and datasets.
   - **Widgets:**
     - Text blocks.
     - Images or icons representing key concepts.

2. **Data Analysis**
   - **Content:**
     - Results of exploratory data analysis.
     - Visualisations of insider trading and stock price data.
   - **Widgets:**
     - Interactive charts (time-series plots, scatter plots).
     - Filters for date ranges, companies, and insider types.

3. **User Input Interface**
   - **Content:**
     - Forms for user inputs (e.g., company ticker, date range).
   - **Widgets:**
     - Text input fields.
     - Dropdown menus.
     - Sliders and checkboxes.

4. **Project Hypotheses**
   - **Content:**
     - Presentation of hypotheses and validation methods.
     - Visual evidence supporting or refuting hypotheses.
   - **Widgets:**
     - Tabs or accordions to navigate between hypotheses.
     - Interactive graphs.

5. **Model Performance**
   - **Content:**
     - Performance metrics and evaluation results.
     - Comparison of different models.
   - **Widgets:**
     - Charts showing model performance.
     - Tables summarizing metrics.

6. **Insider Trading Patterns**
   - **Content:**
     - Analysis of patterns found in insider trading data.
     - Implications of these patterns.
   - **Widgets:**
     - Cluster diagrams.
     - Heatmaps.
     - Interactive pattern selectors.

7. **Conclusions and Recommendations**
   - **Content:**
     - Summarised findings.
     - Recommendations for users.
     - Potential actions based on insights.
   - **Widgets:**
     - Text blocks.
     - Downloadable reports or summaries.

---

## Unfixed Bugs
At this stage, there are no known unfixed bugs. All features are expected to function as intended. Any future bugs discovered during development, especially those arising from limitations in frameworks or technologies used, will be documented here along with explanations.

---

## Deployment
**Heroku**  
The App live link is: [https://insider-trading-analysis.herokuapp.com/](https://insider-trading-analysis.herokuapp.com/) (Link will be updated upon deployment.)  
**Python Version:** Specified in runtime.txt (e.g., python-3.8.10)  

**Deployment Steps:**
1. **Log in to Heroku and Create an App**
   - Navigate to the Heroku dashboard and click "New" > "Create new app."
   - Enter a unique app name and select the appropriate region.
   
2. **Set Up Deployment Method**
   - Go to the "Deploy" tab.
   - Under "Deployment method," select "GitHub."
   - Connect your GitHub account if not already connected.

3. **Connect to GitHub Repository**
   - Search for your repository name and click "Connect."

4. **Deploy the Branch**
   - Select the branch you want to deploy (e.g., main).
   - Click "Deploy Branch."
   - Monitor the build logs to ensure successful deployment.

5. **Open the App**
   - Once deployment is complete, click "Open App" to access the application.

6. **Handle Slug Size Issues**
   - If the slug size exceeds Heroku's limits:
     - Add large files not required for the app to the .slugignore file.
     - Optimize dependencies listed in requirements.txt by removing unnecessary packages.

---

## Main Data Analysis and Machine Learning Libraries

- **Pandas**
  - Used for data manipulation and analysis.
  - Example: Reading CSV and TSV files, merging datasets, handling missing values.

- **NumPy**
  - Supports numerical operations on large, multi-dimensional arrays.
  - Example: Calculating statistical measures like mean and standard deviation.

- **Scikit-Learn**
  - Provides tools for data mining and data analysis.
  - Example:
    - Implementing regression models (Linear Regression, Random Forest Regressor).
    - Clustering algorithms (K-Means).
    - Model evaluation metrics (R² score, MSE).

- **Matplotlib and Seaborn**
  - Libraries for creating static, animated, and interactive visualisations.
  - Example: Plotting time-series data, creating scatter plots, heatmaps.

- **Plotly**
  - Enables creation of interactive visualisations.
  - Example: Building interactive charts in the Streamlit dashboard.

- **Streamlit**
  - An open-source app framework for machine learning and data science.
  - Example: Developing the user interface and interactive elements of the dashboard.

- **Requests**
  - Used for making HTTP requests to access APIs.
  - Example: Fetching data from financial APIs if needed.

- **Statsmodels**
  - Provides classes and functions for the estimation of statistical models.
  - Example: Conducting hypothesis testing, performing time-series analysis.

---

## Credits

**Content**
- **Data Sources:**
  - Insider Trading Data: Sourced from the SEC EDGAR Database.
  - Financial Data: Collected from company financial statements and public records(kaggle).
  - Stock Price Data: Retrieved from financial data providers(kaggle).

- **Documentation and Tutorials:**
  - Machine learning implementations referenced from the Scikit-Learn documentation.
  - Streamlit app development guided by the Streamlit documentation.

- **Code Snippets:**
  - Data processing methods inspired by examples from the Pandas and NumPy documentation.
  - Visualisation techniques adapted from Matplotlib, Seaborn, and Plotly examples.

**Media**
- **Icons and Images:**
  - Icons used in the dashboard are from Font Awesome.
  - Stock images used are sourced from Unsplash and Pexels, which provide free-to-use images.

---

## Acknowledgements
- **Instructors and Mentors:**
  - Special thanks to the educators and mentors who provided guidance throughout the development of this project.
  
- **Open-Source Community:**
  - Gratitude to the developers and contributors of the open-source libraries used in this project.
  
- **Data Providers:**
  - Appreciation for the platforms (Kaggle & SEC) offering access to financial and market data, enabling the creation of this application.

---

## Commit Messages Prefixes
1. **feat:** A new feature or functionality.
   - Example: `feat: add data preprocessing step`
   
2. **fix:** A bug fix or correction.
   - Example: `fix: resolve data loading error`
   
3. **docs:** Documentation changes or updates.
   - Example: `docs: update README with usage instructions`
   
4. **style:** Changes that do not affect the logic (e.g., formatting).
   - Example: `style: reformat code for readability`
   
5. **refactor:** Code changes that neither fix a bug nor add a feature but improve structure.
   - Example: `refactor: simplify model training code`
   
6. **test:** Adding or modifying tests.
   - Example: `test: add unit tests for data validation`
   
7. **chore:** Routine tasks that are not related to code (e.g., updates to dependencies).
   - Example: `chore: update project dependencies`

8. **perf:** A performance improvement.
   - Example: `perf: optimize data loading speed`
   
9. **ci:** Changes related to continuous integration.
   - Example: `ci: update CI configuration for testing`
   
10. **build:** Changes that affect the build system or external dependencies.
    - Example: `build: update Dockerfile for new libraries`
    
11. **revert:** Reverting a previous commit.
    - Example: `revert: undo changes from commit 1234567`
    
12. **config:** Changes to configuration files.
    - Example: `config: update model hyperparameters`
    
13. **init:** Initial commit for setting up the project.
    - Example: `init: create initial project structure`
    
14. **example:** Adding examples or demo files.
    - Example: `example: add demo notebook for model usage`

This README provides a comprehensive overview of the Insider Trading Analysis and Prediction Platform, integrating detailed information about datasets, variables, and data structures based on the collected data. It serves as a guiding document for stakeholders and contributors involved in the project.

