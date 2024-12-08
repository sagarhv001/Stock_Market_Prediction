# GitHub Repository: Stock Market Sentiment Analysis and Prediction Model

Below is the comprehensive and well-documented **README.md** file tailored for a repository containing Reddit scraping, sentiment analysis, and stock prediction steps.

---

# 📊 **Stock Market Sentiment Analysis & Prediction**

## 🚀 **Overview**

This repository provides tools for analyzing **Reddit sentiment** related to specific stock tickers and predicting stock price movements using **Convolutional Neural Networks (CNN)**. 

The analysis involves:
- **Reddit Data Scraping**: Extracting posts/comments related to stock discussions.
- **Sentiment Analysis**: Using tools like **VADER** and **TextBlob** for sentiment scoring.
- **Stock Price Data Integration**: Using **Yahoo Finance** to fetch historical stock price data.
- **Model Training**: Predicting stock price movement with a CNN trained on sentiment & stock market data.

---

## 🛠️ **Key Features**

1. **Scraper for Reddit Sentiment**:
   - Scrapes data from a specific subreddit (e.g., `IndianStockMarket`) for mentions of a given stock ticker.
   - Analyzes sentiment trends using **VADER** and **TextBlob**.

2. **Stock Data Fetching**:
   - Fetches historical stock price data from **Yahoo Finance**.

3. **Sentiment & Stock Data Integration**:
   - Merges Reddit sentiment data with historical stock data for comprehensive analysis.

4. **Stock Movement Prediction**:
   - Trains a **Convolutional Neural Network (CNN)** to predict stock movements using sentiment and stock trends.

---

## 📜 **Table of Contents**

1. [📋 Dependencies](#dependencies)  
2. [🛠️ Installation](#installation)  
3. [🔧 Setup Instructions](#setup-instructions)  
4. 🏆 [How to Run](#how-to-run)  
5. 📊 [Model Training & Evaluation](#model-training--evaluation)  
6. 🔮 [Outputs & Expected Results](#outputs--expected-results)  
7. 🚀 [Future Improvements](#future-improvements)  
8. 🐛 [Debugging Tips](#debugging-tips)

---

## 📋 **Dependencies**

Below are the libraries needed for running this project:

Install them using:

```bash
pip install asyncpraw praw
pip install --upgrade pip
pip install yfinance --no-dependencies
pip install multitasking --no-dependencies
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn nltk textblob
```

### 📌 **Key Libraries**

1. **Scraping Data from Reddit**:
   - `asyncpraw`, `praw`

2. **Fetching Stock Data**:
   - `yfinance`

3. **Sentiment Analysis**:
   - `nltk`, `textblob`

4. **Data Processing**:
   - `pandas`, `numpy`

5. **Model Training**:
   - `tensorflow`

6. **Visualization**:
   - `matplotlib`, `seaborn`

---

## 🛠️ **Installation**

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/stock-market-sentiment.git
   cd stock-market-sentiment
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🔧 **Setup Instructions**

### 1. **Configure Reddit API Credentials**:
To use the Reddit scraper:
1. Visit [Reddit's app creation page](https://www.reddit.com/prefs/apps).
2. Create an application of type **Script** and save the credentials:
   - `client_id`
   - `client_secret`
   - `user_agent`
3. Store them securely in a `.env` file or directly in the script as variables.

### Example `.env`:
```env
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=your_user_agent
```

Load them in your script using:
```python
import os
from dotenv import load_dotenv
load_dotenv()

client_id = os.getenv("REDDIT_CLIENT_ID")
client_secret = os.getenv("REDDIT_CLIENT_SECRET")
user_agent = os.getenv("REDDIT_USER_AGENT")
```

---
##Documentation on each jupyter notebook
Telegram/Reddit.ipynb
Overview
This script:
Scrapes posts and comments from a specified subreddit (e.g., IndianStockMarket) using Reddit's API.
Analyzes the sentiment of post titles and comments using VADER and TextBlob sentiment analysis tools.
Identifies mentions of a specific stock ticker in posts/comments and counts them.
Fetches historical stock price data for the specified ticker from Yahoo Finance and saves it to a CSV.

Dependencies
Libraries
Install the required libraries before running the code:
pip install asyncpraw praw
pip install --upgrade pip
pip install yfinance --no-dependencies
pip install multitasking --no-dependencies

Python Libraries Used
Reddit API:
praw: For connecting to and scraping data from Reddit.
asyncpraw: Asynchronous version of praw for efficiency.
Finance API:
yfinance: To fetch historical stock price data.
Data Manipulation and Visualization:
pandas: For data cleaning and DataFrame manipulation.
numpy: For numerical operations.
Sentiment Analysis:
nltk: Includes VADER sentiment analyzer.
textblob: For polarity-based sentiment scoring.
Others:
datetime: To handle date formats.
time: For handling time delays in API requests.

Setting Up
1. Configure Reddit API Credentials
Before running the script, set up Reddit API credentials:
Visit the Reddit Apps page.
Create an app (type: script) and note down:
client_id
client_secret
user_agent
Store these credentials in the Colab or your local environment using:
 from google.colab import userdata
c_id = userdata.get('client_id')
c_secret = userdata.get('client_secret')
u_agent = userdata.get('user_agent')


2. Set the Stock Ticker
Define the stock ticker symbol you want to analyze:
STS = "ZOMATO.NS"  # Example: ZOMATO for NSE

3. Run in Colab or Jupyter Notebook
To execute:
Upload the script to Google Colab or run it in your local Jupyter environment.

Code Walkthrough
1. Sentiment Analysis on Reddit
Purpose
Analyze sentiment in posts and comments mentioning a specific stock ticker.
Steps
Initialize Reddit Connection:

 reddit = praw.Reddit(client_id=c_id, client_secret=c_secret, user_agent=u_agent)
subreddit = reddit.subreddit("IndianStockMarket")


Fetch Posts: Fetch all "top" posts from the specified subreddit:

 top_posts = reddit.subreddit("IndianStockMarket").top(limit=None)


Sentiment Analysis Functions:


TextBlob: Analyzes polarity (positive, neutral, or negative) based on text sentiment.
 def text_blob_sentiment(review, sub_entries_textblob):
    analysis = TextBlob(review)
    if analysis.sentiment.polarity > 0:
        sub_entries_textblob['positive'] += 1
    elif analysis.sentiment.polarity < 0:
        sub_entries_textblob['negative'] += 1
    else:
        sub_entries_textblob['neutral'] += 1


VADER: Determines sentiment using VADER's lexicon-based approach:
 def nltk_sentiment(review, sub_entries_nltk):
    vs = sia.polarity_scores(review)
    if vs['pos'] > vs['neg']:
        sub_entries_nltk['positive'] += 1
    elif vs['neg'] > vs['pos']:
        sub_entries_nltk['negative'] += 1
    else:
        sub_entries_nltk['neutral'] += 1


Count Stock Mentions: Searches for the stock ticker symbol (STS) in comments/posts:

 if comment.body.find(STS) != -1:
    STSC += 1


Save Results: Stores sentiment data and stock mentions in a CSV:

 commentDF.to_csv(f'comment_analysis_{STS}.csv', index=False, header=True)



2. Fetch Stock Price Data from Yahoo Finance
Purpose
Retrieve historical stock data for the specified ticker.
Steps
Fetch Data: Using yfinance:

 selectedTicker = yf.Ticker(ticker)
hist = selectedTicker.history(period="max")
Select the company you want the historic data of and enter its symbol here.
Do this for all the companies you need, if you want the NIFTY 50 index, use its symbol.
Save Data: Saves stock history as a CSV:

 hist.to_csv(f'stockhistory_{ticker}.csv', index=True, header=True)


How to Run the Script
Sentiment Analysis:
	
Run all the cells using Ctrl + F9 first to prime the sentiment analysing cells and install all the dependencies
Run the main() function:
if __name__ == '__main__':
    main()


Output:
Prints sentiment counts for each post.
Saves sentiment and ticker mention data in comment_analysis_<ticker>.csv.
Stock Price Analysis:


Call fetch_and_save_stock_history_with_pcr():
fetch_and_save_stock_history_with_pcr(STS)


Output:
Saves historical stock price data in stockhistory_<ticker>.csv.

Outputs
Sentiment Analysis CSV: Contains:


Title: Post title
Ticker: Stock ticker analyzed
Date: Post date
NumberOfTickerMentions: Count of mentions in comments
Sentiment values: VADER Neg, VADER Pos, TextBlob Negative, etc.
Stock History CSV: Contains:


Date, Open, Close, High, Low, Volume

Future Improvements
Real-Time Scraping: Automate periodic scraping of Reddit posts and Yahoo Finance data.
Additional Sentiment Sources: Include Twitter, Telegram, or Discord sentiment.
Advanced Analytics:
Correlate stock price movements with sentiment trends.
Use predictive models for price forecasting.
Collect PCR along with other stock data as PCR is more accurate metric to predict stock movement.

