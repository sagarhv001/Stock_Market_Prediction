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

## 🏆 **How to Run**

### 1. **Run the Scraper & Sentiment Analysis**:
```python
# Initialize Reddit connection & fetch comments
from scraper import fetch_reddit_data
fetch_reddit_data()
```

---

### 2. **Fetch & Process Stock Data**:
```python
from stock_fetcher import fetch_stock_history
fetch_stock_history(ticker="ZOMATO.NS")
```

---

### 3. **Train the Prediction Model**:
```python
from model import train_cnn_model
train_cnn_model()
```

---

## 📊 **Model Training & Evaluation**

The CNN model is trained on sentiment analysis scores and historical stock data:

### Steps:
1. Preprocess and prepare data (`merged_data_all.csv`).
2. Split data into train/test sets.
3. Train CNN:
   - `Conv1D`, `Dropout`, `Dense` layers.
4. Evaluate model accuracy with metrics like:
   - **Precision**
   - **Recall**
   - **F1-Score**

---

## 🔮 **Outputs & Expected Results**

### CSV Outputs:
1. `comment_analysis_<ticker>.csv`: Sentiment data extracted and analyzed.
2. `stockhistory_<ticker>.csv`: Historical stock price data.
3. `merged_data_<ticker>.csv`: Combined sentiment & stock data.
4. `merged_data_all.csv`: Consolidated data of all analyzed tickers.

### Model Outputs:
- Training history graphs (loss vs. accuracy).
- Evaluation results with confusion matrices.

---

## 🚀 **Future Improvements**

1. **Real-Time Scraping**: Integrate with periodic data scrapers for live data analysis.
2. **Add Twitter or Telegram sentiment analysis**.
3. **Explore other models**: Compare CNN with other models like LSTMs or Gradient Boosting.

---

## 🐛 **Debugging Tips**

### Missing Data?
- Double-check ticker symbols (e.g., `ZOMATO.NS`) and ensure correct stock exchange codes.

### Model Not Training Correctly?
- Ensure data is preprocessed properly using StandardScaler normalization.

---

## 🏆 **Contribute**

We welcome contributions! If you have improvements, fix a bug, or can add documentation, feel free to submit a pull request.

---

📧 For inquiries or collaboration: **your.email@example.com**

🔗 Repository maintained by: [Your GitHub username]

---

Thank you for checking out this project! 🚀
