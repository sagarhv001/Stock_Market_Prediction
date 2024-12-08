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
## Documentation on each jupyter notebook
### Telegram/Reddit.ipynb
### Overview
### This script:
Scrapes posts and comments from a specified subreddit (e.g., IndianStockMarket) using Reddit's API.
Analyzes the sentiment of post titles and comments using VADER and TextBlob sentiment analysis tools.
Identifies mentions of a specific stock ticker in posts/comments and counts them.
Fetches historical stock price data for the specified ticker from Yahoo Finance and saves it to a CSV.

### Dependencies
### Libraries
Install the required libraries before running the code:
pip install asyncpraw praw
pip install --upgrade pip
pip install yfinance --no-dependencies
pip install multitasking --no-dependencies

### Python Libraries Used
### Reddit API:
praw: For connecting to and scraping data from Reddit.
asyncpraw: Asynchronous version of praw for efficiency.
### Finance API:
yfinance: To fetch historical stock price data.
### Data Manipulation and Visualization:
pandas: For data cleaning and DataFrame manipulation.
numpy: For numerical operations.
### Sentiment Analysis:
nltk: Includes VADER sentiment analyzer.
textblob: For polarity-based sentiment scoring.
### Others:
datetime: To handle date formats.
time: For handling time delays in API requests.

### Setting Up
#### 1. Configure Reddit API Credentials
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


#### 2. Set the Stock Ticker
Define the stock ticker symbol you want to analyze:
STS = "ZOMATO.NS"  # Example: ZOMATO for NSE

#### 3. Run in Colab or Jupyter Notebook
To execute:
Upload the script to Google Colab or run it in your local Jupyter environment.

### Code Walkthrough
#### 1. Sentiment Analysis on Reddit
Purpose
Analyze sentiment in posts and comments mentioning a specific stock ticker.
### Steps
### Initialize Reddit Connection:

 reddit = praw.Reddit(client_id=c_id, client_secret=c_secret, user_agent=u_agent)
subreddit = reddit.subreddit("IndianStockMarket")


### Fetch Posts: Fetch all "top" posts from the specified subreddit:

 top_posts = reddit.subreddit("IndianStockMarket").top(limit=None)


### Sentiment Analysis Functions:


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


### Count Stock Mentions: Searches for the stock ticker symbol (STS) in comments/posts:

 if comment.body.find(STS) != -1:
    STSC += 1


### Save Results: Stores sentiment data and stock mentions in a CSV:

 commentDF.to_csv(f'comment_analysis_{STS}.csv', index=False, header=True)



### 2. Fetch Stock Price Data from Yahoo Finance
### Purpose
Retrieve historical stock data for the specified ticker.
### Steps
#### Fetch Data: Using yfinance:

 selectedTicker = yf.Ticker(ticker)
hist = selectedTicker.history(period="max")
Select the company you want the historic data of and enter its symbol here.
Do this for all the companies you need, if you want the NIFTY 50 index, use its symbol.

#### Save Data: Saves stock history as a CSV:

 hist.to_csv(f'stockhistory_{ticker}.csv', index=True, header=True)


How to Run the Script
### Sentiment Analysis:
	
Run all the cells using Ctrl + F9 first to prime the sentiment analysing cells and install all the dependencies
Run the main() function:
if __name__ == '__main__':
    main()


### Output:
Prints sentiment counts for each post.
Saves sentiment and ticker mention data in comment_analysis_<ticker>.csv.

### Stock Price Analysis:


Call fetch_and_save_stock_history_with_pcr():
fetch_and_save_stock_history_with_pcr(STS)


### Output:
Saves historical stock price data in stockhistory_<ticker>.csv.

### Outputs
### Sentiment Analysis CSV: Contains:


Title: Post title
Ticker: Stock ticker analyzed
Date: Post date
NumberOfTickerMentions: Count of mentions in comments
Sentiment values: VADER Neg, VADER Pos, TextBlob Negative, etc.

### Stock History CSV: Contains:


Date, Open, Close, High, Low, Volume

### Future Improvements
Real-Time Scraping: Automate periodic scraping of Reddit posts and Yahoo Finance data.
Additional Sentiment Sources: Include Twitter, Telegram, or Discord sentiment.
Advanced Analytics:
Correlate stock price movements with sentiment trends.
Use predictive models for price forecasting.
Collect PCR along with other stock data as PCR is more accurate metric to predict stock movement.

## Reddit_analysis.ipynb
Reddit Sentiment Analysis and Stock Market Correlation
This script analyzes Reddit comments for sentiment trends and correlates them with stock price movements. It involves calculating sentiment metrics, fetching stock prices, merging datasets, and producing combined insights.

### 1. Prerequisites
### Dependencies
Install the required Python libraries:
pip install pandas numpy yfinance glob2

### Key Libraries and Their Functions:
pandas: Data manipulation and cleaning.
numpy: Numerical computations.
yfinance: Fetching stock data from Yahoo Finance.
glob: File pattern matching.
re: Regular expressions for text analysis.
datetime: Handling dates.
### Input Files
Reddit Sentiment Data (comment_analysis_Indian_Stock_Market_NSE.csv):


Includes sentiment scores (Vader Pos, Vader Neg, TextBlob Positive, etc.) and mentions of stock tickers.
Stock Price Data (stockhistory_<symbol>.csv):


Contains daily stock price details (Open, High, Low, Close, Volume).
General Data (general.csv, obtained as an intermediate here):


Broader market comments for trend analysis.

### 2. Script Overview
Step 1: Sentiment Data Processing
Compute Rolling Averages


Rolling averages of sentiment scores are calculated over a 5-day window.
 New Columns:
Rolling_Vader_Pos, Rolling_Vader_Neg
Rolling_TextBlob_Pos, Rolling_TextBlob_Neg.
Weighted Sentiments by Mentions


Sentiment scores are multiplied by the number of ticker mentions.
 New Columns:
Weighted_Vader_Pos, Weighted_TextBlob_Pos, etc.
Normalize Sentiments


Weighted scores are normalized using the total mentions in the window.
 New Columns:
Normalized_Weighted_Vader_Pos, Normalized_Weighted_TextBlob_Pos.
Combined Sentiment Scores


Aggregates rolling and normalized sentiments into a single metric.
 New Columns:
Combined_Sentiment_Pos, Combined_Sentiment_Neg.
Step 2: Stock Data Integration
Filter Relevant Comments


Extract comments mentioning specific stock tickers using regex (filter_comments_by_ticker function).
Merge Sentiment and Stock Data


Combines Reddit sentiment and stock price data by matching the Date field.
Saves the merged data for each stock ticker to merged_data_<TICKER>.csv.

### Step 3: General Market Sentiment
Analyzes sentiment trends across the NIFTY 50 index using general Reddit comments.
### Step 4: Consolidating All Data
Merges all individual ticker-based files into merged_data_all.csv, sorted by Date.

### 3. Functions
### Data Preparation
Rolling Averages and Weighted Sentiments

 commentDF['Rolling_Vader_Pos'] = commentDF['Vader Pos'].rolling(window=5).mean()


Normalization

 commentDF['Normalized_Weighted_Vader_Pos'] = commentDF['Weighted_Vader_Pos'] / (commentDF['NumberOfTickerMentions'] + 1e-5)


### Filtering Comments
Filters comments mentioning a specific stock ticker:
def filter_comments_by_ticker(comment_df, ticker_keywords):
    pattern = rf"\b(?:{'|'.join(ticker_keywords)})\b"
    return comment_df[comment_df['Title'].str.contains(pattern, case=False, na=False)]

### Merging Data
Combines sentiment and stock price data:
def merge_sentiment_and_stock_data(comment_df, stock_df):
    final_df = pd.merge(comment_df, stock_df, on='Date', how='inner')
    return final_df

### Merging All Data
Combines all merged_data_<TICKER>.csv files into a single file:
def merge_all_csvs(output_file, sort_by="Date"):
    combined_df = pd.concat([pd.read_csv(file) for file in glob.glob("merged_data_*.csv")])
    combined_df.sort_values(by=sort_by).drop_duplicates().to_csv(output_file, index=False)


### 4. How to Use
### Step 1: Prepare Data Files
Place the following files in the working directory:
comment_analysis_Indian_Stock_Market_NSE.csv
stockhistory_<Name>.csv,(output from previous notebook)

### Step 2: Execute the Script
Run the script in a Python environment (e.g., Jupyter Notebook or Google Colab).
Run the code for different Company’s Historic Data(stockhistory_<Name>.csv,(output from previous notebook))
This will give merged_data_<ticker>.csv files for that company.
Now, we know that all Titles refer to a particular stock exchange(NSE), and the ones referring to any particular company are dealt with, we are left with general comments, which will be saved in general.csv (Intermediate output obtained from this code)
Processed data will be saved as:
comment_analysis_with_sentiments.csv
merged_data_<TICKER>.csv
merged_data_all.csv.

### Step 3: Adjust Parameters
Update rolling_window to change sentiment calculation window.
Modify ticker_keywords to analyze a different stock.

### 5. Outputs
Individual Files
comment_analysis_with_sentiments.csv
 Contains rolling averages, weighted sentiments, and combined scores.


merged_data_<TICKER>.csv
 Merged dataset for individual stocks.


Consolidated File
merged_data_all.csv
 Aggregates all ticker-specific data into one sorted file.

### 6. Debugging Tips
Missing Stock Data:
 Verify the ticker format (e.g., ZOMATO.NS).


Invalid Dates:
 Ensure date formats in input files are consistent.


Empty Outputs:
 Check if input data contains entries for the specified ticker or period.


## model.ibynb
Stock Price Movement Prediction Using CNN: Documentation
This documentation provides a comprehensive guide on the implementation, including the required dependencies, dataset preparation, model architecture, training process, evaluation metrics, and model saving.

### 1. Dependencies
Install the required libraries using the following commands:
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn

pandas: Data manipulation and analysis.
numpy: Numerical operations and array handling.
scikit-learn: Preprocessing, train-test splitting, and evaluation metrics.
tensorflow: Building and training the CNN model.
matplotlib and seaborn: Visualization of evaluation results (e.g., confusion matrix).

### 2. Code Overview
### 2.1 Dataset Preparation
Input Dataset: The dataset (merged_data_all.csv) includes stock features and sentiment analysis scores. The goal is to predict the binary target variable Price_Movement:


1: Stock price increased.
0: Stock price decreased.
Feature and Target Selection:


Features (X): All numerical columns, excluding Price_Movement.
Target (y): Binary column Price_Movement.
### Data Preprocessing:


Missing values in features are replaced with the mean.
Features are scaled to a standard normal distribution using StandardScaler.
data['Price_Change'] = data['Close'] - data['Open']
data['Price_Movement'] = np.where(data['Price_Change'] > 0, 1, 0)

X = data.select_dtypes(include=[np.number]).drop(columns=['Price_Movement'])
X = X.fillna(X.mean())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


### 2.2 CNN-Specific Data Preparation
Reshape the dataset into 3D format required for CNN:
 X_cnn = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])



### 2.3 Train-Test Split
Split the data into training and testing sets:
 X_train, X_test, y_train, y_test = train_test_split(X_cnn, y, test_size=0.2, random_state=42)



### 2.4 CNN Model Architecture
Layers:
Conv1D: Convolutional layer to detect patterns across features.
Dropout: Regularization layer to reduce overfitting.
Flatten: Converts the 3D tensor into a 1D tensor for the dense layers.
Dense: Fully connected layers for decision-making.
Sigmoid Activation: Outputs probabilities for binary classification.
model = Sequential([
    Conv1D(64, kernel_size=1, activation='relu', input_shape=(1, X_cnn.shape[2])),
    Dropout(0.2),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


### 2.5 Model Training
EarlyStopping: Stops training when the validation loss does not improve for 5 epochs.
Train the model with 50 epochs and a batch size of 32:
 early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    epochs=50, batch_size=32, callbacks=[early_stop], verbose=1)



### 2.6 Model Evaluation
Evaluate the model's performance on the test data:

 test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.2f}")


Additional Metrics: Evaluate the model using metrics like precision, recall, F1-score, and confusion matrix:

 from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

y_pred_probs = model.predict(X_test).flatten()
y_pred = (y_pred_probs > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)





### 2.7 Saving the Model
Save the trained CNN model for reuse:
 model.save("cnn_model.keras")
print("CNN model saved as 'cnn_model.keras'")



### 3. Expected Outputs
Training Progress:
Epoch-wise loss and accuracy for both training and validation data.
Evaluation Metrics:
Accuracy, Precision, Recall, F1-score, and Confusion Matrix.
Saved Model:
The trained CNN model is saved as cnn_model.keras.

### 4. Improvements and Extensions
Data Augmentation:
Include more features, such as moving averages or volatility.
Integrate advanced sentiment embeddings (e.g., BERT, GPT embeddings).
Hyperparameter Tuning:
Experiment with different kernel sizes, dropout rates, and layer configurations.
Model Comparison:
Compare the CNN with other models like Random Forest, Gradient Boosting, or LSTM.
Deployment:
Use a Flask or FastAPI application to deploy the model for real-time predictions.




