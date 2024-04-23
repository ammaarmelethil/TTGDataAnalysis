# This script functions as a comprehensive data processing pipeline for financial trade data analysis. It starts by 
# connecting to a SQL database to fetch and load trade data into a DataFrame. Data preprocessing includes converting
# specific columns to datetime formats and extracting time and date components, which are used to determine market 
# sessions, including handling sessions that cross midnight. The script also scrapes the web to download an Excel file
# containing sentiment data, merges this with the trade data based on date, and then exports the merged data to an Excel 
# file for review. Finally, it uploads the merged dataset to a SQL database using SQLAlchemy, facilitating efficient data 
# storage and retrieval for further analysis or reporting.


# Imports
import os
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from sqlalchemy import create_engine
import urllib
import pyodbc

# ============================
# Database Connection Setup
# ============================

# Database connection parameters
server = '192.168.50.221'
database = 'CapUOC_DataAnalysis'
username = 'USERNAME'
password = 'PASSWORD'  

# Establish a connection to the SQL database
connection_string = f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'
connection = pyodbc.connect(connection_string)

# Define the SQL query to fetch data
sql_query = "SELECT * FROM dbo.FXandTrades2"

# Load data into DataFrame
df1_cleaned = pd.read_sql(sql_query, connection)

# Close the database connection
connection.close()

# ============================
# Data Preprocessing
# ============================

# Ensure the 'When' column is in datetime format and extract relevant parts
df1_cleaned['When'] = pd.to_datetime(df1_cleaned['When'])
df1_cleaned['Time_Only'] = df1_cleaned['When'].dt.time
df1_cleaned['date_only'] = df1_cleaned['When'].dt.date

# ============================
# Determine Market Sessions
# ============================

# Define market open and close times in MST
market_times = {
    'LOND': ('01:00:00', '10:00:00'),
    'NY': ('06:00:00', '15:00:00'),
    'SYD': ('15:00:00', '23:59:59'),  # Ends next day
    'TOK': ('18:00:00', '03:00:00')   # Crosses midnight
}

# Function to check if a time falls within a given market session, handling sessions that cross midnight
def in_market_session(row, open_time, close_time):
    if open_time < close_time:
        return open_time <= row['Time_Only'] <= close_time
    return row['Time_Only'] >= open_time or row['Time_Only'] <= close_time

# Apply the function to create columns indicating if time falls within the market session
for market, (open_time, close_time) in market_times.items():
    open_time = pd.to_datetime(open_time, format='%H:%M:%S').time()
    close_time = pd.to_datetime(close_time, format='%H:%M:%S').time()
    df1_cleaned[market] = df1_cleaned.apply(in_market_session, axis=1, args=(open_time, close_time))

# ============================
# Web Scraping
# ============================

# URL of the webpage to scrape the Excel file link
url = 'https://www.frbsf.org/research-and-insights/data-and-indicators/daily-news-sentiment-index/'

# Make a request to the webpage and ensure the request was successful
response = requests.get(url)
response.raise_for_status()

# Parse the HTML content of the page to find the specific Excel file link
soup = BeautifulSoup(response.text, 'html.parser')
link_tag = soup.find('a', href=True, text='Daily News Sentiment data')
excel_file_link = link_tag['href'] if link_tag else None

# If the link is found, download and save the Excel file
if excel_file_link:
    if excel_file_link.startswith('/'):
        excel_file_link = f'https://www.frbsf.org{excel_file_link}'
    
    excel_response = requests.get(excel_file_link)
    excel_response.raise_for_status()
    
    with open('daily_news_sentiment_data.xlsx', 'wb') as f:
        f.write(excel_response.content)
        
    print("Excel file downloaded successfully.")
else:
    print("Excel file link not found.")

# Load and display the first few rows of the sentiment data to confirm it's loaded correctly
df_sentiment = pd.read_excel("daily_news_sentiment_data.xlsx", sheet_name='Data')
print(df_sentiment.head())

# ============================
# Data Merging and Export
# ============================

# Convert 'date_only' and 'date' to datetime in both dataframes
df1_cleaned['date_only'] = pd.to_datetime(df1_cleaned['date_only'])
df_sentiment['date'] = pd.to_datetime(df_sentiment['date'])

# Perform the merge on dates and print the resulting DataFrame
df_merged_sentiment = pd.merge(df1_cleaned, df_sentiment, left_on='date_only', right_on='date', how='inner')
print(df_merged_sentiment.head())
print(len(df_merged_sentiment))

# Export the cleaned DataFrame to an Excel file and open it
file_path = "Code SQL DF1_CLEANED.xlsx"
df1_cleaned.to_excel(file_path, index=False)
os.startfile(file_path)

# ============================
# Database Upload
# ============================

# Create the connection URL and engine for SQLAlchemy
connection_string_alchemy = f"mssql+pyodbc://{username}:{urllib.parse.quote_plus(password)}@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server"
engine = create_engine(connection_string_alchemy)

# Save the merged DataFrame to SQL database
df_merged_sentiment.to_sql('SentimentandTrades', engine, if_exists='replace', index=False)