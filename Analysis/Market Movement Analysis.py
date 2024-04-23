"""
The scrip below works by using the PYODBC library to connect to the SQL sever, importing a table as a dataframe, and then calculating the sum of the absoloute 
changes in price (or exchange rates depending on what table is uploaded) between trades. This counter resets every time a trade is made. It then creates 
a new dataframe showing only the trades and the corresponding sum of how much the market moved leading up to the trade, and finally uploads the new dataframe to SQL.

This code is currently set to run on the USD.CAD table, but can run on the SPX data with some slight modifications outlined in the code below. 
All that changes is what table is selected (line 34 and 44), if the price changes are calculated in pips (line 66), and the name of the new table that is uploaded
to SQL (line 96).
"""


# Imports
import pyodbc
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import urllib

# ============================
# Database Connection Setup
# ============================

# Connection parameters for SQL database
server = '192.168.50.221'
database = 'CapUOC_DataAnalysis'
username = 'USERNAME'
password = 'PASSWORD'  # Your Password

# Connect to SQL Server database
connection_string = f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'
connection = pyodbc.connect(connection_string)
cursor = connection.cursor()

# Fetch column names for the specified table in the database
cursor.execute("SELECT * FROM dbo.MarketMovementUSD_CAD")
columns = [column[0] for column in cursor.description]
print("Columns:", columns)
cursor.close()

# ============================
# Data Acquisition
# ============================

# Define SQL query to fetch market movement data
sql_query1 = "SELECT * FROM dbo.MarketMovementUSD_CAD"  #change to select either USD_CAD, SPX, Or any other tables
df1 = pd.read_sql(sql_query1, connection)
connection.close()

# ============================
# Data Processing
# ============================

# Ensure 'BarDateTime' is in datetime format and sort the DataFrame by this column
df1['BarDateTime'] = pd.to_datetime(df1['BarDateTime'])
df1 = df1.sort_values(by='BarDateTime', ascending=True)

# Convert 'Profit_Net' to numeric type and count NaN values
df1['Profit_Net'] = pd.to_numeric(df1['Profit_Net'], errors='coerce')
nan_count = df1['Profit_Net'].isna().sum()
print(f"Number of NaN values in 'Profit_Net': {nan_count}")

# Create a 'Trades' column based on 'Profit_Net'; 1 if not NaN, 0 if NaN
df1['Trades'] = np.where(df1['Profit_Net'].isnull(), 0, 1)


# Calculate price differences in pips and initialize columns for running totals
df1['Price_Diff'] = (df1['Open'].diff().abs().fillna(0)) * 10000 #the multiply by 10000 is to calculate PIPS, delete the "*1000" if not using FOREX
df1['Movement'] = 0
df1['Pre_Trade_Max_Movement'] = np.nan  # Will store max movement value just before a trade

# Calculate running totals and pre-trade max movements
running_total = 0
max_movement_before_trade = 0
for i, row in df1.iterrows():
    running_total += row['Price_Diff']
    if row['Trades'] == 1:
        max_movement_before_trade = running_total
        running_total = 0  # Reset after a trade
    df1.at[i, 'Movement'] = running_total
    if row['Trades'] == 1:
        df1.at[i, 'Pre_Trade_Max_Movement'] = max_movement_before_trade

# Extract rows where trades occurred into a new DataFrame
df2 = df1[df1['Trades'] == 1].copy()
df2.reset_index(drop=True, inplace=True)

# ============================
# Database Upload
# ============================

# Create connection URL and engine for SQLAlchemy to upload data
connection_string_alchemy = f"mssql+pyodbc://{username}:{urllib.parse.quote_plus(password)}@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server"
engine = create_engine(connection_string_alchemy)

# Upload DataFrame to SQL database, replacing existing table
df2.to_sql('Calc_mkt_mvmnt_USD_CAD', engine, if_exists='replace', index=False)
