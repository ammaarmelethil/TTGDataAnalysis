#SQL database into pandas dataframes
from sqlalchemy import create_engine, exc
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime
import math
from sklearn.linear_model import LassoCV
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import re #for SPX cleaning. Some dates were snuck in there. Will replace with previous valid value
import psycopg2
from sqlalchemy.exc import SQLAlchemyError
t1=datetime.now()

def TimeToDate(timestamps):
    datetime_objects = [datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S') for timestamp in timestamps]
    dates = [dt.strftime('%Y-%m-%d') for dt in datetime_objects]
    return dates
def TimeToDate2(timestamps):
    datetime_objects = [datetime.strptime(timestamp, '%Y-%m-%d') for timestamp in timestamps]
    dates = [dt.strftime('%Y-%m-%d') for dt in datetime_objects]
    return dates
def DoW(TimeList):
    try:
        # Convert the list to datetime format
        datetime_objects = [datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S') for timestamp in TimeList]
        # Extract the day of the week
        day_of_week = [dt.strftime('%A') for dt in datetime_objects]
        return day_of_week
    except ValueError as e:
        return f"Error: {e}. Invalid input format. Please provide times in 'YYYY-MM-DD HH:MM:SS' format."
server='192.168.50.221'
database='CapUOC_DataAnalysis'
username=''
password=''######Credentials

# Replace 'username', 'password', 'hostname', 'database_name' with your SQL Server credentials
connection_string = 'mssql+pyodbc://username:password@192.168.50.221/CapUOC_DataAnalysis?driver=ODBC+Driver+17+for+SQL+Server'

# Create SQLAlchemy engine
engine = create_engine(connection_string)

# Execute a SQL query and load the results into a DataFrame
query1 ='SELECT [When], Profit_Net,[Level1%], [Level2%] ,[Strength],[TH], Side, Contracts FROM Trades ORDER BY [When]'
df1=pd.read_sql(query1, engine)
print(datetime.now()-t1)

setup=input('type \'y\' to include tick data')
if setup=='y':
    #Setting up massive dataframe from the time period where trades exist
    query3="""Select      
                [Open],
                [Close],
                [High],
                [Low],
                Identifier,
                BarDateTime
            FROM
    Template_HistoData
Where BarDateTime > '2022-05-27 08:00:00.000'
    AND BarDateTime < '2023-12-29 23:00:00.000' AND Identifier<>'TLT'
Order By
    BarDateTime
    """
    
    ###^ you can write queries any which way. I had it like this from SQL. 
    df3=pd.read_sql(query3, engine)
print(datetime.now()-t1) #Depending on network connection and speed, this takes between 3 and 8 minutes to get to this point

# Now you have your data in a DataFrame, you can work with it as needed
if setup=='y':
    df3['When']=df3['BarDateTime']
    df3=df3.drop(columns='BarDateTime')#Renaming in preparation for a merge on the time values
    print(df3)
engine.dispose()    #Must close connection after retrieving data to prevent errors and slowdowns
print(datetime.now()-t1)
times = []

for time in df1['When']:
    times.append(str(time))

Dates = TimeToDate(times)
df1['Date'] = Dates





DFA = df1
DFA = DFA[pd.notna(DFA['Side'])]  # Filter out rows where 'Side' is not NaN

sides=[]
for i in DFA.Side:
    sides.append(i)
adjsides=[]
for i in sides:
    if i =='A  ':
        adjsides.append('A')
    elif i =='B  ':
        adjsides.append('B')
DFA=DFA.drop(columns='Side')
DFA['Side']=adjsides #Dealing with the collumn's extra ' ' on its strings
    
    


NP=[]
for i in DFA['Profit_Net']:
    NP.append(i)
for i in NP:
    if i =='#DIV/0!':
        NP[NP.index(i)]=float(0.0)
    else:
        NP[NP.index(i)]=float(i)
DFA['Net Profit']=NP
DFA=DFA.drop(columns='Profit_Net') #Giving the variable a more conventional name


Levels=['Level1%','Level2%'] #Dealing with missing values
for level in Levels:
    levellist=[]
    for thing in DFA[level]:
        levellist.append(thing)
    for value in levellist:
        if value=="#VALUE!":
            levellist[levellist.index(value)]=0
    DFA[level]=levellist
            
            

        
DFA['Level1%'] = DFA['Level1%'].astype(str).str.rstrip('%')#% was attached to each value making Power BI recognize it as text and not a number
DFA['Level1%'] = pd.to_numeric(DFA['Level1%']) / 100

DFA['Level2%'] = DFA['Level2%'].astype(str).str.rstrip('%')
DFA['Level2%'] = pd.to_numeric(DFA['Level2%']) / 100





if setup=='y':
    DFB=DFA.merge(df3, on='When', how='outer') #Creation of another DF with the trading and 5s data included
    DFB=DFB.sort_values(by='When')
    times = []
    for time in DFB['When']:
        times.append(str(time))
        

    daysofweek=DoW(times)
    DFB['DAY']=daysofweek #To get the named day of the week from the date

    Dates = TimeToDate(times)
    DFB['DATE'] = Dates
    DFB=DFB.drop(columns='Date')
    #DFB.fillna(float(0))
    print(datetime.now()-t1)
    Trade=[]
    trades=[]
    for i in DFB.Side:
        trades.append(i)
    for i in trades:
        if i=='A' or i=='B':
            Trade.append(1)
        else:
            Trade.append(0) #Creating dummy variable to see when a trade happened
    DFB['Trade']=Trade
    DFB=DFB.sort_values('When')
    print(datetime.now()-t1)
    print('-'*80)
    

    
###################################################################################################################
    
  
    
    
    
    
       
        
    

    #DFB.to_csv('Mega.csv', index=False)        #Upload of massive table to SQL to be used in PBI
    #print('in .csv')
    #DFB=DFB.drop(columns='ID')
newsql=''
newsql=input('Want to update SQL table?')
if newsql=='y':
    engine = create_engine(connection_string)
    chunks = np.array_split(DFB, 400)
    chunks[0].to_sql('PBI', engine, if_exists='replace', index=False) #roughly 48,000 rows per chunk 


    for i in range(1, 400):
        chunks[i].to_sql('PBI', engine, if_exists='append', index=False)
        print(datetime.now()-t1)            
      
engine.dispose()


    


    
