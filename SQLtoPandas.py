#SQL database into pandas dataframes
import pandas as pd
from sqlalchemy import create_engine
server='192.168.50.221'
database='CapUOC_DataAnalysis'
username=''
password
# Replace 'username', 'password', 'hostname', 'database_name' with your SQL Server stuff
connection_string = 'mssql+pyodbc://username:password@server/database?driver=ODBC+Driver+17+for+SQL+Server'

# Create SQLAlchemy engine(idk what it does)
engine = create_engine(connection_string)

#Make a query to get the data you need
query1 ='SELECT [When], Profit_Net, [ SPX ], [ NDX ], VIX, SPY FROM Trades ORDER BY [When]'
df1=pd.read_sql(query1, engine)
query2= 'SELECT Date, [1Mo], [1Yr], [10Yr] FROM RiskFreeCurve ORDER BY Date'
df2=pd.read_sql(query2, engine)
# Close the connection
engine.dispose()

# Data is in df and can be used normally
print(df1.head())
print(df2.head())
