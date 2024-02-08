#SQL database connector test
import pyodbc
server='192.168.50.221'
database='CapUOC_DataAnalysis'
username='clong'
password=''#Your Password

connection_string = f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'
connection = pyodbc.connect(connection_string)
cursor = connection.cursor()

# Fetch column names from the description
cursor.execute("SELECT * FROM dbo.Trades")
columns = [column[0] for column in cursor.description]
print("Columns:", columns)

cursor.close()
connection.close()
