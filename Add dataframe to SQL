import pandas as pd
from sqlalchemy import create_engine
DF=pd.read_csv('DATA.csv')
connection_string = 'mssql+pyodbc://username:password@192.168.50.221/CapUOC_DataAnalysis?driver=ODBC+Driver+17+for+SQL+Server'

# Create SQLAlchemy engine
engine = create_engine(connection_string)
try:
# Begin a transaction
    with engine.connect() as connection:
    # Begin a nested transaction
        trans = connection.begin()
        
        try:
            # Send DataFrame to SQL database table
            DF.to_sql('WHEN', engine, if_exists='replace', index=False)
            
            # Commit the transaction if successful
            trans.commit()
        except Exception as e:
            # Rollback the transaction in case of an error
            trans.rollback()
            print(f"Error occurred: {e}")
except SQLAlchemyError as e:
        print(f"SQLAlchemy error occurred: {e}")
finally:
    # Dispose of the connection
    engine.dispose()
