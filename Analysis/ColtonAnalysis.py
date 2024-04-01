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

data = pd.read_csv("DATA.csv") #Initial data

# Data Cleaning
data = data[data.apply(lambda x: x != "#VALUE!").all(axis=1)]
data['When'] = pd.to_datetime(data['When'])

# Convert columns to numeric and remove missing values
numeric_cols = ["Net Profit", "SPY", "VIX", "TLT", "SPX", "NDX"]
data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce')
data = data.dropna(subset=numeric_cols)

# Split data into features (X) and target (y)
X = data[["SPY", "VIX", "TLT", "SPX", "NDX"]]
y = data["Net Profit"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Fit Lasso regression model with cross-validation
lasso_cv_model = LassoCV(cv=5,max_iter=2700)
lasso_cv_model.fit(X_train, y_train)

# Print coefficients
lasso_coefs = dict(zip(X.columns, lasso_cv_model.coef_))
print(lasso_coefs)
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
query1 ='SELECT [When], Profit_Net, [ SPX ],[Level1%], [Level2%] ,[Strength],[TH],[ NDX ], VIX, Side, SPY, Contracts FROM Trades ORDER BY [When]'
df1=pd.read_sql(query1, engine)
query2= 'SELECT Date,[1Mo], [1Yr], [10Yr] FROM RiskFreeCurve WHERE  (Date >= CONVERT(DATETIME, \'2022-05-27 00:00:00\', 102)) and (Date <= CONVERT(DATETIME, \'2023-12-29 00:00:00\', 102)) ORDER BY Date'
df2=pd.read_sql(query2, engine)
print(datetime.now()-t1)

setup=input('type \'y\' to include tick data')
if setup=='y':
    #Setting up massive dataframe from the time period where trades exist
    query3="""Select      
                [Open],
                [Close],
                [High],
                [Low],
                BarDateTime
            FROM
    Template_HistoData
Where BarDateTime > '2022-05-27 08:00:00.000'
    AND BarDateTime < '2023-12-29 23:00:00.000'
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
    df3 = df3.drop_duplicates(subset=['When'])
    print(df3)
engine.dispose()    #Must close connection after retrieving data to prevent errors and slowdowns
print(datetime.now()-t1)
times = []

for time in df1['When']:
    times.append(str(time))

Dates = TimeToDate(times)
df1['Date'] = Dates

times = []
for time in df2['Date']:
    times.append(str(time))

Dates = TimeToDate2(times)
df2 = df2.drop(columns='Date')#Trying to get by the date 
df2['Date'] = Dates

df1['SPX'] = df1[' SPX ']#Dealing with the collumns with empty spaces as part of the strings
df1['NDX'] = df1[' NDX ']
df1 = df1.drop(columns=[' SPX ', ' NDX '])

DFA = df1.merge(df2, on='Date', how='outer')
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
    

columns=['1Mo','1Yr','10Yr']
for column in columns:
    DFA[column].ffill(inplace=True)
    #DFA[column].bfill(inplace=True)
    
mktvars=['VIX','NDX'] #Easy stuff to work with
for var in mktvars:
    mktvarcleaning = []
    for value in DFA[var]:
        mktvarcleaning.append(value)

    for i in range(len(mktvarcleaning)):
        if mktvarcleaning[i] == '#VALUE!':
            mktvarcleaning[i] = mktvarcleaning[i - 1]

    newmktvar = []
    for i in mktvarcleaning:
        newmktvar.append(float(i.rstrip('%')))
    DFA=DFA.drop(columns=var)
    DFA[var]=newmktvar

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

regex=re.compile(r'\d\d[/]\d\d[/]\d\d\d\d') #SPX cleaning was weird because some dates were mixed in with the sample data. Had to get its own section outside of loop. 
SPX=[]
for value in DFA['SPX']:
    SPX.append(value)
dateissues=[]
SPXcleaning=[]
for value in SPX:
    if regex.search(value):
        dateissues.append(value)
for issue in dateissues:
    SPX[SPX.index(issue)]=SPX[SPX.index(issue)-1]
SPXcleaning=SPX
for i in range(len(SPXcleaning)):
    if SPXcleaning[i]=='#VALUE!':
        SPXcleaning[i]=SPXcleaning[i-1]
newSPX=[]      
for i in SPXcleaning:
    newSPX.append(float(i))
DFA.drop(columns='SPX')
DFA['SPX']=newSPX

SPYcleaning=[] #SPY had a few hundred missing values as opposed to less that 10. I converted using SPX values to get a clearer picture. A spread will have to be implemented in this calculation for further accuracy
SPXstuff=[]
for value in DFA['SPY']:
    SPYcleaning.append(value)
for value in DFA['SPX']:
    SPXstuff.append(value)
newSPY=[]
for value in SPYcleaning:
    if value!='#VALUE!':
        newSPY.append(value)
    else:
        newSPY.append(float((float(SPXstuff[SPYcleaning.index(value)])/10)))
DFA=DFA.drop(columns='SPY')
#float commands werent working so had to hard list it
SPY=[]
for string in newSPY:
    SPY.append(float(string))
DFA['SPY']=SPY



if setup=='y':
    DFB=DFA.merge(df3, on='When', how='outer') #Creation of another DF with the trading and 5s data included
    DFB=DFB.sort_values(by='When')
    times = []
    for time in DFB['When']:
        times.append(str(time))
        
    columns=['SPX','SPY','NDX','VIX','1Mo','1Yr','10Yr']
    for column in columns:
        DFB.loc[0,column]=DFA.loc[0,column]
        DFB[column].ffill(inplace=True)
        DFB[column].bfill(inplace=True) #filling NaN

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
if setup!='y':
    Restart='yes'
    while Restart=='yes':
        valid=['1Mo','1Yr','10Yr']
        irate=''
        while irate not in valid:
            print(valid)
            irate=input('Please select rate you wish to see always change. Choose from printed list') 

        
        ratelist=[]
        for i in DFA[irate]:
            ratelist.append(float(i))
        intratelist=[]
        for i in range(len(ratelist)):
            i=(ratelist[i]-ratelist[i-1])/ratelist[i-1]
            i*=100
            intratelist.append(i)
        finalratelist=[]
        for i in intratelist:
            if i <= 0:
                i*=-1
            finalratelist.append(i)
        DFA['%Change '+irate]=finalratelist  #Creation of rate collumns with recognizable names along with rate volatility
        print(DFA)
        
        engine = create_engine(connection_string)
        DFA.to_sql('With Rate Volatility', engine, if_exists='replace', index=False) #adding the table to SQL
        engine.dispose()
        
        DFRateChanges=DFA[(DFA[irate]!=0) & (DFA[irate]<=5)]#Most movement of rates occurs at in small oscillation-like ways (under 0.5%). Omitted when rates did not change and when changes were large. Made data messy and provided no insight
    #Will change the rate chages df to accurately see results for other rates. Can also tweak the numbers to change how large the sample data is for %Change. Smaller shows more but decreases sample size
        DFA.name='Aggregate'
        DFRateChanges.name='Only When Rates Change'
        YVars=['SPX','NDX','SPY','VIX','Net Profit']
        XVars=['1Mo','1Yr','10Yr','%Change '+irate]
        dataframes=[DFA,DFRateChanges]        
        Regression=''
        while Regression.upper() != 'OLS' and Regression.upper() !='LASSO':
            Regression=input('What kind of regression do you want? (OLS/LASSO)')

        #OLS Regression   
        if Regression.upper() == 'OLS':
            for df in dataframes:
                for responding in YVars:
                    for independent in XVars:
                        print('-'*80)
                        print('Dataframe: '+df.name)
                        print('Responding Variable: '+responding)
                        print('Independent Variable: '+independent, end='')
                        y=df[responding]
                        y.name=responding
                        x=df[independent]
                        x.name=independent
                        x=sm.add_constant(x)
                        results=sm.OLS(y,x).fit()
                        print('')
                        Table=input('Press \'t\' to see table. enter to continue')
                        if Table=='t':
                            print(results.summary())
                            graph=input('press \'y\' to see graph')
                            if graph=='y':
                            
                                predictions=results.predict(x)
                                plt.plot(df[independent],predictions)
                                plt.scatter(df[independent],y)
                                plt.title(df.name)
                                plt.xlabel(independent)
                                plt.ylabel(responding)
                                plt.show()
                            print('-'*80)
                NEXTDF=input('hit enter to go to next DF')

        elif Regression.upper()=='LASSO':
            alpha_values = []
            for i in range(1,401):
                alpha_values.append(float(i)/40)
            print('-'*80)
            print('Graphs come in 4s. Always in this order for independent variables: 1Mo,1Yr,10Yr,%Change '+ irate)
            print('-'*80)
            LASSOResults={}
            for df in dataframes:
                print('-'*80)
                print('Current Dataframe: '+df.name)
                print('-'*80)
                df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
                for Y in YVars:
                    print('Responding: '+Y)
                    X = df[['1Mo','1Yr','10Yr','%Change '+irate]]
                    y = df[Y]
                    #Scaling the data
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    # Splitting data into training and test sets
                    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.75, random_state=42)
                    #Optimizing for alpha parameter for LASSO model
                    mse_scores = []

                    for alpha in alpha_values:
                        lasso_reg = Lasso(alpha=alpha) #Alpha optimization
                
                        scores = cross_val_score(lasso_reg, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
                        mse_scores.append(np.mean(-scores))

                    optimal_alpha = alpha_values[np.argmin(mse_scores)]
                    print("Optimal alpha:", optimal_alpha)
            
                    lasso_reg = Lasso(alpha=optimal_alpha)
                    lasso_reg.fit(X_train, y_train)

            
                    y_pred_test = lasso_reg.predict(X_test)
                    mse_test = mean_squared_error(y_test, y_pred_test)
                    print(f"MSE for {Y} on test set:", mse_test)

                    print('1Mo')
                    graph=input('see graph, press \'y\'')
                    #Visualization. When I tried making a for loop to iterate over the xvars, it refused to run. Had to make one for each Xvar
                    if graph=='y':
                        plt.scatter(X_test[:, 0], y_test, color='blue', label='Actual')
                        plt.scatter(X_test[:, 0], y_pred_test, color='red', label='Predicted '+Y)
                        plt.xlabel('1Mo')
                        plt.ylabel(Y)
                        plt.title(df.name)
                        plt.legend()
                        plt.show()

                    print('1Yr')
                    graph=input('see graph, press \'y\'')
                    if graph=='y':
                        plt.scatter(X_test[:, 1], y_test, color='blue', label='Actual')
                        plt.scatter(X_test[:, 1], y_pred_test, color='red', label='Predicted '+Y)
                        plt.xlabel('1Yr')
                        plt.ylabel(Y)
                        plt.title(df.name)
                        plt.legend()
                        plt.show()

                    print('10Yr')
                    graph=input('see graph, press \'y\'')
                    if graph=='y':
                        plt.scatter(X_test[:, 2], y_test, color='blue', label='Actual')
                        plt.scatter(X_test[:, 2], y_pred_test, color='red', label='Predicted '+Y)
                        plt.xlabel('10Yr')
                        plt.ylabel(Y)
                        plt.title(df.name)
                        plt.legend()
                        plt.show()

                    print('%Change '+irate)
                    graph=input('see graph, press \'y\'')
                    if graph=='y':
                        plt.scatter(X_test[:, 3], y_test, color='blue', label='Actual')
                        plt.scatter(X_test[:, 3], y_pred_test, color='red', label='Predicted '+Y)
                        plt.xlabel('%Change '+irate)
                        plt.ylabel(Y)
                        plt.title(df.name)
                        plt.legend()
                        plt.show()

                 
    ############### Will add a way to indicate important graphs in future
                    
                    for feature, coefficient in zip(['1Mo', '1Yr', '10Yr', '%Change ' + irate], lasso_reg.coef_):
                        if df.name=='Aggregate':
                            LASSOResults[df.name+' '+feature+' '+Y] = coefficient
                        
                    print(f"Lasso Coefficients for {Y} in {df.name}:")
                    for feature, coefficient in zip(['1Mo','1Yr','10Yr','%Change '+irate], lasso_reg.coef_):
                        print(f"{feature}: {coefficient}")

                    print(f"Intercept: {lasso_reg.intercept_}")

                    nextY=input('Press enter to go to next Y variable')
                nextdf=input('press enter to go to the next dataframe')
        Restart=input('Type \'yes\' to return to regression selection or type \'no\' to end program and review coefficients')
        while Restart != 'yes' and Restart!= 'no':
            Restart=input('Type \'yes\' to return to regression selection or type \'no\' to end program and review coefficients.')

if setup!='y':
    for i in LASSOResults:
        print('Dataframe,rate,responding: '+str(i)+':'+str(LASSOResults[i]))

    mkts=['VIX','SPX','SPY','NDX']
    EFFECT={}
    for i in mkts:
        for j in LASSOResults:
            for k in lasso_coefs:                               
                if i in j and i in k and irate in j:
                    EFFECT[j]=float(LASSOResults[j])*float(lasso_coefs[k])

    EFFECT.update(lasso_coefs)
    print(EFFECT)
    
  
    
    
    
    
       
        
        
else:

    #DFB.to_csv('Mega.csv', index=False)        #Upload of massive table to SQL to be used in PBI
    #print('in .csv')
    #DFB=DFB.drop(columns='ID')
    newsql=''
    newsql=input('Want to update SQL table?')
    if newsql=='y':
        engine = create_engine(connection_string)
        chunks = np.array_split(DFB, 125)
        chunks[0].to_sql('5sDataMerged', engine, if_exists='replace', index=False) #roughly 48,000 rows per chunk 


        for i in range(1, 125):
            chunks[i].to_sql('5sDataMerged', engine, if_exists='append', index=False)
            print(datetime.now()-t1)            
      
    engine.dispose()


    


    
