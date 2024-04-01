#588 Regression for interest rates
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.optimize import curve_fit
import math



def TimeToDate(timestamps):
    datetime_objects = [datetime.strptime(timestamp, '%m/%d/%y %H:%M') for timestamp in timestamps]
    dates = [dt.strftime('%m/%d/%Y') for dt in datetime_objects]
    return dates


dfraw=pd.read_csv('ProjectData - in.csv')
dfrates=pd.read_csv('Interest Rates.csv')
dfRates=dfrates.loc[:,'Date':'USD']

#TIME MANIPULATION    
times=[]
for time in dfraw['When']:
    times.append(time)

Dates=TimeToDate(times)

dfraw['Date']=Dates
#__________________________________________________________

#DF PREP
unwanted=[]
for interestrate in dfRates:
    if interestrate != 'CAD' and interestrate!= 'USD' and interestrate!='Date':
        unwanted.append(interestrate)
dfRates=dfRates.drop(columns=unwanted)

Rates=[]
for rate in dfRates['CAD']:
    Rates.append(rate)

days=[]
for day in dfRates['Date']:
    days.append(day)




dfrelevant=dfraw.drop(columns=['SPX','SPY','NDX','VIX','TLT','1st Level', '2nd Level'])
#________________________________________________________________________

#Merging
dfrelevant['Date']=pd.to_datetime(dfrelevant['Date'], format='%m/%d/%Y')
dfrelevant=dfrelevant.sort_values(by=['Date'], axis=0, ascending=True)
dfRates['Date']=pd.to_datetime(dfRates['Date'], format='%m/%d/%Y')
dfRates=dfRates.sort_values(by=['Date'], axis=0, ascending=True)

DF=pd.merge_asof(dfrelevant,dfRates, on='Date', direction='backward')

Trouble=[]
for i in DF['Net Profit']:
    Trouble.append(i)
indexes=[]    
for i in Trouble:
    if i=='#DIV/0!':
        indexes.append(Trouble.index(i))
for i in indexes:
    Trouble[i]=0
Trouble[6604]=0
DF=DF.drop(columns='Net Profit')
DF['Net Profit']=Trouble

Trouble=[]
for i in DF['Gross Profit']:
    Trouble.append(i)
indexes=[]    
for i in Trouble:
    if i=='#DIV/0!':
        indexes.append(Trouble.index(i))
for i in indexes:
    Trouble[i]=0
Trouble[6604]=0
DF=DF.drop(columns='Gross Profit')
DF['Gross Profit']=Trouble

        

Required=['Net Profit', 'Gross Profit', 'Contracts']
for category in Required:
    floatingvalues=[]
    for value in DF[category]:
        floatingvalues.append(float(value))
    DF=DF.drop(columns=[category])
    DF[category]=floatingvalues

DF['CAD']=DF['CAD'].fillna(method='bfill')
DF['USD']=DF['USD'].fillna(method='bfill')
A_DF=DF[DF['Side']=='A' ]
B_DF=DF[DF['Side']=='B']
#_____________________________________________

#Regression:
Yvars=['Net Profit','Gross Profit','Contracts']
dataframes=[DF,A_DF,B_DF]
DF.name='Joint'
A_DF.name='Side A'
B_DF.name='Side B'
RATES=['CAD','USD']
for df in dataframes:
    print('')
    print('')
    for variable in Yvars:
        for rate in RATES:
            y=df[variable]
            y.name=variable
            x=df[rate]
            x=sm.add_constant(x)
            results=sm.OLS(y,x).fit()
            print(df.name)
            print(results.summary())
            predictions=results.predict(x)
            plt.plot(df[rate],predictions)
            plt.scatter(df[rate],y)
            plt.title(df.name)
            plt.xlabel(x)
            plt.ylabel(y.name)
            plt.show()
            if variable=='Contracts':
                floatlist=[]
                for i in df['Contracts']:
                    floatlist.append(float(i))
                logs=[]
                for floater in floatlist:
                    if floater > 0:
                        floater=math.log(floater)
                    logs.append(floater)    
                df['LogContracts']=logs
                results=sm.OLS(df['LogContracts'],x).fit()
                print(results.summary())
                predictions=results.predict(x)
                plt.plot(df[rate],predictions)
                plt.scatter(df[rate],df['LogContracts'])
                plt.xlabel(x)
                plt.ylabel('Log Contracts')
                plt.title(df.name)
                plt.show()
            testing=input('Press enter to go to next graph')  
#__________________________________________________________
#Multivariate
print('')
print('')
print('Multivariate Analysis')
resume=input('press enter to continue')
print('')
print(df.head())
print('')
for df in dataframes:
    for variable in Yvars:
        y=DF[variable]
        x=DF.loc[:,'CAD':'USD']
        x=sm.add_constant(x)
        results=sm.OLS(y,x).fit()
        print(df.name)
        print(results.summary())
        predictions =results.predict(x)
        plt.plot(x.iloc[:, 1],predictions)
        plt.scatter(x.iloc[:, 1],y)
        plt.title(df.name)
        plt.xlabel(x)
        plt.ylabel(y.name)
        plt.show()
#____________________________________________________________________

#Non-linear regression

#proposed equation= aX**b+c. Looking to optimize a,b,c





#________________________________________________________________________

#Will try to run a Lasso regression to see if I should pay attention to US or CAD rates







