#Cumulative code
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime
import math
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import re #for SPX cleaning. Some dates were snuck in there. Will replace with previous valid value



def TimeToDate(timestamps):
    datetime_objects = [datetime.strptime(timestamp, '%m/%d/%y %H:%M') for timestamp in timestamps]
    dates = [dt.strftime('%Y-%m-%d') for dt in datetime_objects]
    return dates

def TimeToDate2(timestamps):
    datetime_objects = [datetime.strptime(timestamp, '%m/%d/%Y') for timestamp in timestamps]
    dates = [dt.strftime('%Y-%m-%d') for dt in datetime_objects]
    return dates

#Importing
ttgdf=pd.read_csv('DATA.csv')
dfinterestrates=pd.read_csv('NewRates.csv')
dfUSrates=pd.read_csv('USrates.csv')

#Date manipulation
times=[]
for date in dfUSrates['Date']:
    times.append(date)
dfUSrates=dfUSrates.drop(columns='Date')
USDates=TimeToDate2(times)
dfUSrates['Date']=USDates
Usrates=[]
for rate in dfUSrates['LT COMPOSITE (>10 Yrs)']:
    Usrates.append(float(rate))
dfUSrates=dfUSrates.drop(columns=['LT COMPOSITE (>10 Yrs)','TREASURY 20-Yr CMT'])
dfUSrates['USRates']=Usrates


times=[]
for time in ttgdf['When']:
    times.append(time)

Dates=TimeToDate(times)
ttgdf['Date']=Dates
dfrelevant=ttgdf.drop(columns=['Gross Profit','TLT','1st Level', '2nd Level','Contracts'])#Dropping temporarily irrelevant columns

############MERGING############################################################# For hand-in, will simplify into a for loop
dfrelevant['Date']=pd.to_datetime(dfrelevant['Date'],format='%Y-%m-%d')
dfrelevant=dfrelevant.sort_values(by=['Date'],axis=0, ascending=True)
dfinterestrates['Date']=pd.to_datetime(dfinterestrates['Date'],format='%Y-%m-%d')
dfinterestrates=dfinterestrates.sort_values(by=['Date'],axis=0, ascending=True)
dfUSrates['Date']=pd.to_datetime(dfUSrates['Date'],format='%Y-%m-%d')
dfUSrates=dfUSrates.sort_values(by=['Date'],axis=0, ascending=False)
DF=dfrelevant.merge(dfinterestrates, on='Date',how='inner')
#################################################################################
DF=DF.merge(dfUSrates, on='Date', how='inner',suffixes=(None,None))#inner merge was needed with the IB rates but with my new rates it was uneccesary. This code still works but a simpler version exists


#Cleaning


Trouble=[]
for i in DF['Net Profit']:
    Trouble.append(i)
for issue in Trouble:
    if issue=='#DIV/0!':
        Trouble[Trouble.index(issue)]=Trouble[Trouble.index(issue)-1]
Solved=[]
for i in Trouble:
    Solved.append(float(i))
DF=DF.drop(columns='Net Profit')
DF['Net Profit']=Solved



ratelist=[]
for i in DF['Rates']:
    ratelist.append(i)
for i in ratelist:
    if i=='Bank holiday':
        ratelist[ratelist.index(i)]=ratelist[ratelist.index(i)-1] #Rates cleaning
floatingrates=[]
for i in ratelist:
    floatingrates.append(float(i))
DF=DF.drop(columns='Rates')
DF['Rates']=floatingrates
DF=DF.merge(dfUSrates, on='Date',how='inner')

DF=DF.drop(columns='USRates_y')
DF['USRates']=DF['USRates_x'] #Don't know why it did this but all required data is intact afterwards
DF=DF.drop(columns='USRates_x')


regex=re.compile(r'\d\d[/]\d\d[/]\d\d\d\d') #SPX cleaning was weird because some dates were mixed in with the sample data. Had to get its own section outside of loop. 
SPX=[]
for value in DF['SPX']:
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
DF.drop(columns='SPX')
DF['SPX']=newSPX
    
    
mktvars=['VIX','NDX'] #Easy stuff to work with
for var in mktvars:
    mktvarcleaning = []
    for value in DF[var]:
        mktvarcleaning.append(value)

    for i in range(len(mktvarcleaning)):
        if mktvarcleaning[i] == '#VALUE!':
            mktvarcleaning[i] = mktvarcleaning[i - 1]

    newmktvar = []
    for i in mktvarcleaning:
        newmktvar.append(float(i))
    DF=DF.drop(columns=var)
    DF[var]=newmktvar

SPYcleaning=[] #SPY had a few hundred missing values as opposed to less that 10. I converted using SPX values to get a clearer picture. A spread will have to be implemented in this calculation for further accuracy
SPXstuff=[]
for value in DF['SPY']:
    SPYcleaning.append(value)
for value in DF['SPX']:
    SPXstuff.append(value)
newSPY=[]
for value in SPYcleaning:
    if value!='#VALUE!':
        newSPY.append(value)
    else:
        newSPY.append(float((float(SPXstuff[SPYcleaning.index(value)])/10)))
DF=DF.drop(columns='SPY')
#float commands werent working so had to hard list it
SPY=[]
for string in newSPY:
    SPY.append(float(string))
DF['SPY']=SPY

keyrates=['Rates','USRates']
for key in keyrates:
    cadlist=[]
    for i in DF['Rates']:
        cadlist.append(float(i))
    intcadlist=[]
    for i in range(len(cadlist)):
        i=(cadlist[i]-cadlist[i-1])/cadlist[i-1]
        i*=100
        intcadlist.append(i)
    finalcadlist=[]
    for i in intcadlist:
        if i <= 0:
            i*=-1
        finalcadlist.append(i)
    DF['Delta% '+key]=finalcadlist
DFnoZero=DF[(DF['Delta% Rates']!=0) & (DF['Delta% Rates']<=0.5)]#Most movement of rates occurs at in small oscillation-like ways (under 0.5%). Omitted when rates did not change and when changes were large. Made data messy and provided no insight
DFnoZeroUS=DF[(DF['Delta% USRates']!=0) & (DF['Delta% USRates']<=0.5)]#^^^^^This is due to factors other than the FED meetings and are responses to supply and demand as well as Labor mkt factors
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^This in turn impacts the lending/borrowing mkts which creates these small shifts that are seen daily*
#Regression

DF.name='Aggregate'
DFnoZero.name='Rates Nonzero %Change'
DFnoZeroUS.name='USRates Nonzero %Change'
YVars=['SPX','NDX','SPY','VIX','Net Profit']
XVars=['Rates','Delta% Rates','USRates','Delta% USRates']
dataframes=[DF,DFnoZero,DFnoZeroUS]


Restart='yes'
while Restart=='yes':
    Regression=''
    while Regression != 'OLS' and Regression !='LASSO':
        Regression=input('What kind of regression do you want? (OLS/LASSO)')

    #OLS Regression   
    if Regression == 'OLS':
        for df in dataframes:
            for responding in YVars:
                for independent in XVars:
                    print('Dataframe: '+df.name)
                    print('Responding Variable: '+responding)
                    print('Independent Variable: '+independent, end='')
                    y=df[responding]
                    y.name=responding
                    x=df[independent]
                    x.name=independent
                    x=sm.add_constant(x)
                    results=sm.OLS(y,x).fit()
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
            NEXTDF=input('hit enter to go to next DF')

    elif Regression=='LASSO':
        alpha_values = []
        for i in range(1,201):
            alpha_values.append(float(i)/20)
        print('-'*80)
        print('Graphs come in 4s. Always in this order for independent variables: Delta% Rates, Delta% USRates, USRates, Rates')
        print('-'*80)
        for df in dataframes:
            print('Dataframe: '+df.name)
            df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
            for Y in YVars:
                print('Responding: '+Y)
                X = df[['Delta% Rates', 'Delta% USRates', 'USRates', 'Rates']]
                y = df[Y]
                #Scaling the data
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                # Splitting data into training and test sets
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.75, random_state=42)
                #Optimizing for alpha parameter for LASSO model
                mse_scores = []

                for alpha in alpha_values:
                    lasso_reg = Lasso(alpha=alpha)
            
                    scores = cross_val_score(lasso_reg, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
                    mse_scores.append(np.mean(-scores))

                optimal_alpha = alpha_values[np.argmin(mse_scores)]
                print("Optimal alpha:", optimal_alpha)
        
                lasso_reg = Lasso(alpha=optimal_alpha)
                lasso_reg.fit(X_train, y_train)

        
                y_pred_test = lasso_reg.predict(X_test)
                mse_test = mean_squared_error(y_test, y_pred_test)
                print(f"MSE for {Y} on test set:", mse_test)
                graph=input('see graph, press \'y\'')
                #Visualization. When I tried making a for loop to iterate over the xvars, it refused to run. Had to make one for each Xvar. Needs fixing for final copy
                if graph=='y':
                    plt.scatter(X_test[:, 0], y_test, color='blue', label='Actual')
                    plt.scatter(X_test[:, 0], y_pred_test, color='red', label='Predicted Net Profit')
                    plt.xlabel('Delta% Rates')
                    plt.ylabel(Y)
                    plt.title(df.name)
                    plt.legend()
                    plt.show()

                graph=input('see graph, press \'y\'')
                if graph=='y':
                    plt.scatter(X_test[:, 1], y_test, color='blue', label='Actual')
                    plt.scatter(X_test[:, 1], y_pred_test, color='red', label='Predicted Net Profit')
                    plt.xlabel('Delta% USRates')
                    plt.ylabel(Y)
                    plt.title(df.name)
                    plt.legend()
                    plt.show()

                graph=input('see graph, press \'y\'')
                if graph=='y':
                    plt.scatter(X_test[:, 3], y_test, color='blue', label='Actual')
                    plt.scatter(X_test[:, 3], y_pred_test, color='red', label='Predicted Net Profit')
                    plt.xlabel('Rates')
                    plt.ylabel(Y)
                    plt.title(df.name)
                    plt.legend()
                    plt.show()

                graph=input('see graph, press \'y\'')
                if graph=='y':
                    plt.scatter(X_test[:, 2], y_test, color='blue', label='Actual')
                    plt.scatter(X_test[:, 2], y_pred_test, color='red', label='Predicted Net Profit')
                    plt.xlabel('USRates')
                    plt.ylabel(Y)
                    plt.title(df.name)
                    plt.legend()
                    plt.show()

                print(f"Lasso Coefficients for {Y} in {df.name}:")
                for feature, coefficient in zip(['Delta% Rates', 'Delta% USRates', 'USRates', 'Rates'], lasso_reg.coef_):
                    print(f"{feature}: {coefficient}")

                print(f"Intercept: {lasso_reg.intercept_}")#this may or may not be important so i left it in

                nextY=input('Press enter to go to next Y variable')
            nextdf=input('press enter to go to the next dataframe')
    Restart=input('Type \'yes\' to return to regression selection or type \'no\' to end program')
    while Restart != 'yes' and Restart!= 'no':
        Restart=input('Type \'yes\' to return to regression selection or type \'no\' to end program')
    
        
