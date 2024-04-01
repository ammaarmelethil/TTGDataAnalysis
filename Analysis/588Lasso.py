#NEW Rates
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime
import math
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import cross_val_score

def TimeToDate(timestamps):
    datetime_objects = [datetime.strptime(timestamp, '%m/%d/%y %H:%M') for timestamp in timestamps]
    dates = [dt.strftime('%Y-%m-%d') for dt in datetime_objects]
    return dates

def TimeToDate2(timestamps):
    datetime_objects = [datetime.strptime(timestamp, '%m/%d/%Y') for timestamp in timestamps]
    dates = [dt.strftime('%Y-%m-%d') for dt in datetime_objects]
    return dates

ttgdf=pd.read_csv('ProjectData - in.csv')
dfinterestrates=pd.read_csv('NewRates.csv')
dfUSrates=pd.read_csv('USrates.csv')

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

dfrelevant=ttgdf.drop(columns=['SPX','SPY','NDX','VIX','TLT','1st Level', '2nd Level'])
############################################################################### For hand-in, will simplify into a for loop
dfrelevant['Date']=pd.to_datetime(dfrelevant['Date'],format='%Y-%m-%d')
dfrelevant=dfrelevant.sort_values(by=['Date'],axis=0, ascending=True)
dfinterestrates['Date']=pd.to_datetime(dfinterestrates['Date'],format='%Y-%m-%d')
dfinterestrates=dfinterestrates.sort_values(by=['Date'],axis=0, ascending=True)
dfUSrates['Date']=pd.to_datetime(dfUSrates['Date'],format='%Y-%m-%d')
dfUSrates=dfUSrates.sort_values(by=['Date'],axis=0, ascending=False)
DF=dfrelevant.merge(dfinterestrates, on='Date',how='inner')
#################################################################################
DF=DF.merge(dfUSrates, on='Date', how='inner',suffixes=(None,None))
#Cleaning
Issues=['Net Profit', 'Gross Profit']
for issue in Issues:
    Trouble=[]
    for i in DF[issue]:
        Trouble.append(i)
    indexes=[]    
    for i in Trouble:
        if i=='#DIV/0!':
            indexes.append(Trouble.index(i))
    for i in indexes:
        Trouble[i]=0
    Trouble[6566]=0
    DF=DF.drop(columns=issue)
    DF[issue]=Trouble


ratelist=[]
for i in DF['Rates']:
    ratelist.append(i)
for i in ratelist:
    if i=='Bank holiday':
        ratelist[ratelist.index(i)]=ratelist[ratelist.index(i)-1]
floatingrates=[]
for i in ratelist:
    floatingrates.append(float(i))
DF=DF.drop(columns='Rates')
DF['Rates']=floatingrates
DF=DF.merge(dfUSrates, on='Date',how='inner')
Required=['Net Profit', 'Gross Profit', 'Contracts']
for category in Required:
    floatingvalues=[]
    for value in DF[category]:
        floatingvalues.append(float(value))
    DF=DF.drop(columns=[category])
    DF[category]=floatingvalues
DF=DF.drop(columns='USRates_y')
DF['USRates']=DF['USRates_x']
DF=DF.drop(columns='USRates_x')



keyrates=['Rates','USRates']
for key in keyrates:
    cadlist=[]
    for i in DF['Rates']:
        cadlist.append(float(i))
    intcadlist=[]
    for i in range(len(cadlist)):
        i=(cadlist[i]-cadlist[i-1])/cadlist[i-1]
        intcadlist.append(i)
    finalcadlist=[]
    for i in intcadlist:
        if i <= 0:
            i*=-1
        finalcadlist.append(i)
    DF['Delta '+key]=finalcadlist
DFnoZero=DF[(DF['Delta Rates']!=0) & (DF['Delta Rates'] <0.01)]
DFnoZeroUS = DF[(DF['Delta USRates'] != 0) & (DF['Delta USRates'] < 0.01)]
#Regression
DF.name='Aggregate'
DFnoZero.name='DF no zeros'
DFnoZeroUS.name='US Change != 0'
Ys=['Net Profit','Contracts']#'Gross Profit', 'Contracts'] add back later
X=DF[['Delta Rates','Delta USRates','USRates','Rates']]
aggdataframes=[DF,DFnoZero,DFnoZeroUS]




for df in aggdataframes:
    print(df.name)
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    for Y in Ys:
        train_size = int(len(df) * 0.7)
        val_size = int(len(df) * 0.15)
        train, val, test = df[:train_size], df[train_size:train_size+val_size], df[train_size+val_size:]

        X_train, y_train = train[['Delta Rates', 'Delta USRates', 'USRates', 'Rates']], train[Y]
        X_val, y_val = val[['Delta Rates', 'Delta USRates', 'USRates', 'Rates']], val[Y]
        X_test, y_test = test[['Delta Rates', 'Delta USRates', 'USRates', 'Rates']], test[Y]

        


        alpha_values = [0.0125,0.025,0.05, 0.075, 0.1,0.125,0.15,0.175,0.2,0.225,0.25,0.275,0.3,0.4,0.6,0.8,0.9,1]


        mse_scores = []


        for alpha in alpha_values:
            lasso_reg = Lasso(alpha=alpha)
    
            scores = cross_val_score(lasso_reg, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

            mse_scores.append(np.mean(-scores))


        optimal_alpha = alpha_values[np.argmin(mse_scores)]
        print("Optimal alpha:", optimal_alpha)

        end=input('look at optimal')
        lasso_reg.fit(X_train, y_train)

        # Tune hyperparameters using the validation set (if necessary)
        # For example, you can use cross-validation or grid search here

        # Evaluate model performance on the validation set
        y_pred_val_net_profit = lasso_reg.predict(X_val)
        mse_val_net_profit = mean_squared_error(y_val, y_pred_val_net_profit)
        print(f"MSE for {Y} on validation set:", mse_val_net_profit)

        # Evaluate model performance on the test set
        y_pred_test_net_profit = lasso_reg.predict(X_test)
        mse_test_net_profit = mean_squared_error(y_test, y_pred_test_net_profit)
        print(f"MSE for {Y} on test set:", mse_test_net_profit)

        # Plotting code for Delta Rates
        plt.scatter(X_test['Delta Rates'], y_test, color='blue', label='Actual')
        plt.scatter(X_test['Delta Rates'], y_pred_test_net_profit, color='red', label='Predicted Net Profit')
        plt.xlabel('Delta Rates')
        plt.ylabel(Y)
        plt.title(df.name)
        plt.legend()
        plt.show()

        # Plotting code for Delta USRates
        plt.scatter(X_test['Delta USRates'], y_test, color='blue', label='Actual')
        plt.scatter(X_test['Delta USRates'], y_pred_test_net_profit, color='red', label='Predicted Net Profit')
        plt.xlabel('Delta USRates')
        plt.ylabel(Y)
        plt.title(df.name)
        plt.legend()
        plt.show()

        # Plotting code for Rates
        plt.scatter(X_test['Rates'], y_test, color='blue', label='Actual')
        plt.scatter(X_test['Rates'], y_pred_test_net_profit, color='red', label='Predicted Net Profit')
        plt.xlabel('Rates')
        plt.ylabel(Y)
        plt.title(df.name)
        plt.legend()
        plt.show()

        # Plotting code for USRates
        plt.scatter(X_test['USRates'], y_test, color='blue', label='Actual')
        plt.scatter(X_test['USRates'], y_pred_test_net_profit, color='red', label='Predicted Net Profit')
        plt.xlabel('USRates')
        plt.ylabel(Y)
        plt.title(df.name)
        plt.legend()
        plt.show()

        print(f"Lasso Coefficients for {Y} in {df.name}:")
        for feature, coefficient in zip(X.columns, lasso_reg.coef_):
            print(f"{feature}: {coefficient}")

        # Additional information
        print(f"Intercept: {lasso_reg.intercept_}")

        nextY=input('Press enter to go to next Y variable')
    nextdf=input('press enter to go to the next dataframe')
