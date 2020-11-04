import numpy as np
import pandas as pd

df=pd.read_csv('https://raw.githubusercontent.com/anujvyas/IPL-First-Innings-Score-Prediction-Deployment/master/ipl.csv')
df.head()

df.drop(labels=['mid', 'date','venue', 'batsman', 'bowler', 'striker', 'non-striker'] , axis=1, inplace=True)

df.head()



df=df[df['overs']>=5.0]
df.head()

consistent_teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
                    'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
                    'Delhi Daredevils', 'Sunrisers Hyderabad']
df = df[(df['bat_team'].isin(consistent_teams)) & (df['bowl_team'].isin(consistent_teams))]

df=pd.get_dummies(df)
df.columns
df=df[['bat_team_Chennai Super Kings', 'bat_team_Delhi Daredevils',
       'bat_team_Kings XI Punjab', 'bat_team_Kolkata Knight Riders',
       'bat_team_Mumbai Indians', 'bat_team_Rajasthan Royals',
       'bat_team_Royal Challengers Bangalore', 'bat_team_Sunrisers Hyderabad',
       'bowl_team_Chennai Super Kings', 'bowl_team_Delhi Daredevils',
       'bowl_team_Kings XI Punjab', 'bowl_team_Kolkata Knight Riders',
       'bowl_team_Mumbai Indians', 'bowl_team_Rajasthan Royals',
       'bowl_team_Royal Challengers Bangalore',
       'bowl_team_Sunrisers Hyderabad','runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5',
       'total']]

df.head()



X=df.iloc[:,:-1]
Y=df.iloc[:,-1]
X.head()

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=0)

from sklearn.linear_model  import LinearRegression
lr=LinearRegression()
lr.fit(xtrain,ytrain)

pred=lr.predict(xtest)
pred

import matplotlib.pyplot as plt
plt.scatter(ytest,pred)

import seaborn as sns
sns.distplot(ytest-pred)

print(lr.predict([[0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,40,8,7.2,30,3]]))