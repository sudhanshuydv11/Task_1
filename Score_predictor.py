# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 16:30:47 2021

@author: sudha
"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model as lm
url="https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
df2=pd.read_csv(url)
print(df2)
plt.scatter(df2.Hours,df2.Scores,color='red',marker='+')
plt.grid()
new_df2=df2.drop('Scores',axis='columns')
predictor=lm.LinearRegression()
predictor.fit(new_df2,df2.Scores)
m=predictor.coef_
b=predictor.intercept_
x=df2.Hours
y=m*x+b
plt.plot(x,y,color='green')
pm=predictor.predict([[9.25]])
print("By studying 9.25 hrs a student might score=",pm)

