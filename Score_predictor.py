# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 16:30:47 2021

@author: sudha
"""
import pandas as pd
from sklearn import linear_model as lm
url="https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
df2=pd.read_csv(url) 
new_df2=df2.drop('Scores',axis='columns')
print(new_df2)
predictor=lm.LinearRegression()
predictor.fit(new_df2,df2.Scores)
predictor.predict([[9.25]])


