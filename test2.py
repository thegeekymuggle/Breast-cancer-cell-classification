import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

df = pd.read_csv("breast-cancer-wisconsin.csv", header=None)

#print(df)
for i in column:
	print(i)