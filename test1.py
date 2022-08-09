import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style 
from time import time
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

df = pd.read_csv("wdbc.csv", header=None)

malig = df[1] == 'M'
nonmalig = df[1] == 'B'

df.loc[malig, 1] = 1
df.loc[nonmalig, 1] = 0

y = df[1]
df = df.drop([0,1],axis = 1)

X = df
'''
print(y)
print("...........................................................................")
print(X)
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

t4=time()
print ("***** Naive Bayes *****")
nb = BernoulliNB()
clf_nb=nb.fit(X_train,y_train)
score_nb=clf_nb.score(X_test,y_test)
print ("Acurracy: "+str(score_nb*100))
t5=time()
dur_nb=t5-t4
print ("time elapsed: "+str(dur_nb))
print("\n")

'''
zero, one = [], [] 

for i in y:
	if i == 0:
		zero.append(i)
	elif i == 1:
		one.append(i)

print(zero)
print(one)

style.use('ggplot')

plt.plot(zero, one)
plt.title("Histogram")
plt.show()
'''