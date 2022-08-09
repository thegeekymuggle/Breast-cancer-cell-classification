
# coding: utf-8

# In[2]:


# pandas for handling data
import pandas as pd
# pandas for numeric opertions
import numpy as np
# matplotlib for visualization
import matplotlib.pyplot as plt
import seaborn as sns 
# tensorflow our machine learning library
import numpy as np #array manipulations
import pandas as pd  # high level data structure manipulation and other data analysis operation
import matplotlib.pyplot as plt # data visualization
import os # operating system functionality
import math # math operation
from sklearn import linear_model,metrics,tree # building statistical models 
from sklearn import model_selection,cross_validation # for test train split , cross validation and related functionality
from IPython.display import SVG # to show graphs in Notebook
import graphviz # ort tensorflow as tf
# train test split for spliting our data
from sklearn.model_selection import train_test_split
# one hot encoding for one hot encoding
from sklearn.preprocessing import OneHotEncoder


# In[ ]:


COLUMNS = ['Sample_code_number', 'Clump_Thickness','Uniformity_of_Cell_Size','Uniformity_of_Cell_Shape',
           'Marginal_Adhesion','Single_Epithelial_Cell_Size','Bare_Nuclei','Bland_Chromatin','Normal_Nucleoli',
           'Mitoses','Class']


# In[ ]:


#COLUMNS[]


# In[3]:


import numpy as np #array manipulations
import pandas as pd  # high level data structure manipulation and other data analysis operation
import matplotlib.pyplot as plt # data visualization
import os # operating system functionality
import math # math operation
from sklearn import linear_model,metrics,tree # building statistical models 
from sklearn import model_selection,cross_validation # for test train split , cross validation and related functionality
from IPython.display import SVG # to show graphs in Notebook
import graphviz # 


# In[4]:


path = os.getcwd()


# In[5]:


print(path)


# In[6]:


header = open("data/field_names.txt","r")

#read the lines from the text file and split each one in a new line
header_line = header.read().splitlines()
header_line


# In[7]:


data = pd.read_csv("data/breast-cancer.csv",names=header_line)

#print the top 5 rows the head()
print(data.head())


# In[8]:


#data.head()


# In[9]:


print(data)


# In[10]:


print(type(data))


# In[11]:


print(data.dtypes)


# In[12]:


data.ID = data.ID.astype('object')

#converting diagnosis to nominal categorical data type (there is no order between malignant or benign hence specify ordered = False as a parameter)
data.diagnosis = pd.Categorical(data.diagnosis,ordered = False)

#check the data types after conversion
#data.dtypes


# In[13]:


print(np.isnan(data.loc[:,'radius_mean':'fractal_dimension_worst' ].any().any()))


# In[15]:


print(data.describe())


# In[16]:

 	
data_M = data[data['diagnosis']=='M']
data_B = data[data['diagnosis']=='B']


# In[17]:


data_M = data[data['diagnosis']=='M']
data_B = data[data['diagnosis']=='B']


# In[18]:


data.diagnosis.value_counts().plot(kind ='bar')
plt.title("Histogram of Breast Cancer Diagnosis results")
plt.ylabel("Frequency")


# In[20]:


data.diagnosis.value_counts().plot(kind ='pie')
plt.title("Histogram of Breast Cancer Diagnosis results")
plt.ylabel("Frequency")


# In[24]:


fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(data_M['smoothness_mean'], data_M['compactness_mean'], s=50, c='b', marker='o', label='Malignant')
ax.scatter(data_B['smoothness_mean'], data_B['compactness_mean'], s=50, c='g', marker='XX', label='Benign')
ax.legend()
ax.set_xlabel('smoothness mean')
ax.set_ylabel('compactness mean')


# In[26]:


fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(data_M['smoothness_mean'], data_M['compactness_mean'], s=50, c='g', marker='o', label='Malignant')
ax.scatter(data_B['smoothness_mean'], data_B['compactness_mean'], s=50, c='b', marker='X', label='Benign')
ax.legend()
ax.set_xlabel('smoothness mean')
ax.set_ylabel('compactness mean')


# In[27]:



#randomly choosing columns radius mean and smoothness mean to plot on 2-D axis for visual exploration

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(data_M['radius_mean'], data_M['smoothness_mean'], s=50, c='b', marker='o', label='Malignant')
ax.scatter(data_B['radius_mean'], data_B['smoothness_mean'], s=50, c='r', marker='x', label='Benign')
ax.legend()
ax.set_xlabel('radius mean')
ax.set_ylabel('smoothness mean')


# In[28]:


model_log = linear_model.LogisticRegression()
model_log.fit(x_train,y_train)


# In[29]:


x_train, x_test, y_train, y_test = model_selection.train_test_split(data.loc[:,'radius_mean':], data['diagnosis'], test_size=0.3, random_state=0)


# In[30]:


x_train.shape, y_train.shape, x_test.shape, y_test.shape


# In[31]:


print(data.xtrain_shape)


# In[32]:


model_log = linear_model.LogisticRegression()
model_log.fit(x_train,y_train)


# In[33]:


pd.DataFrame(list(zip(x_train.columns,np.transpose(model_log.coef_))))


# In[34]:



#evaluating the fitness of model on train data itself
model_log.score(x_train,y_train)


# In[35]:


y_predicted_log = model_log.predict(x_test)
y_prob_predicted_log = model_log.predict_proba(x_test)
print (y_predicted_log)
print(y_prob_predicted_log)


# In[36]:


print(metrics.accuracy_score(y_test,y_predicted_log))


# In[37]:


fpr, tpr, thresholds = metrics.roc_curve(y_test, y_prob_predicted_log[:,1],pos_label= "M")
roc_auc = metrics.auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc


# In[39]:


fpr, tpr, thresholds = metrics.roc_curve(y_test, y_prob_predicted_log[:,1],pos_label= "M")
roc_auc = metrics.auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)
plt.plot(fpr,tpr,label='AUC = %0.2f'% roc_auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.title('Receiver operating characteristic')print('confusion Matrix\n',metrics.confusion_matrix(y_test, y_predicted_log))
print('\n Classification report\n', metrics.classification_report(y_test, y_predicted_log))


# In[40]:


print('confusion Matrix\n',metrics.confusion_matrix(y_test, y_predicted_log))
print('\n Classification report\n', metrics.classification_report(y_test, y_predicted_log))


# In[41]:


model_dtree = tree.DecisionTreeClassifier(criterion= "gini",random_state= 10).fit(x_train,y_train)


# In[42]:


graph = graphviz.Source( tree.export_graphviz(model_dtree, out_file=None, feature_names=x_train.columns))
SVG(graph.pipe(format='svg'))


# In[49]:


graph = graphviz.Source( tree.export_graphviz(model_dtree, out_file=None, feature_names=x_train.columns))
SVG(graph.pipe(format='svg'))


# In[44]:


Var_imp = pd.DataFrame(model_dtree.feature_importances_, index=x_train.columns, 
                          columns=["Var_imp"])
Var_imp.sort_values(['Var_imp'],ascending= False)


# In[45]:


plt.bar(range(len(model_dtree.feature_importances_)),model_dtree.feature_importances_)
ticks = np.arange(0,len(Var_imp))
plt.xticks(ticks,Var_imp.index.values,rotation='vertical')


# In[46]:


y_predicted_dtree =model_dtree.predict(x_test)
y_prob_predicted_dtree = model_dtree.predict_proba(x_test)
print(y_predicted_dtree)


# In[47]:


print("The accuracy of the classifier is ", metrics.accuracy_score(y_test,y_predicted_dtree)*100)


# In[48]:


fpr, tpr, thresholds = metrics.roc_curve(y_test, y_prob_predicted_dtree[:,1],pos_label= "M")
roc_auc = metrics.auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)


# In[50]:


graph = graphviz.Source( tree.export_graphviz(model_dtree, out_file=None, feature_names=x_train.columns))
SVG(graph.pipe(format='svg'))

