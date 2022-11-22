from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
#Read File
data = pd.read_csv("breastCancer.csv")
#Drop column
data.drop(["id","Unnamed: 32"],axis=1,inplace=True)
#เก็บค่า 
y = data.diagnosis.values
#normalization
x_data=data.drop(["diagnosis"], axis=1)
x=(x_data-np.min(x_data))/(np.max(x_data) -np.min(x_data))
#train test split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)
#fit
dt= DecisionTreeClassifier()
dt.fit(x_train,y_train)
#Predict data
y_pred = dt.predict(x_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Accuracy : {:.3f} ".format(dt.score(x_test,y_test)))
#Show
tree.plot_tree(dt.fit(x_train,y_train))
plt.show()









