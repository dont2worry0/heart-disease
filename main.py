from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from sklearn import preprocessing
import statsmodels.api as sm
import scipy.optimize as opt
import numpy as np
import pylab as pl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import jaccard_score

chd_data = pd.read_csv('train.csv')
chd_data.drop(['education'],inplace = True,axis = 1)
chd_data.dropna(axis=0,inplace=True)

#train and test sets
x = np.asarray(chd_data[['age', 'male', 'cigsPerDay', 'totChol', 'glucose']])
y = np.asarray(chd_data['TenYearCHD'])
#Normalize the dataset
x = preprocessing.StandardScaler().fit(x).transform(x)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=4)
print('Train set: ', x_train.shape, y_train.shape)
print('Test set: ', x_test.shape, y_test.shape)

#Modeling the dataset using Logistic Regression
log_reg=LogisticRegression()
log_reg.fit(x_train,y_train) #.fit is equal to training
y_pred=log_reg.predict(x_test)

print('')
print("Accuracy of the model in Jaccard score is : ",jaccard_score(y_test,y_pred))

#confusionmatrix top left : True postive,Top right: False Positives,bottom-right : True Negatives,Bottom-left : False Negatives
cm = confusion_matrix(y_test,y_pred)
cm_setup = pd.DataFrame(data=cm, columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize=(9,6))
sn.heatmap(cm, annot=True,fmt="d",cmap="Greens")
plt.show()

print('The details for confusion matrix is :')
print(classification_report(y_test, y_pred))