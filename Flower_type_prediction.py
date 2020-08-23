import pandas as pd
import numpy as np
train=pd.read_csv(r'C:\Users\TARUN\Desktop\New folder (3)\Flower_category_prediction\Participants_Data\Train.csv')

#train['new_1']=np.mean(train['Area_Code'])


print(train.corr()['Class'])
Labels=train['Class']
train=train.drop('Class',axis=1)
Features=train[:]
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_sc=sc.fit_transform(Features)
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from xgboost import XGBClassifier

from catboost import CatBoostClassifier

catmodel = CatBoostClassifier()
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(4)
from sklearn.decomposition import PCA
x_poly = poly.fit_transform(x_sc)
catmodel.fit(x_poly,Labels)
test=pd.read_csv(r'C:\Users\TARUN\Desktop\New folder (3)\Flower_category_prediction\Participants_Data\Test.csv')
test_features=test[:]
test_sc=sc.transform(test_features)
test_poly=poly.transform(test_sc)

ans=catmodel.predict_proba(test_sc)
df=pd.DataFrame(ans)
df.to_csv(r'C:\Users\TARUN\Desktop\New folder (3)\Flower_category_prediction\Participants_Data\S1.csv')



