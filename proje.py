# -*- coding: utf-8 -*-
"""
Created on Fri May  7 03:55:12 2021

@author: fener
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier,NeighborhoodComponentsAnalysis,LocalOutlierFactor

#warning Library
import warnings
warnings.filterwarnings("ignore")
#%% Data preprocessing

# data.csv dosyasını okur
data = pd.read_csv("data.csv")

# 'Unnamed: 32','id' sütunlarını drop eder
data.drop(['Unnamed: 32','id'],inplace=True,axis = 1)

# diagnosis sütununu target olarak değiştirir
data = data.rename(columns = {"diagnosis":"target"})

sns.countplot(data["target"])
print(data.target.value_counts())

# target sütunundaki string M ve B değerlerini 1 ve 0 olarak değiştirir
data["target"] = [1 if i.strip() == "M" else 0 for i in data.target]

print(len(data))

print(data.head())

print("Data shape",data.shape)

print(data.info())

describe = data.describe()

"""
Standardization

#Missing value olmadığı belirtilmiştir.
"""

#%% EDA (Explorer Data Anlaysis)

#Correlation (korelasyon matrisi)
corr_matrix = data.corr()  
#korelasyon matrisi gösterimi
sns.clustermap(corr_matrix, annot = True, fmt = ".2f") 
plt.title("Correlation Between Features")
plt.show()

threshold = 0.75
filtre = np.abs(corr_matrix["target"]) > threshold
corr_features = corr_matrix.columns[filtre].tolist()
sns.clustermap(data[corr_features].corr(),annot = True, fmt = ".2f")
plt.title("Correlation Between Features w Corr Threshold 0.75")

"""
there some correlated features
"""
# box plot 
data_melted = pd.melt(data,id_vars = "target",
                        var_name="features",value_name="value")
plt.figure()
sns.boxplot(x="features",y="value",hue="target",data=data_melted)
plt.xticks(rotation=90)
plt.show()

"""
Standardization-normalization
"""

#pair plot
sns.pairplot(data[corr_features],diag_kind = "kde",markers = "+",hue = "target")
plt.show()

# datada skewneslık var 
# skewnesslık varsa outliner detectionı doğru yapmalı ve verileri
# gaussiana çevirmeli normalization yapmalı

#%% Outlier Detection 
"""
# Veri seti içinde bulunan aykırı değerler
# local outlier global outlier
# outlier Detection yöntemi olarak Density Based --> Local Outlier Factor(LOF)

LOF tanım --> compare local density of one point to local density
of its K-NN

LOF değeri = (LRD. + LRD. / LRDa) X 1/2 --> LRD (Local Reachability Density)

LRD = 1/ ARD --> ARD  (Avarage Reachability Distance) RD/K-NN = ARD

RD (Reachability Distance) 
"""

# y değişkenine target sütununa koyar
y = data.target 

# x değikenine datadan target sütununu drop edip kalan sütunları koyar
x=data.drop(["target"],axis=1)
# x dataframe inin sütunlarını listeye çevirip comluns değşikenine koyar
columns = x.columns.tolist()

# Local outlier factor kullanarak outlier ve inlier verileri tahmin eder.
clf = LocalOutlierFactor()
lof_pred = clf.fit_predict(x)

print(lof_pred)

X_score = clf.negative_outlier_factor_

outlier_score = pd.DataFrame()
outlier_score["score"] = X_score

#threshold
threshold = -2.5
filtre = outlier_score["score"] < threshold
#outlier olanları bu değikende tutar
outlier_index = outlier_score[filtre].index.tolist()

plt.figure()
plt.scatter(x.iloc[outlier_index,0], x.iloc[outlier_index,1], color="blue",s=50,label="Outliers")
plt.scatter(x.iloc[:,0],x.iloc[:,1],color="k",s=3,label="Data Points")

# X_score değerini 0 ile 1 arasında normalizasyon yapar
radius = (X_score.max() - X_score)/(X_score.max() - X_score.min())
outlier_score["radius"] = radius

plt.scatter(x.iloc[:,0],x.iloc[:,1],s=1000*radius,edgecolors="r",facecolors="none",label="Outlier Scores")
plt.legend()
plt.show()


# Drop Outliers

x = x.drop(outlier_index)
y = y.drop(outlier_index).values


#%% Train Test Split

test_size = 0.3
X_train,X_test,Y_train,Y_test = train_test_split(x,y, test_size=test_size,random_state=42)

#%% Standardization

scaler = StandardScaler()
# X_train verisi ile scaeler eğilip ve transform yapılır
X_train = scaler.fit_transform(X_train)

# X_test verisi eğitilmiş scalera verilir ve X_test verisi transform yapılır
X_test = scaler.transform(X_test)

# X_train verisini daha önce alınan sütunlarla birleştirilip dataframe oluşturulur
X_train_df = pd.DataFrame(X_train,columns = columns)
# X_train verisinin describe ı
X_train_df_describe = X_train_df.describe()
# X_train dataframe ine target sütununu ekler
X_train_df["target"] = Y_train
print(X_train_df_describe)


data_melted = pd.melt(X_train_df,id_vars="target",var_name="features",value_name="value")

plt.figure()
sns.boxplot(x="features",y="value",hue="target",data=data_melted)
plt.xticks(rotation=90)

# pair plot
sns.pairplot(X_train_df[corr_features],diag_kind = "kde",markers ="+",hue = "target")
plt.show()

 
#%% BASIC K-NN METHOD

"""
KNN PROS AND CROSS
#EĞİTME YOK
#İMPLEMENT ETMESİ KOLAY
#TUNE ETMESİ KOLAY

-----------------

#OUTLİER VARSA MODEL DÜZGÜN ÇALIŞMAZ
#ÇOK VERİ VARSA KNN ALGIRİTMASI YAVAŞLAR
#CURSE OF DİMENSİONALİTY - ÇOK FAZLA FEATURE VARSA ALGORİTMA SIKINTI ÇIKARIR
#FEATURE SCALİNG YAPILMALI
#İNBALANCE DATA VARSA SIKINTI OLUR

"""

# knn sınıflandırıcısını oluşturur
knn = KNeighborsClassifier(n_neighbors =2)
# sınıflandırıcıyı x_train verisi ve y_train target verisi ile fit edder
knn.fit(X_train,Y_train)
# X_test verisi ile prediction gerçekleştirir
y_pred = knn.predict(X_test)
# Y_test i ve y_pred i karşılaştırarak bir confusion matris çıkartır
cm = confusion_matrix(Y_test, y_pred)
# accuracy score unu belirler
acc = accuracy_score(Y_test,y_pred)
# score değerini gösterir
score = knn.score(X_test,Y_test)
print("score: ",score)
print("CM: ",cm)
print("Basic KNN Acc: ",acc)

# score:  0.9532163742690059
# [[108   1] 108 doğru tahmin 1 yanlış tahmin
#  [  7  55]] 55 doğru tahmin 7 yanlış tahmin
# Basic KNN Acc:  0.9532163742690059

#%% KNN Tune (En iyi parametre bulma)



























