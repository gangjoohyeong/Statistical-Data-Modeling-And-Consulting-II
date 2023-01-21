#모듈
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, accuracy_score, confusion_matrix


data=pd.read_csv('C:/Users/user/Desktop/team/SAheart.txt')
data=data.iloc[:,1:]
data.head()

plt.figure(figsize=(12,10))
sns.heatmap(data.corr(), annot=True, cmap='Blues')
plt.title('South African Heart Disease Correlation')
plt.show()

data.loc[data['famhist']=='Present','famhist']=1
data.loc[data['famhist']=='Absent','famhist']=0
pd.DataFrame(data['famhist'].value_counts()).T 

plt.figure(figsize=(16,7))
for idx, col in enumerate(data.columns): 
    plt.subplot(2, 5, idx + 1)
    sns.histplot(data[col], color='lightgreen', bins=13)
    plt.ylabel('')
plt.show()

data.isnull().sum()

data.describe()

plt.figure(figsize=(16,7))
for idx, col in enumerate(data.drop(columns=['chd','famhist']).columns): 
    plt.subplot(2,4, idx + 1)
    sns.boxplot(y=data[col], color='skyblue')
    plt.xlabel(data.drop(columns=['chd','famhist']).columns[idx])
    plt.ylabel('')
plt.show()


df=data.copy()
df['sbp_class1']=90000
df.loc[(data['sbp']<140), 'sbp_class1']=0
df.loc[(data['sbp']>=140), 'sbp_class1']=1
pd.DataFrame(df['sbp_class1'].value_counts()).T



df['sbp_class2']=10000
df.loc[(data['sbp']<140), 'sbp_class2']=0
df.loc[(data['sbp']>=140) & (data['sbp']<160), 'sbp_class2']=1
df.loc[(data['sbp']>=160) & (data['sbp']<180), 'sbp_class2']=2
df.loc[(data['sbp']>=180), 'sbp_class2']=3
pd.DataFrame(df['sbp_class2'].value_counts()).T



df['smoker']=4000
df.loc[(data['tobacco']==0), 'smoker']=0
df.loc[(data['tobacco']>0), 'smoker']=1
pd.DataFrame(df['smoker'].value_counts()).T


df['drinker']=3
df.loc[(data['alcohol']==0), 'drinker']=0
df.loc[(data['alcohol']>0), 'drinker']=1
pd.DataFrame(df['drinker'].value_counts()).T

df['ldl_class']=90
df.loc[(data['ldl']<3.36), 'ldl_class']=0
df.loc[(data['ldl']>=3.36), 'ldl_class']=1
pd.DataFrame(df['ldl_class'].value_counts()).T

df['obesity_class']=5
df.loc[(data['obesity']<25), 'obesity_class']=0 
df.loc[(data['obesity']>=25) & (data['obesity']<30), 'obesity_class']=1  
df.loc[(data['obesity']>=30), 'obesity_class']=2  
pd.DataFrame(df['obesity_class'].value_counts()).T


df['adiposity_class']=100
df.loc[(data['adiposity']<30), 'adiposity_class']=0
df.loc[(data['adiposity']>=30), 'adiposity_class']=1
pd.DataFrame(df['adiposity_class'].value_counts()).T 


df['typea_class']=9
df.loc[(data['typea']<53), 'typea_class']=0
df.loc[(data['typea']>=53), 'typea_class']=1
pd.DataFrame(df['typea_class'].value_counts()).T 


df['age_class']=6
df.loc[(data['age']<20), 'age_class']=0 
df.loc[(data['age']>=20) & (data['age']<40), 'age_class']=1
df.loc[(data['age']>=40) & (data['age']<60), 'age_class']=2
df.loc[(data['age']>=60), 'age_class']=3
pd.DataFrame(df['age_class'].value_counts()).T 


x = data[['adiposity', 'obesity']]
scaler = MinMaxScaler()
x = scaler.fit_transform(x)

from sklearn.decomposition import PCA
pca = PCA(n_components=1)
printcipalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data=printcipalComponents, columns = ['PC1'])
df['pca']=principalDf



df['body_shape']=0 
df.loc[(data['obesity']<25) & (data['adiposity']>=30), 'body_shape']=1 
df.loc[(data['obesity']>=25) & (data['obesity']<30) & (data['adiposity']>=30), 'body_shape']=2 
df.loc[(data['obesity']>=30) & (data['adiposity']>=30), 'body_shape']=3 
pd.DataFrame(df['body_shape'].value_counts()).T 


#accuracy가 더 떨어져서 body_shape와 pca변수는 제거
df2=df.drop(columns=['body_shape','pca']); df2.head() 


df3=df2.iloc[:,:9]
df3=df3.drop(columns='famhist')
df3.head()



#스케일링
scaler = MinMaxScaler()
Min_Max_scaled = scaler.fit_transform(df3)

scaler2=StandardScaler()
Standard_scaled = scaler2.fit_transform(df3)

scaled1=pd.DataFrame(data=Min_Max_scaled, columns=df3.columns)
scaled2=pd.DataFrame(data=Standard_scaled, columns=df3.columns)

final1=scaled1.join(df2['famhist']); final1=final1.join(df2.iloc[:,9:])
final2=scaled2.join(df2['famhist']); final2=final2.join(df2.iloc[:,9:])

#vif 이후 sbp_class2 drop => R에서 확인 
#정규 스케일링 보다 최대최소 스케일링이 더 좋아서 final1 선택

final1=final1.drop(columns='sbp_class2')

#데이터 분할 (7:3, seed=42)
x=final1.drop(columns='chd')
y=final1['chd']

x_train, x_test, y_train, y_test=train_test_split(x, y, random_state=42, test_size=0.3, stratify=y)
print(x_train.shape); print(y_train.shape)
print(x_test.shape); print(y_test.shape)
print('train data의 sqrt 값: ', round(np.sqrt(323),2))

#kNN
k_acc=[]
k_auc=[]
for i in range(3,26,2):
    KNN=KNeighborsClassifier(n_neighbors=i)
    
    scores = cross_val_score(KNN, x_train, y_train, cv=10, scoring = "accuracy")    
    scores2 = cross_val_score(KNN, x_train, y_train, cv=10, scoring = "roc_auc")

    k_acc.append(round(scores.mean(),4))
    k_auc.append(round(scores2.mean(),4))

neighbors_settings=range(3,26,2)

plt.figure(figsize=(12,12))
plt.subplot(2,1,1)
plt.plot(neighbors_settings, k_acc, 'o-',label="accuracy")
plt.plot(neighbors_settings, k_auc, 'o-',label="AUC")
plt.ylim(0.5,0.9)
plt.ylabel('')
plt.xlabel("n_neighbors")
plt.legend()
plt.grid(True)
#plt.savefig('C:/Users/user/Desktop/team/plot3.png', dpi=100, facecolor='w')
plt.show()


KNN_17=KNeighborsClassifier(n_neighbors=17)
KNN_19=KNeighborsClassifier(n_neighbors=19)

KNN_17.fit(x_train, y_train)
pred1=KNN_17.predict(x_test)

KNN_19.fit(x_train, y_train)
pred2=KNN_19.predict(x_test)

print('--- 70:30 17NN ---')
print('accuracy:', round(accuracy_score(y_test, pred1), 4), end = ' / ')
print('recall:', round(recall_score(y_test, pred1), 4), end = ' / ')
print('precision:', round(precision_score(y_test, pred1), 4), end = ' / ')
print('f1 Score:', round(f1_score(y_test, pred1), 4), end = ' / ')
print('AUC Score:', round(roc_auc_score(y_test, pred1), 4))
print()

print('--- 70:30 19NN ---')
print('accuracy:', round(accuracy_score(y_test, pred2), 4), end = ' / ')
print('recall:', round(recall_score(y_test, pred2), 4), end = ' / ')
print('precision:', round(precision_score(y_test, pred2), 4), end = ' / ')
print('f1 Score:', round(f1_score(y_test, pred2), 4), end = ' / ')
print('AUC Score:', round(roc_auc_score(y_test, pred2), 4))
print()

##나이브 베이즈
NB = GaussianNB()
NB.fit(x_train, y_train)
pred3=NB.predict(x_test)


print('--- 70:30 NB ---')
print('accuracy:', round(accuracy_score(y_test, pred3), 4), end = ' / ')
print('recall:', round(recall_score(y_test, pred3), 4), end = ' / ')
print('precision:', round(precision_score(y_test, pred3), 4), end = ' / ')
print('f1 Score:', round(f1_score(y_test, pred3), 4), end = ' / ')
print('AUC Score:', round(roc_auc_score(y_test, pred3), 4))
print()
