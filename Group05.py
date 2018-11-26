# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 19:30:02 2018

@author: nikri
"""
#----------------------------------------------------------------------------------------
#THINGS WE USE
#----------------------------------------------------------------------------------------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import date

trab = pd.read_excel("C:\\Users\\nikri\\OneDrive\\Documentos\\NOVA IMS\\1st Semester\\Data Mining 1\\Assignment\\Group_05_SixPM_Cluster.xlsx")

#----------------------------------------------------------------------------------------
#STEP 1
#----------------------------------------------------------------------------------------

# Create a graph that shows the missing values
import missingno as msno
msno.matrix(trab,figsize=(12,5))


#FILL MISSING VALUES WITH MEAN (INTERVAL )
trab = trab.fillna(trab.mean())


#----------------------------------------------------------------------------------------
#STEP 2
#----------------------------------------------------------------------------------------

#CREATE NEW VARIABLES

#  1 - total amount of purchases for each customer
trab['MntTotal'] = trab['MntAcessories'] + trab['MntClothing'] + trab['MntBags'] + trab['MntAthletic'] + trab['MntShoes']

#  2 - total amount of regular products
trab['MntRegularProds'] = trab['MntTotal'] - trab['MntPremiumProds']

#  3 - Age
from datetime import date
trab['Age'] = date.today().year - trab['Year_Birth']

#  4 - total number of purchases for each customer
trab['Frequency'] = trab['NumCatalogPurchases'] + trab['NumStorePurchases'] + trab['NumWebPurchases']

#  5 - AVG money spent for each customer by purchase
trab['Avg_Mnt_Freq'] = trab['MntTotal'] / trab['Frequency']

#  6 - binary - 1: gradutation,master or PhD and 0: the rest
import numpy as np
trab['Higher_Education'] = np.where((trab['Education'] == 'Graduation') | \
    (trab['Education'] == 'Master') | (trab['Education'] == 'PhD'), 1, 0)

#  7 - total number of accepted campaigns
trab['totalAcceptedCmp'] = trab['AcceptedCmp1'] + trab['AcceptedCmp2'] + trab['AcceptedCmp3'] + trab['AcceptedCmp4'] + trab['AcceptedCmp5']

#  8 - number of years as a customer
trab['AgeAsCustomer'] = trab['Custid']
from datetime import datetime
for x,y in trab['Dt_Customer'].iteritems():
    y = date.today().year - y.date().year
    trab['AgeAsCustomer'] = trab['AgeAsCustomer'].set_value(x,y)

#  9 - Effort
trab['Effort'] = (trab['MntTotal'] / trab['Income']) * 100
# BELOW - JUST TO BUT AN IMAGE IN THE REPORT 
trab['Marital_Status_High_Effort'] = trab['Marital_Status']


#----------------------------------------------------------------------------------------
#STEP 3
#----------------------------------------------------------------------------------------

#CHECK COHERENCE

moneySpent = trab['MntTotal'] > 0
noFrequency = trab['Frequency'] ==0
aux  = trab['NumWebPurchases'].mean()
import numpy as np
trab['NumWebPurchases']=np.where(moneySpent & noFrequency, aux, trab['NumWebPurchases']) 


#----------------------------------------------------------------------------------------
#STEP 4
#----------------------------------------------------------------------------------------

#RECALCULATE VARIABLES USING FREQUENCY

trab['MntTotal'] = trab['MntAcessories'] + trab['MntClothing'] + trab['MntBags'] + trab['MntAthletic'] + trab['MntShoes']

trab['MntRegularProds'] = trab['MntTotal'] - trab['MntPremiumProds']

trab['Frequency'] = trab['NumCatalogPurchases'] + trab['NumStorePurchases'] + trab['NumWebPurchases']

trab['Avg_Mnt_Freq'] = trab['MntTotal'] / trab['Frequency']

trab['Effort'] = (trab['MntTotal'] / trab['Income']) * 100


#----------------------------------------------------------------------------------------
#STEP 5
#----------------------------------------------------------------------------------------

#VARIABLES STATISTICS
print(trab['Year_Birth'].describe())
print(trab['Income'].describe())
print(trab['Kidhome'].describe())
print(trab['Teenhome'].describe())
print(trab['Recency'].describe())
print(trab['MntAcessories'].describe())
print(trab['MntBags'].describe())
print(trab['MntClothing'].describe())
print(trab['MntAthletic'].describe())
print(trab['MntShoes'].describe())
print(trab['MntPremiumProds'].describe())
print(trab['NumDealsPurchases'].describe())
print(trab['NumWebPurchases'].describe())
print(trab['NumCatalogPurchases'].describe())
print(trab['NumStorePurchases'].describe())
print(trab['NumWebVisitsMonth'].describe())



#HISTOGRAMS
Marital_status = sns.countplot(trab['Marital_Status'])
Education = sns.countplot(trab['Education'])
AcceptedCmp1 = sns.countplot(trab['AcceptedCmp1'])
AcceptedCmp2 = sns.countplot(trab['AcceptedCmp2'])
AcceptedCmp3 = sns.countplot(trab['AcceptedCmp3'])
AcceptedCmp4 = sns.countplot(trab['AcceptedCmp4'])
AcceptedCmp5 = sns.countplot(trab['AcceptedCmp5'])


Income = sns.distplot(trab['Income'] )



#BOXPLOTS
import numpy
from scipy import stats

sns.boxplot(data=trab,x="Income",orient="v")
sns.boxplot(data=trab,x="MntAcessories",orient="v")
sns.boxplot(data=trab,x="MntBags",orient="v")
sns.boxplot(data=trab,x="MntAthletic",orient="v")
sns.boxplot(data=trab,x="MntShoes",orient="v")
sns.boxplot(data=trab,x="MntClothing",orient="v")
sns.boxplot(data=trab,x="MntPremiumProds",orient="v")


#----------------------------------------------------------------------------------------
#STEP 6
#----------------------------------------------------------------------------------------


#PCA
from sklearn.decomposition import PCA
from pandas import DataFrame

#CUMULATIVE PROPORTION OF VARIANCE EXPLAINED
prodUse = trab[['MntAcessories', 'MntClothing', 'MntBags', 'MntAthletic', 'MntShoes']]
pca = PCA(n_components = 5)
pca.fit(prodUse)
projected = pca.fit_transform(prodUse)

print('nComps', pca.components_)
print('explained' ,np.round( pca.explained_variance_,decimals = 4)*100)

var1= np.cumsum(pca.explained_variance_ratio_)
plt.xlabel('Principal Components')
plt.ylabel('Cumulative Proportion of Variance Explained')
plt.plot(var1)


#PCA NORMALIZED AND SIMPLIFIED

#Run this first
def z_score(pdusage):
    """Remove a média e normaliza-os pelo desvio padrão"""
    return (pdusage - pdusage.mean()) / pdusage.std()

pca = PCA(n_components=3)
pca.fit(prodUse.apply(z_score).T)

#loadings
loadings = DataFrame(pca.components_.T)
loadings.index = ['PC %s' % pc for pc in loadings.index + 1]
loadings.columns = ['TS %s' % pc for pc in loadings.columns + 1]
loadings

PCs = np.dot(loadings.values.T, prodUse)

marker = dict(linestyle='none', marker='o', markersize=7, color='blue', alpha=0.5)

fig, ax = plt.subplots(figsize=(7, 2.75))
ax.plot(PCs[0], PCs[1], label="Scores", **marker)
plt.grid(True)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")

text = [ax.text(x, y, t) for x, y, t in
        zip(PCs[0], PCs[1]+0.5, prodUse.columns)]


#----------------------------------------------------------------------------------------
#STEP 7
#----------------------------------------------------------------------------------------


#CORRELATION MATRICES
import seaborn as sns

corr = trab[['Income','Age', 'AgeAsCustomer','Avg_Mnt_Freq', 'Effort', 'Frequency', 'Teenhome', 'Kidhome', 'Recency', 'MntPremiumProds', 'MntRegularProds', 'MntTotal', 'NumDealsPurchases', 'totalAcceptedCmp']]
corr=trab[['MntAcessories', 'MntBags', 'MntClothing', 'MntAthletic', 'MntShoes']]


mask=np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)]=True

f, ax = plt.subplots(figsize=(11,9))
cmap=sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap =cmap, vmax=1, vmin=-1, center=0, square=True, linewidth=.5,cbar_kws={"shrink":.5})
f.savefig('myimage.png', format='png', dpi=1200)

corr = trab.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220,10,as_cmap=True), square=True, ax=ax)


#----------------------------------------------------------------------------------------
#STEP 8
#----------------------------------------------------------------------------------------


#SOM
import somoclu
import numpy as np

dataProd = trab[['MntAcessories', 'MntClothing', 'MntBags', 'MntAthletic', 'MntShoes']]
data = trab[['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']]
dataCustomer = trab[['Income','Age','Kidhome', 'Teenhome', 'NumDealsPurchases', 'Avg_Mnt_Freq', 'totalAcceptedCmp', 'AgeAsCustomer', 'Recency', 'Frequency', 'Effort', 'MntPremiumProds', 'MntRegularProds']]

df = np.float32(dataProd.values)
df = np.float32(dataChannel.values)
df = np.float32(dataCustomer.values)

rows, cols = 5, 8

som = somoclu.Somoclu(cols, rows, initialization = "pca", gridtype="rectangular", maptype="toroid")
som.train(df)

#U-MATRIX
som.view_umatrix()

#COMPONENT PLANES
som.view_component_planes()
