import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
df=pd.read_csv('Mall_Customers.csv')
df=df.drop(columns=['Gender','CustomerID'])
sc=StandardScaler()
df=pd.DataFrame(sc.fit_transform(df),columns=df.columns)


df=df.drop(columns='Age')
krange=range(1,11)
sse=[]
for k in krange:
    model = KMeans(n_clusters=k)
    model.fit_predict(df)
    sse.append(model.inertia_)

sns.scatterplot(x=list(krange),y=sse)
plt.show()
