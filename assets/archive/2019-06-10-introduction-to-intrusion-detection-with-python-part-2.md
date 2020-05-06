---
layout: post
title:  "Introduction To Intrusion Detection With Python - Unsupervised Learning"
date: "2019-06-10"
author: "Ruth Ikwu"
comments: true
---

## Introduction
Clustering is a common form of unsupervised learning, employed to find patterns of relationships in unlabelled data. Unlabelled data is data that is not categorised into groups or where groups of records are not known. The task is to group a set of objects in such a way that objects in the same group (a cluster) are similar-in some sense-to each other than objects in other groups.

## Table of Content
* TOC
 {:toc}

The task of 'grouping' or 'clustering' objects is implemented by various algorithms that differ in their understanding of what makes up a 'cluster'. Basically, most clustering algorithms try to determine 'closeness’, ‘similarity' or ‘dissimilarity’ between objects, based on certain characteristics. The distance and similarity measures are the basis for constructing clustering algorithms.

Based on ‘similarity’ is determined, clustering algorithms are narrowly grouped into connectivity-based clustering, centroid or partition-based clustering and density-based clustering.

K-Means is a partition-based clustering algorithm. Partition-based clustering algorithms divides data into subsets based on some similarity or distance measure. It iteratively assigns each point in the dataset to those within a cluster most similar to it and most dissimilar from it. Kmeans is commonly used in intrusion detection because it provides a simple analysis of clustered data with less complexity.


```python
##load data
from os import chdir
from pandas import Series, DataFrame, read_csv, concat, concat, get_dummies, crosstab
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from random import seed

## set random seed for reproducibility
seed(1000)
```


```python
##specify data directory
datadir = "data"
```


```python
##lets read in our data
with open("data/target.txt","r+") as ff: trainy = Series([i.strip() for i in ff.readlines()])
trainx=read_csv("data/predictors.csv")
trainx.drop("Unnamed: 0", axis=1, inplace=True)
```

As KMeans algorithm does not work well on non-numeric data, we may either subset our data to only continuous features or transform categorical features to continuous features. Subsetting our data to only continuous features may potentially loose us some useful information. To convert categorical features to numeric features, we use a method called 'One-Hot Encoding'.

## One-Hot Encoding

OHE (one-host encoding) is used to indicate the state of a feature that can be in exactly one of a finite number of states at any given time. For example, for a simple a binary encoding, a single record can either be a Yes or a No for a certain feature. If the values 'Yes' and 'No' are featurized and we decided to assign a 1 to yes, if yes (therefore 0 to no) or a 1 to no if no (and therefore a 0 to yes) for each record in a dataset. Read More Here.

The function 'ohe' below takes a dataframe and performs one hot encoding by featuring all categorical features.


```python
#one hot encoding
def ohe(data):
    ##selects only columns with categorical variables
    categories=[i for i in data.columns if str(np.dtype(data[i]))=='object']
    data_copy = data.copy()

    for i in categories:
        #deal with binary variable encoding
        if len(data_copy[i].unique()) == 2:
            x,y = list(data_copy[i].unique())
            data_copy[i] = [0 if i == x else 1 for i in data_copy[i]]
            print("0 is "+x+": 1 is "+y)
        else:
            ##drops all categorical columns
            data_copy.drop(i,axis=1, inplace=True)

    ##encodes categorical features    
    data_copy = concat([get_dummies(data[categories]), data_copy], axis=1)
    return data_copy
```


Now since we are only interested in our attack traffic, we create a mask to subset only attack data.


```python
#split normal and attack traffic

##creates a mask that assigns the value of True to every value where trainy is 'normal' and False otherwise
mask = trainy == "normal."

##use mask to filter our normal traffic
normal_traffic = trainx[mask]

##use mask to filter out attack traffic
attack_traffic = trainx[~mask]
```

Let's 'hot-encode' our attack data subset


```python
from sklearn.cluster import KMeans

##clustering attack_traffic
attack_ohe = ohe(attack_traffic)
```


```python
attack_ohe.shape
```




    (396743, 114)



## Determining K

One of the very 'studied' and 'discussed' concepts in K-Means clutering is 'How to determine the number of clusters'. Afterall, the idea of unsupervised learning is that we do not know, prior, what patterns exist in the data. So how do we determine 'prior' what number of groups records may have based on the inter-relationship between them? Unfortunately, there is no blanket solution to this (yet....). The correct (ish) answer to this will always be subjective. However, there a number of ways to determine the optimal number of clusters in an unlabelled dataset.

### Through Dendrograms
One way is to first use a Hierarchical clustering algorithm to visualize the dendogram. These are particularly useful as they are not dependent on the need for a prior number of clusters. Rather, a simple distance matrix is used to construct a taxonomy tree. Hierarchical clustering algorithms can be [divisive](https://www.datanovia.com/en/lessons/divisive-hierarchical-clustering/) or [agglomerative](https://www.datanovia.com/en/lessons/agglomerative-hierarchical-clustering/).

Now, it is extremely unwise to attempt to estimate a distance matrix with 396743 records and 115 features, we will use a simple subset of the first 100 rows to demonstrate this dendogram.


```python
temp=attack_ohe.head(100)
```


```python
from scipy.cluster.hierarchy import dendrogram, linkage
attack_dm = linkage(temp, 'ward') ###'ward' is the distance algorithm on which clusters will be determined
```


```python
plt.figure(figsize=(10,7))
plt.title("Dendogram of Clustered Network Connections")
plt.axhline(y=2000, color='brown')
dendrogram(attack_dm)
plt.show()
```


![png](/assets/images/kdd2/dendrogram.png)

### Visualizing Within Cluster Sum of Squares
Another useful trick is to test a number of clusters from 2-N and select the number of clusters where the intra-cluster variation (total within sum of squares) is greatly minimized. Plotting the total within sum of squares for each cluster will create an elbow at the lowest point of variation. This is called an elbow.

Here we simply run kmeans algorithm iteratively for number of clusters (1:10) and plot for elbow where the total within sum of squares flattens. If our KMeans works correctly we should get an elbow somewhere around 4 clusters. See task description as attacks fall into four categories:

* DOS: denial-of-service, e.g. syn flood;
* R2L: unauthorized access from a remote machine, e.g. guessing password;
* U2R:  unauthorized access to local superuser (root) privileges, e.g., various buffer overflow attacks
* probing: surveillance and other probing, e.g., port scanning.

Let's define a function to do this


```python
def get_k(data, clusters):
  wss =[]
  for i in range(2,clusters+1):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=100, n_init=10)
    kmeans.fit(data.values)
    wss.append(kmeans.inertia_)
  return wss
```

The get_k function takes a data matrix and maximum number of clusters 'C' to test for. It runs the kmeans algorithm clusters-1 times for each cluster number from 2 through 'C'.


```python
wses =[]
for i in range(1,2):
    wses.append(get_k(attack_ohe, 10))
```


```python
fig, ax = plt.subplots()
for ss in wses:
    ax.plot(range(2,len(ss)+2),ss, color='purple')
plt.title("Within Sum Of Squares for Clusters")
plt.xlabel("Clusters")
plt.ylabel("WSS")
plt.show()
```


![png](/assets/images/kdd2/wss.png)


We see elbow just where we expect it to be-at 4 clusters. So let's build an actual k-means model with 4 clusters.

### Building the KMeans Model

```python
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
model = kmeans.fit(attack_ohe)
clusters = list(model.predict(attack_ohe))
centers = model.cluster_centers_
```

We plot the percentage of connections that ended up in each cluster.


```python
vc = DataFrame(Series(clusters).value_counts(), columns=['Freq'])
vc['normed'] = (vc['Freq']-vc['Freq'].mean())/np.std(vc['Freq'])
vc['percentage'] = (vc['Freq']/vc['Freq'].sum())*100
```


```python
sns.set(style="whitegrid")

bplot = sns.barplot(x=vc.index, y=vc['normed'], ci=.95)

for index, row in vc.iterrows():
    bplot.text(row.name,row.normed, round(row.percentage,3), color='black', ha="center")
```


![png](/assets/images/kdd2/cbplot.png)


We see that majority (99%) of our attack traffic belong to one type of attack.

Using unsupervised learning, we are able to create segments of connections that are similar. 'Similar' based on what attack features constitutes those types of attacks. Let's add this new clusters back the original dataset as a new predictor. We will assign all our normal traffic to a new cluster, cluster '4' and all other attack traffic as grouped by our clustering algorithm (0 through 3).


```python
cc = DataFrame(np.full((len(clusters),),4))
```


```python
attack_traffic['cluster'] = clusters
normal_traffic['cluster'] = 4
```


```python
clustered=concat([normal_traffic,attack_traffic], axis=0,ignore_index=False)
clustered=clustered.sort_index()
clustered.to_csv("data/clustered.csv")
```

## Visualizing Cluster Characteristics

What features actually characterize our clusters? We can build a simple decision tree with the clusters as our target variable to see how the clusters are determined.


```python
from sklearn.tree import DecisionTreeClassifier  
from sklearn.metrics import classification_report, confusion_matrix
classifier = DecisionTreeClassifier()  
```


```python
clusters2 = ["Cluster "+str(i) for i in clusters]
```

Since we are only interested in visualizing attributes of a cluster, we would skip the splitting part of model building. We will simply train our model on the entire training set.


```python
classifier.fit(attack_ohe, clusters2)
```




    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, presort=False,
                           random_state=None, splitter='best')



Now let's plot a decision tree using our clusters as the outcome feature.


```python
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
```

```python
dot_data = StringIO()
export_graphviz(classifier, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names=attack_ohe.columns)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
```




![png](/assets/images/kdd2/dstree.png)




```python
Series(clusters2).value_counts()
```




    Cluster 0    396668
    Cluster 2        59
    Cluster 3        15
    Cluster 1         1
    dtype: int64



We see the root node for cluster division is the number of bytes sent in a single connection. This is the defining feature of the various types of attacks in the data.
- Cluster 0 : connections where the number of bytes sent is less than or equal to 3,154,316.5 and the number of bytes recieved is also less than 3,815,741.0 bytes.
- Cluster 1 : connections where the of bytes sent is greater than 3,154,316.5 and the rate of 'SYN' errors on the destination host is greater than 17.5%.
- Cluster 2 : connections where the of bytes sent is greater than 3,154,316.5 and the rate of 'SYN' errors on the destination host is less than 17.5%.
- Cluster 3 : connections where the number of bytes sent is less than or equal to 3,154,316.5 and the number of bytes recieved is greater than 3,815,741.0 bytes.

Let's take a look at the mean of the total source and destination bytes and the 'SYN' error rate on the destination host.


```python
print("Mean of source bytes: ", attack_ohe['src_bytes'].mean())
print("Maximum number of bytes sent in a single connection: ", attack_ohe['src_bytes'].max())
```

    Mean of source bytes:  3483.7659517622237
    Maximum number of bytes sent in a single connection:  693375640.0



```python
print("Mean of source bytes: ", attack_ohe['dst_bytes'].mean())
print("Maximum number of bytes recieved in a single connection: ", attack_ohe['dst_bytes'].max())
```

    Mean of source bytes:  251.60160859800928
    Maximum number of bytes recieved in a single connection:  5155468.0



```python
print("Mean of source bytes: ", attack_ohe['dst_host_srv_serror_rate'].mean())
print("Maximum Rate of 'SYN' errors on destination host: ", attack_ohe['dst_host_srv_serror_rate'].max())
```

    Mean of source bytes:  0.21944293913188134
    Maximum Rate of 'SYN' errors on destination host:  1.0


Clusters 1 and 3 are characterized by the number of source and destination bytes while clusters 0 and 2 are characterized by the number of source bytes and the rate of SYN errors on the destination host.


```python
fig, axs = plt.subplots(3,1, figsize=(15, 10))
axs = axs.ravel()

axs[0].plot(attack_ohe['src_bytes'].diff(),color='orange')
axs[0].set_title("Number Of Connection Source Bytes")
axs[0].set_xlabel("Connection Index")
axs[0].set_ylabel("Source Bytes")

axs[1].plot(attack_ohe['dst_bytes'].diff(),color='orange')
axs[1].set_title("Number Of Connection Destination Bytes")
axs[1].set_xlabel("Connection Index")
axs[1].set_ylabel("Destination Bytes")

axs[2].plot(attack_ohe['dst_host_srv_serror_rate'].diff(),color='orange')
axs[2].set_xlabel("Connection Index")
axs[2].set_ylabel("Source Bytes")

plt.tight_layout()
```


![png](/assets/images/kdd2/cluster_determinants.png)


In the next artcle we would go into feature selection to reduce the number of features in the data to only those we really need to seperate good and bad connections. In article 4, we build a classification model to identify good or bad connections using these clusters and compare how the prediction performance is improved by adding these clusters.
