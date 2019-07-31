---
layout: post
title:  "Introduction To Intrusion Detection With Python - Modelling"
date: "2019-06-12"
author: "Ruth Ikwu"
---

## Introduction
Modelling attempts to build a blueprint for analysing data, from previously observed patterns in the data. Modelling is often predictive in nature in that it tries to use this developed ‘blueprint’ in predicting the values of future or new observations based on what it has observed in the past. Based on our question - ‘Can we separate bad traffic from good traffic?”-this is where we select a blueprint that best captures the nature of dynamics in our data.

* TOC
 {:toc}

The decision for the best blueprint is complex given the complexities of the real-world and personal definitions of ‘best’. In this case, we are simply interested in determining what distinguishes good connections from bad connections. Easy enough. However, there are multiple types of bad connections with distinguishing features that may not be common across all types. For example, we could simply classify all connections with source bytes less than 2809074. This will capture over 90% of bad traffic in our data. Unfortunately, our data is skewed with DDOS traffic and this could potentially work well in identifying DDOS traffic. What happens to the other types of bad traffic? But more importantly, will this model be able to identify a new bad connection that is not a DDOS connection?

The answer to the question ‘what is the right model?’ is never an easy decision and there is no blanket answer to it. The best model is dynamic and flexible to world change with the capacity for continuous learning and updating. The quality of a model is however highly dependent on the size of data, type of data, quality of your data, time and computational resources available. More observations mean your model is able to capture more of the variation in the real-world event you are attempting to computationally capture.

Methods for developing an appropriate model are different when the outcome feature is a nominal variable as opposed to a continuous one. Models predicting nominal features would be based on some type of classification algorithm. There are a couple of different [cheat sheets](https://docs.microsoft.com/en-us/azure/machine-learning/studio/algorithm-cheat-sheet) available online which have a flowchart that helps you decide the right algorithm based on the type of classification or regression problem you are trying to solve. Consider a single connection viewed in isolation from the others. This connection has characteristics i.e the IP address it comes from, the IP address it if song to, the size of the packet it is sending, the port it is connection to and the port it is receiving its message from etc. The simple question is, based on these characteristics and given a new connection, can we tell if it is a good or bad connection?

In this section, we would build a simple Logistic regression and Decision tree model and evaluate the performance based on different metrics of ‘performance’.

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from random import seed
seed(1000)
```

Read in elected features from previous tutorial.

```python
data = pd.read_csv(“data/features.csv”)
```

Read in target data

```
with open(“data/target.txt”, “r+”) as ff: target = [i.strip() for I in ff.readlines()]; ff.close();
y = [“Good” if I==“normal.” else “Bad” for I in target]
data.drop(“Unnamed: 0”,axis=1, inplace=True)
```

```python
fig = plt.figure(figsize=(20,8))
sns.countplot(y)
plt.title(“Training Data Distribution”)
plt.show()
```

![![Training Data Class Distribution](/assets/images/kdd4/TrainingDataClassDistribution.png)

We can see that the traffic is dominated by attack traffic in both the training and evaluation sets. This is an imbalanced class distribution and therefore poses a peculiar problem for our classification efforts. This happens when the number of observations in one class is significantly higher than the number of observations in other classes.

This is a tricky problem. On one hand, this data is not ‘complete’ in representing the ‘real-world’ analytical domain-as a network would get significantly more normal traffic than attack traffic. On the other hand, if we flipped this graph, therefore reducing the amount of ‘attack’ traffic and increasing the amount of ‘normal’ traffic , we stand the risk of loosing otherwise useful information (since the idea is to identify attacks).

In addition to this domain mis-representation, there is the issue of inbalanced representation of classes as mentioned above. Machine learning algorithms end up treating events in the minority class as rare events by treating them as noise rather than outliers.

We are therefore left with data that is not ‘complete’ in its representation of the real-world problem of interest and inbalanced for machine learning algorithms. This problem is relevant as analyst would be particularly interested in identifying ‘real’ actionable anomalies while reducing the probability of false positives (or panic attacks).

First let’s estimate the event rate for each class in our data.

```python
xx=pd.Series(y).value_counts()
print(xx[0]-xx[1], “ more attack traffic”)
```

    299465  more attack traffic

```python
attack_event_rate, normal_event_rate = round((pd.Series(y).value_counts()/len(y))*100,2)
print(“Attack Event Rate: “, attack_event_rate,”%”)
print(“Normal Event Rate: “, normal_event_rate,”%”)
```

    Attack Event Rate:  80.31 %
    Normal Event Rate:  19.69 %


There a number of ways to address this problem, however the simplest way is to balance out the data with more observations from the minority class. We could take this further to skew the data in favour of normal traffic-therefore the data is ‘completely’ representative. However, our ML model will treat attacks as ‘rear events or noise’ rather than outliers.

The best way around this is to simply balance out the data. To do this, So I have randomly sampled of 299465 normal traffic observations from the [complete dataset](http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz). The real test for whether this is a good trade-off for data representation would be the performance of models’ expost predictions. We will simply read in this data and add it to our current data set, since the chronological order of connections is not relevant in this scenario.


```python
balanced = pd.read_csv(“data/balance.csv”)

##join target from balanced to current target vector
target_bal = [“Good” for I in balanced[‘target’]]
to_balance = y+target_bal

##make outcome feature
outcome = [0 if I ==“Good” else 1 for I in to_balance]

##drop target vector from balanced dataset
balanced.drop(“target”, axis=1, inplace=True)
balanced.drop(“Unnamed: 0”, axis=1, inplace=True)

###add cluster 4 to new attack traffic refer to tutorial 2 (all normal traffic are assigned to cluster 4)
balanced[‘cluster’] = 4

##concat data (row bind) and balanced
newd = pd.concat([data,balanced], axis=0)
```


```python
sns.countplot(to_balance)
plt.title(“Distribution of Class Observations”)
```

    Text(0.5, 1.0, ‘Distribution of Class Observations’)

![Distribution Of Class Observations](/assets/images/kdd4/DistributionOfClassObservations.png)


Now we have a balanced dataset, where each class is equally represented, we can move on to building a good model. First, let’s add our clusters from our unsupervised learning task to our predictor set.


```python
clusters = newd[‘cluster’]
```

We can also visualise the cluster distribution of observations. Most of our observations belong to cluster 0 [(variations of DDOS attacks that make up over 90% of our attack traffic) and Cluster 4 (our normal traffic)](https://eneyi.github.io/2019/06/09/Introduction-to-intrusion-detection-with-python-part-1.html).


```python
sns.countplot(clusters)
```


![Cluster Plot 1](/assets/images/kdd4/Clusterplot1.png)


Next we can also examine within class means of the various predictors.


```python
newd['target'] = to_balance

##drop cluster feature from old clustering task
newd.drop("cluster", axis=1, inplace=True)

newd.groupby("target").mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dst_host_count</th>
      <th>protocol_type_icmp</th>
      <th>service_http</th>
      <th>service_smtp</th>
      <th>service_urp_i</th>
    </tr>
    <tr>
      <th>target</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bad</th>
      <td>253.056722</td>
      <td>0.711579</td>
      <td>0.006067</td>
      <td>0.000315</td>
      <td>0.000003</td>
    </tr>
    <tr>
      <th>Good</th>
      <td>148.443700</td>
      <td>0.013134</td>
      <td>0.636359</td>
      <td>0.098479</td>
      <td>0.005404</td>
    </tr>
  </tbody>
</table>
</div>



We can observe greater mean differences in the features 'protocol_type_icmp', 'dst_bytes', 'service_http' and 'service_smtp'. Now, we will drop the target variable from the feature set and build our classifiers.


```python
newd.drop("target", axis=1, inplace=True)
```

Let's make sure our features are represented in the correct format for modelling.


## Unsupervised Learning

We looked at unsupervised learning in the [second article](https://eneyi.github.io/2019/06/10/introduction-to-intrusion-detection-with-python-part-2.html) of this series. Unsupervised learning is a method employed to find patterns prior unknown in unlabelled data. Unlabelled data is data for which the observations belong to no prior known group. The goal of unsupervised learning is to capture the pattern of variation in the data such that observations in the same group (a cluster) are similar-in some sense-to each other than observations in other groups.

### K-Means Clustering

In the previous tutorial, we assumed no groups to our attack (bad) traffic data and applied unsupervised learning to capture the various types of attacks in our 'bad' traffic. As expected, our clustering task returned 4 clusters similar to the [task description](http://kdd.ics.uci.edu/databases/kddcup99/task.html). Our previous clustering task was done with all features for just the attack traffic.

Here, we apply similar clustering technique to our selected feature subset with all traffic (good and bad). To do this let's import our 'get_k' function to find the appropriate number of clusters given a dataset.


```python
from Modules.get_k import get_k
from sklearn.cluster import KMeans
```

We run 9 iterations of Kmeans clustering algorithm and plot the within sum of squares for each iteration. First, we would make a copy of our dataset and cask all features as floats, as the Kmeans algorithm requires numeric data.


```python
cdata = newd.apply(lambda x: x.astype(float))
```


```python
wses =[]
for i in range(1,2):
    wses.append(get_k(cdata, 10))
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


![Elbow for wss](/assets/images/kdd4/wss.png)


We see the elbow at 3 or 4 clusters. We will construct our Kmeans model with 4 clusters and assign the predicted clusters to observations in our data.


```python
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
model = kmeans.fit(cdata)
clusters = list(model.predict(cdata))
newd['clusters'] = clusters
```


```python
sns.countplot(newd.clusters)
plt.title("Cluster Distribution")
plt.xlabel("Clusters")
plt.ylabel("Frequency")
```
    Text(0, 0.5, 'Frequency')

![Cluster Distirbution](assets/images/kdd4/ClusterDistribution.png)

## Classification

Classification is simply the art of putting things into the appropriate group to which they belong. Classification in machine learning seeks to mimic how children learn-understand and group animate objects based on similar characteristics. Classification is a form of [supervised](https://www.geeksforgeeks.org/regression-classification-supervised-machine-learning/) learning, where we know the nature of relationships that exist and the groups to which a sample from our observed data belongs. Our task is simply to identify which of these finite number of groups, a new observation belongs to.

A binary classification problem is when the number of finite group to which new observations (k) can belong is 2. A multi-class classification means K > 2. Like our problem, it is a binary classification as packets can only be 'good' or 'bad'. The goal of classification is to build a concise model of the distribution of class labels in terms of
predictor features. The resulting classifier is then used to assign class labels to the testing instances
where the values of the predictor features are known, but the value of the class label is unknown.

Classifiers fall under one of the following groups:
- Logic-Based Classifiers
    - Decision Trees
    - Rule-Based Classifiers
- Classifiers based on Statistical Learning
    - Naive Bayes
    - Bayesian Networks
- Perceptron-Based Classifiers
    - Artificial Neural Networks
    - Convolutionary Neural Networks
- Instance-Based Classifiers
    - K-Nearest Neighbours (KNN)
- Support Vector Machines

The process for training and choosing a model includes the following steps:

- Split the input data randomly for modelling into a training data set and a test data set.
- Build the models by using the training data set.
- Evaluate the training and the test data set. Use a series of competing machine-learning algorithms along with the various associated tuning parameters (known as a parameter sweep) that are geared toward answering the question of interest with the current data.
- Determine the “best” solution to answer the question by comparing the success metrics between alternative methods.

Let's split our data into two, 80% for training the and 20% for evaluating the model.


```python
objs = list(newd.columns[1::])
newd[objs]=newd[objs].apply(lambda x: x.astype(object))
```


```python
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(np.array(newd,dtype="float"), np.array(outcome,dtype="float"), test_size=0.20)
```

Let's see the class distribution of observations within our training and evaluation sets.


```python
fig = plt.figure(figsize=(20,8))
ax=fig.add_subplot(1,2,1)
sns.countplot(y_train)
ax.set_title("Training Data Distribution")

ax=fig.add_subplot(1,2,2)
sns.countplot(y_test)
ax.set_title("Evaluation set Distribution")
```

    Text(0.5, 1.0, 'Evaluation set Distribution')

![Train And Test Set Data Distribution](/assets/images/kdd4/tandtclassDistribution.png)


Now we will build different classification models. Note, 0 represents 'an absence of the event of interest, i.e a Good Connection" and 1 represents "a presence of the event of interest, i.e a Bad connection".

### Logistic Regression

Logistic regression is a linear method for classification based on specifying a decision boundary. The decision boundary between the two classes is a single line through the feature vector space.
The logistic regression classifier is suited when a single regression line is sufficient to define a class boundary between the two classes. The algorithm transforms its outputs using the sigmoid function to return the probability of an event occurring given the observed features. The algorithm is a discriminative modelling approach, where the estimated posterior probabilities determines the class of the observation.

In the case of our binary-classed Logistic regression, we would be interested in the probability that a connection is 'bad' or an 'attack' given the variation observed in our selected features. Say we wanted to identify good (0) and bad (1) connections using only two of our features, (any two). We can graphically test if a straight line is suitable to divide the good and bad connections.


```python
predictors = pd.read_csv("data/predictors.csv")
pairs = predictors[["dst_bytes","src_bytes","dst_host_same_src_port_rate"]]
pairs['target'] = ["Normal" if i=="normal." else "Attack" for i in target]
sns.pairplot(pairs,hue="target", palette="husl")
```

![Logistic Regression Grid](lr_grid.png)


From this graph, If we were to select only two features from our feature set that clearly divides Good and Bad connections, they would be the 'dst_bytes' and 'dst_host_same_src_port_rate". This line may not do well is distinguishing 'Good Connections' but would identify most "Bad Connections". Let's build a multiple linear regression model with our feature set.


```python
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()

##fit model
LR.fit(X_train, y_train)

##test data prediction
LR_expost_preds = LR.predict(X_test)
```

### Decision Trees

Let's consider another scenario in identifying 'Good' and 'Bad' connections where a linear representation for class separation may not be enough.

Binary Decision Tree Problem             |  Multi-class Decision Tree Problem
:-------------------------:|:-------------------------:
<img src="images/DecisionTreeIntuition2.png" style="width:300px;height:250px;"/> | <img src="images/DecisionTreeIntuition.png" style="width:300px;height:250px;"/>

Why decision trees? Imagine the images (a) and (b) above, where a single line through the graph is not enough to properly separate the different classes. In (a) we need at least 3 lines to separate the classes while in (b) we need at least 2 lines to separate the classes. Therefore, a linear relationship between features cannot represent the separation between classes. Decision trees are one of the basic building blocks of the analytics process and finds its way into majority of data science workflows. They are like 'conditional' flow statements to reach a certain decision.

Cognitively, we all use mental decision trees regularly in our daily lives. Look at a simple task of deciding what restaurant to eat at during launch.



![Lunch Decision](images/ResturantDecision.png)


The most important criteria for deciding where to eat is its walking distance from work. This is called the root node (the criteria on which everything else depends). In decision tree terms, each circle is called a 'node' with the topmost circle as the 'root' node and all other circles as 'leaf' nodes. The root node, is the criterion on which everything else depends.

The question then is how do we decide what attribute to make the root node, what attributes to create leaf nodes (split) from and on what conditions to split on? Well, for us humans, we make a simple logical decision based on our experience of the real world around us. Computers however need a way to computationally decide what is most important in fulfilling their decision task.

These split decisions are made by deciding how much information an attribute gives us about a class. Attributes are split in descending order of the information they contribute to the model. So the attribute with the highest information gain would be the root node. This information gain is derived by estimating the amount of random variation (entropy) in an attribute.

Entropy, in information theory by [Shanon](https://onlinelibrary.wiley.com/doi/abs/10.1002/j.1538-7305.1948.tb01338.x) is the amount of information conveyed by each attribute to the determination of a class.

Given two bags of a 100 balls each. Bag A contains 100 red balls and bag B contains 50 red balls and 50 blue balls. You are asked to use your prior knowledge of the colour of balls in these bags to transfer the balls into a red-ball bag and a blue-ball bag. The entropy of the system Bag A (with 100 red balls) is 0. Because we know that every ball picked will be a red ball; no surprises nothing interesting or unexpected. Therefore, Bag A contributes no new information that helps us decide what differenciates a red ball from a blue ball. On a decision tree, we can construct a leaf node that simply dumps all balls in Bag A in the red-ball bag.

Bag B, on the other hand,  gives us some new information, because we now know what a blue ball looks like and how it is different from a red ball. Most importantly, we can make this differentiation within Bag B (independently). Additionally, there is an equal amount of blue and red balls, so balls are evenly distributed between both classes.

There are many [algorithms](https://books.google.co.uk/books?hl=en&lr=&id=vLiTXDHr_sYC&oi=fnd&pg=PA3&dq=review+of+decision+trees+algorithms&ots=CYquvtXCmo&sig=USmgs97gMwCTn0ZHbjsH5Z0uVdk#v=onepage&q=review%20of%20decision%20trees%20algorithms&f=false) for constructing decision trees, but here we would use the most basic implementation using python's [SK-Learn](http://scikit-learn.org/stable/) library.



```python
from sklearn import tree
###create decision tree classifier object
DT = tree.DecisionTreeClassifier(criterion="entropy")

##fit decision tree model wth training data
DT.fit(X_train, y_train)

##test data prediction
DT_expost_preds = DT.predict(X_test)
```

### Evaluating Model's Performance

Finally, here we would measure the performance of the models we developed earlier. Evaluation is testing that our model does, to the best it can, what it was developed to do. There are various performance metrics used to evaluate classification models. In this post, we will apply the classification accuracy, recall, precision and F1 Scores for evaluating binary classification models. But before we begin evaluating, we must visit the concept of a **'confusion matrix'**.

A confusion matrix is simply a cross-tabulation of our predicted classes against the actual class for each observation. Therefore, it tells us:

- How many ‘good’ connections our model predicted as ‘good’ **(True Positives or TPs)**

- How many ‘bad’ connections our model predicted ‘bad’ **(True Negatives or TNs)**

- How may ‘good’ connections our model predicted as ‘bad’ **(False Positives or FPs or Type I Errors or False Alarms)** and

- How may ‘bad’ connections our model predicted as ‘good’ **(False Negatives FNs or Type II Errors or Misses)**

Where
 - A condition Positive : A case of a bad connection

 - A condition Negative : A case of a good connection

 <br>
 <br>

 From the confusion matrix, a number of performance metrics can be derived.

 - **Accuracy**: The overall ability of a model to get predictions right. $\frac{TPs}{TNs}$

 - **Precision**: The ability of the model to identify only attack classes. $\frac{TPs}{TPs + FPs}$

 - **Recall** : The ability of the model to identify all attack classes. $\frac{TPs}{TPs + FNs}$

 - **F1 Score** : The harmonic mean of precision and recall. $ 2* \frac{precision * recall}{precision + recall}$

 <br>
 <br>

The ‘Accuracy’ is a general form of evaluation that measures , on the average, the model’s ability to identify both bad and good connections. When analysis is put into context, the question of optimizing other evaluation criteria such as recall or precision, becomes important.

 <br>
 <br>

Let’s look at the confusion matrix for our Logistic and Random Forest classification models

## Model Evaluation
## Evaluation of the Logistic Regression Model


```python
from sklearn.metrics import confusion_matrix, classification_report
```


```python
## Build Confusion Matrix
cf_lr = confusion_matrix(LR_expost_preds,y_test)
print(“Logistic Regression Accuracy:”, np.sum(np.diag(cf_lr))/np.sum(cf_lr))
```

    Logistic Regression Accuracy: 0.9009691363470239


|        |  Attack | Normal |
:-------------------------:|:-------------------------:|:-------------------------:
**Attack** | 64248 **(TPs)** | 789 **(FPs)**
**Normal** | 14927 **(FNs)** | 78734 **(TN)**

<br>
<br>

|        |  Precision | Recall | F1-Score |
:-------------------------:|:-------------------------:|:-------------------------:
**Attack** | 0.84 | 0.99 | 0.91
**Normal** | 0.99 | 0.81 | 0.89
||Accuracy | 0.90||

From the confusion matrix, the logistic regression does better at identifying most good connections, therefore optimizing the recall of the ‘GOOD’ class.

## Evaluation Of The Decision Tree Models


```python
## Build Confusion Matrix
cf_dt = confusion_matrix(DT_expost_preds,y_test)
print(“Logistic Regression Accuracy:”, np.sum(np.diag(cf_dt))/np.sum(cf_dt))
```

    Logistic Regression Accuracy: 0.9062433048935714


|        |  Attack | Normal |
:-------------------------:|:-------------------------:|:-------------------------:
**Attack** | 65106 **(TPs)** | 810 **(FPs)**
**Normal** | 14069 **(FNs)** | 78713 **(TN)**

<br>
<br>

|        |  Precision | Recall | F1-Score |
:-------------------------:|:-------------------------:|:-------------------------:
**Attack** | 0.99 | 0.85 | 0.91
**Normal** | 0.82 | 0.99 | 0.90

From the confusion matrix, the decision tree optimizes the ability to identify only bad connections with a precision a 99% precision and an 85% recall.

<br>
Choosing an appropriate evaluation criteria for a model is important as it ensures that the model learns to improve the metric of interest. For example, a threat investigator may be interested in identifying ALL bad connections with a certain tolerance for false alarms or may be interested in ensuring resources are only deployed to mitigate actual BAD connection with a very low tolerance for false positives. On the other hand, a website may be interested in optimizing media marketing metrics and therefore build a model to identify all ‘GOOD’ connections with tolerance for  some ‘BAD’ connections to get as many ‘GOOD’ connections as possible.

<br>
