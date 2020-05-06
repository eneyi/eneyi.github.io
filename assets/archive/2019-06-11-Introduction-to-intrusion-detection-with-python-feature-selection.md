---
layout: post
title:  "Introduction To Intrusion Detection With Python - Feature Selection"
date: "2019-06-11"
author: "Ruth Ikwu"
---

## Introduction
What variables are most useful in identifying a connection as 'good' or 'bad'? Practically, answering this question requires years of experience and a deep understanding of the problem domain. However, computationally, we can automatically select what variables are most useful by testing which ones 'improves' or 'fails to improve' the overall performance of our prediction model. This process is called 'Feature Selection'. Feature selection is the automated process of selecting attributes of a dataset that are most relevant in predicting a target attribute. Feature selection serves one purpose of reducing the dimension of data which addresses issues of computational complexity and model performance. The goal of feature selection is to obtain a useful subset of the original data that is predictive of the target feature in such a way that useful information is not lost (considering all predictors together).

* TOC
 {:toc}

There a number of ways to identify data that are 'removal candidates'. These include:
- Redundant features: Features that are simply variations or replicas of some other feature (just in some other way). e.g 'height (kg)'and 'height (inch)' or 'GBs (Used)' and '(GBs Available)'. These may often times include features that are simply not relevant in the problem space (determined by the analytical context). E.g 'Firstname', 'Lastname','ID'.
- Correlated Features: Correlation between features is seen as another form of redundancy, because the information provided by both features to the model is fairly the same - therefore we need one messanger to carry that information to our model. Additionally, if your model is based on the OLS equation (as most regression models are), correlated features breaks the assumption of 'no multi-collinearity' between features.
- Not so useful features: Features that do not contribute to improving the performance of the model. This means that prior or current knowledge of these features are not needed to infer future values of the target class.

In the end, keep it simple and parsimonious.


![Feature ELimination Process](/assets/images/kdd3/FeatureElimination.png)


## Importance of Feature Selection

Feature selection should be the top priority for any analytical process as performance and generalization can be greatly improved by considering only a subset of correct features based on feature contribution to variation in the data. Feature selection facilitates the development of simpler, faster, beter performing Learning models.

Note, similar to feature reduction, there is dimension reduction. Dimension reduction differs from 'feature selection' as it creates new combinations of attributes, whereas, 'feature selection' simply includes or excludes features. There are however established methods for 'dimension reduction' not covered here.

There are generally three classes of feature selection algorithms: filter methods, wrapper methods and embedded methods. We will cover some of these here to compare the difference in results returned. Let's begin by loading in our data. Our predictor data is the clustered dataset, outcome of the previous tutorial and our target variable remains the same.


Let's read in the data we are working with. For this, we import our target feature and data outcome from the previous tutorial.


```python
import pandas as pd
from Modules.OHE import ohe
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
datadir = "data"
```


```python
##lets read in our data
with open("data/target.txt","r+") as ff: trainy = pd.Series([i.strip() for i in ff.readlines()])

#clustered=read_csv("clustered.csv")
trainx=pd.read_csv("data/clustered.csv")
trainx=trainx[trainx.columns[2::]]

##create dummies for categorical data
train_ohe = ohe(trainx)

#extract cluster
clusters = train_ohe['cluster']

##remove cluster feature
train_ohe.drop("cluster", axis=1, inplace=True)
```


```python
train_ohe.shape
```
    (494021, 117)

```python
trainy2 = ["normal" if i == "normal." else "attack" for i in trainy]
outcome = [0 if i =="normal" else 1 for i in trainy2]
train_ohe['outcome'] = [0 if i =="normal" else 1 for i in trainy2]
```

Here we create a function to plot correlations between pairs of features in the data.


```python
def plot_cormat(cormat):
    ##generate a mask for upper triangle
    mask = np.zeros_like(cormat, dtype=np.bool)
    mask[np.triu_indices_from(mask)] =True

    ##setup a diverging color map
    cmap = sns.diverging_palette(210,10,as_cmap=True)

    ##create figure and axis
    fig,ax=plt.subplots(figsize=(10,10))

    ##draw heatmap
    sns.heatmap(cormat,mask=mask,cmap=cmap,square=True, linewidths=.5,center=0, ax=ax)

```

## Removing Highly Correlated Features

Most linear regression analysis assumes non-multicollinearity between predictor variables. This means that pairs of predictor features must not be correlated. So, for example, given the correlation map below, we would simply can select the 'count' feature to represent all time traffic and machine traffic attributes-as they are all inter-correlated. Let's create a simple correlation heatmap of all features to see which pairs are highly correlated.



```python
#make correlation matrix
cormat = train_ohe.corr()

plot_cormat(cormat)
```


![png](/assets/images/kdd3/HighlyCorrelated.png)


Aditionally, observing correlations between variables is indicative of redundant features in the data. By removing correlated features (and only keeping, one of groups of observed correlated features), we can address the issues of feature redundancy and colinearity between predictors.

However, we do not want to remove all correlated variables-only those with very strong correlation that do not add extra information to the model. This is usually a function of the magnitude of correlation and the amount of data available. Usually, this is implemented as a pre-step to feature engineering to reduce computational complexity and retain only potentially useful information. For this we define a certain 'threshold' for positive and negative correlation observed.

Let's redude the dimension of the data by simply dropping highly-correlated features. First we need to decide at what point to define 'Highly-Correlated'. The stardard definition for correlation is a 0.5 correlation co-efficient. So, here we will define 'Highly' as this standard plus 10%.


```python
##Define threshold to remove pairs of features with correlation coefficient greater than 0.7 or -0.7
threshold = 0.7

# Select upper triangle of correlation matrix
upper = cormat.abs().where(np.triu(np.ones(cormat.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than threshold
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
```
```python
pd.DataFrame(to_drop)
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
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>protocol_type_tcp</td>
    </tr>
    <tr>
      <th>1</th>
      <td>service_ecr_i</td>
    </tr>
    <tr>
      <th>2</th>
      <td>flag_S0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>flag_SF</td>
    </tr>
    <tr>
      <th>4</th>
      <td>hot</td>
    </tr>
    <tr>
      <th>5</th>
      <td>logged_in</td>
    </tr>
    <tr>
      <th>6</th>
      <td>su_attempted</td>
    </tr>
    <tr>
      <th>7</th>
      <td>num_root</td>
    </tr>
    <tr>
      <th>8</th>
      <td>is_guest_login</td>
    </tr>
    <tr>
      <th>9</th>
      <td>count</td>
    </tr>
    <tr>
      <th>10</th>
      <td>srv_count</td>
    </tr>
    <tr>
      <th>11</th>
      <td>serror_rate</td>
    </tr>
    <tr>
      <th>12</th>
      <td>srv_serror_rate</td>
    </tr>
    <tr>
      <th>13</th>
      <td>rerror_rate</td>
    </tr>
    <tr>
      <th>14</th>
      <td>srv_rerror_rate</td>
    </tr>
    <tr>
      <th>15</th>
      <td>same_srv_rate</td>
    </tr>
    <tr>
      <th>16</th>
      <td>dst_host_srv_count</td>
    </tr>
    <tr>
      <th>17</th>
      <td>dst_host_same_srv_rate</td>
    </tr>
    <tr>
      <th>18</th>
      <td>dst_host_diff_srv_rate</td>
    </tr>
    <tr>
      <th>19</th>
      <td>dst_host_same_src_port_rate</td>
    </tr>
    <tr>
      <th>20</th>
      <td>dst_host_serror_rate</td>
    </tr>
    <tr>
      <th>21</th>
      <td>dst_host_srv_serror_rate</td>
    </tr>
    <tr>
      <th>22</th>
      <td>dst_host_rerror_rate</td>
    </tr>
    <tr>
      <th>23</th>
      <td>dst_host_srv_rerror_rate</td>
    </tr>
    <tr>
      <th>24</th>
      <td>outcome</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_ohe.drop(to_drop, axis=1, inplace=True)
```


```python
train_ohe.shape
```
    (494021, 93)

By simply removing 'one of pairs' of correlated features, we have reduced the number of features by 20.5% from 117 features to 93 features, dropping 24 features. Now we are down from 118 to 93 features. A 21.2% decrese in the number of dimensions. We will take just these 93 features into our feature selection methods for further reduction.

## Methodologies for Feature Selection

There are generally three classes of feature selection algorithms: filter methods, wrapper methods and embedded methods. We will cover some of these here to compare the difference in results returned.

We will divide our dataset into two, attack traffic and normal traffic as we did in the previous article. Then we can observe the distribution of selected continuous variables across each traffic type.


```python
##create mask to divide data
mask = trainy == "normal."

## create attack and normal traffic sets
attack_traffic = train_ohe[~mask]
normal_traffic = train_ohe[mask]
```


```python
##subset all continous features
floats = [i for i in train_ohe.columns if np.dtype(train_ohe[i]) == 'float64']
```

We will now create a distribution of normal and attack traffic for each continous feature to see which best seperates the data.


```python
##create distrubitions of attack and normal traffic for each continous feature.
fig = plt.figure(figsize=(20,20))
for i in range(len(floats)):
    ax=fig.add_subplot(5,3,1+i)
    ax.hist(normal_traffic[floats[i]],density=True,bins=10, log=True,color='teal', label="Normal Traffic",alpha=0.5)
    ax.hist(attack_traffic[floats[i]],density=True,bins=10, log=True, color='red', label="Attack Traffic",alpha=0.5)
    ax.set_title(floats[i].upper())

    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.legend()
plt.tight_layout()
```


![png](/assets/images/kdd3/HistogramMatrix.png)


Given these continous features, we can see that features like the destination bytes, wrong fragment, root shell, number of files accessed, the sum of connections to the same IP address, the sum of bad checksum packets in a connection and the sum of operations in control files maximize seperation between the good and bad connections. Let's apply some actual feature selection methods.

### Filter Methods

Filter methods are based on the coefficients of the correlation matrix (usually 'Pearson's Coreelation'), Chi Squared Test or information gain. The contribution of each feature is therefore considered 'independently' of other features in the dataset. In the filter method, we simply drop any feature that does not meet a threshold of 'evaluation' with the outcome feature. For example, using a Pearson's Correlation Coefficient as an evaluation criteria with a ±0.6 threshold, we simply drop all features that do not meet a ±0.6 correlation cofficient with the target feature.

Filter methods are generally used as a pre-processing step as the selection of features is independent of any machine learning algorithm. There are different feature methods for different data types. We use Pearson's Correlation Coefficient or ANOVA (Analysis of Variance) when the outcome feature is a continous feature and Linear Discriminant Analysis (LDA) or Chi-Square test when the outcome feature is categorical-like inthis case.

- Pearson's Coreelation Co-efficient:
- Analysis Of Variance (ANOVA)
- Linear Discriminant Analysis (LDA)
- Chi-Square:



#### Pearson's Correlation Coefficient

Here we define a simple function to eliminate all features in the dataset that are not at least correlated with the outcome feature, up to a certain threshold. The correlation coefficient represents the magnitude and direction of the linear relationship between two continuous variables. A correlation coefficient has a value between -1 and 1. A positive value implies a positive relationship (increase in one unit of a feature given a change in one unit of the other). A negative value implies a decreasing trend (decrease in one unit of a feature given a change in one unit of the other). A value closer to zero implies a weaker correlation between pairs while a value of zero implies no correlation (i.e changes in one feature has no effect on changes in the other).

Filtering features based on the correlation coefficient is a straight forward process. We set a threshold for the magnitude of dependecy between features and the outcome variable we are interested in keeping and simply eliminate all those variables whose correlation coefficient with the outcome feature do not meet that threshold.


```python
#define function to select only features that meet a threshold correlation coefficient with the outcome
def corr_filter(predictor, target, threshold):
    predictor['target'] = target
    cormat = predictor.corr()
    cormat['target'] = [abs(i) for i in cormat['target']]
    coefs = cormat['target'] >= threshold
    dd = cormat.columns[coefs]
    return dd
```


```python
##filter only pairs of features that meet a 0.5 correlation coefficient threshold
tt = train_ohe.copy()
if "outcome" in tt.columns:
    tt.drop("outcome", inplace=True, axis=1)
pearsons = corr_filter(tt,outcome, 0.5)
pearsons = tt[pearsons]
pearsons_features = list(pearsons.columns)
plot_cormat(pearsons.corr())
```


![png](/assets/images/kdd3/pearsons.png)


Here, we have selected all features that have at least a 50% positive or negative linear effect on the outcome feature. We see the 'protocol_type_icmp', 'service_ecr_i', 'service_http' and 'logged_in' features are both correlated with our target feature. Also, there are strong correlations observed between our time traffic and machine traffic attributes with the outcome feature. In a simple filtering method, we reduce or predictors to a subset of these features.

Unfortunately, correlation based on simple OLS equations are only meaningful for continous variables or ordinal data. Given that our outcome feature and some of its predictors are categorical variable, the concept of an 'increasing' or 'decreasing' trend has not analytical meaning. So this is not a suitable feature selection method for this task.

#### Linear Discriminant Analysis (LDA)

![Linear Discrminant Analysis](https://www.digitalvidya.com/wp-content/uploads/2019/02/Image-1-2.png)

Linear Discriminant Analysis is a method used in learning problems to find a linear combination of features that best explain the data or separates two or more classes of objects or events. The resulting combination may be used as a linear classifier, or, more commonly, for dimensionality reduction before later classification.

In a classification problem, Linear discriminant analysis focuses on maximizing the seperability among the classes in the outcome feature by maximizing the distances between the feature means for the two classes. Additionally, we will also like to minimize the variation of features within each class.

![lLinear Discriminant Process](ldaProcess.png)

Before we begin our computation, let's remove the outcome or target feature from our data.


```python
if "target" in train_ohe.columns:
    train_ohe.drop("target", axis=1, inplace=True)
if "outcome" in train_ohe.columns:
    train_ohe.drop("outcome", axis=1, inplace=True)
```

Next, we will create an nXm matrix from our data where n is the number of obersvations and m is the number of features in our data.


```python
lda_x = train_ohe.as_matrix()
lda_y = outcome
```


```python
lda_x.shape ##93 features
```
    (494021, 93)



#### Step 1: Estimate Mean Vectors for Attack and Normal Traffic

Here, we will compute the mean vectors for the feature space for each traffic class. We compute the feature mean space for each of our traffic datasets (attack traffic and normal traffic). Each mean vector should be a 93X1 vector where each item is the mean of the corresponding feature.


```python
mean_vectors = []

###menas for attack traffic
mean_vectors.append(np.mean(lda_x[~mask], axis=0))

##means for normal traffic
mean_vectors.append(np.mean(lda_x[mask], axis=0))
```

#### Step 2: Compute Scatter Matrices

The aim of LDA is to find the maximum of ratio of between-class scatter matirx SB to the within-class scatter matrix SW of the projected features. So we estimate two matrices 'SW' within class scatter matrix and 'SB' between class scatter matrix.

##### 'Within class scatter matrix'

The within class scatter matrix is estimated with the equation below and is simply the sum of the covariance matrix estimated for each class.

![Within Class Scatter Matrix](/assets/images/kdd3/sw.png)


```python
cols = lda_x.shape[1] #number of features
```


```python
##Attack within class scatter matrix
attack_sc_mat = np.zeros((cols,cols))
for row in lda_x[~mask]:
    row, mv = row.reshape(cols,1), mean_vectors[0].reshape(cols,1)
    attack_sc_mat += (row-mv).dot((row-mv).T)


##Normal within class scatter matrix
normal_sc_mat = np.zeros((cols,cols))
for row in lda_x[mask]:
    row, mv = row.reshape(cols,1), mean_vectors[1].reshape(cols,1)
    normal_sc_mat += (row-mv).dot((row-mv).T)


###sum class matrix
sw = np.zeros((cols,cols))
sw = attack_sc_mat + normal_sc_mat
```

##### 'Between class scatter matrix'

![Within Class Scatter Matrix](/assets/images/kdd3/sb.png)

```python
lda_x_omega = np.mean(lda_x, axis=0)
sb = np.zeros((cols, cols))

## attack sb matrix
attack_n = lda_x[~mask].shape[0]
attack_mean_vec = mean_vectors[0].reshape(cols,1)
attack_o_mean = lda_x_omega.reshape(cols,1)
sb += attack_n*(attack_mean_vec - attack_o_mean).dot((attack_mean_vec - attack_o_mean).T)


## attack sb matrix
normal_n = lda_x[mask].shape[0]
normal_mean_vec = mean_vectors[1].reshape(cols,1)
normal_o_mean = lda_x_omega.reshape(cols,1)
sb += normal_n*(normal_mean_vec - normal_o_mean).dot((normal_mean_vec - normal_o_mean).T)
```




```python
eigenvals, eigenvecs = np.linalg.eig(np.linalg.pinv(sw).dot(sb))
egpairs = [(np.abs(eigenvals[i]), eigenvecs[:,i]) for i in range(len(eigenvals))]
```


```python

```


```python
eigens = pd.DataFrame([train_ohe.columns, [i[0] for i in egpairs]]).T
eigens.columns = ["Feature", "Eigen"]
eigens.sort_values("Eigen", ascending=False,inplace=True)
eigens["FeatureM"] = train_ohe.columns
eigens.head(2)
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
      <th>Feature</th>
      <th>Eigen</th>
      <th>FeatureM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>protocol_type_icmp</td>
      <td>11.2063</td>
      <td>protocol_type_icmp</td>
    </tr>
    <tr>
      <th>2</th>
      <td>service_IRC</td>
      <td>1.42311e-15</td>
      <td>protocol_type_udp</td>
    </tr>
  </tbody>
</table>
</div>




```python
egpairs[0][0]/sum(eigenvals).real
```




    0.9999999999999997



All eigen values except 1 are close to or exactly 0. The number of linear discriminants is at most 'c'-1 where 'c' is the number of classes. In our case we can have at most one linear discriminant. Here, we will sort the eigen values in descending order. The first eigen value is the most informative explaining about 99.99% of the variance in the data.


```python
w = np.hstack((egpairs[0][1].reshape(cols,1), egpairs[1][1].reshape(cols,1)))
lda = lda_x.dot(w.real)
```


```python
labels = {0: 'normal', 1: 'attack'}
```


```python
lda.shape
```




    (494021, 2)




```python
checks = [mask,~mask]
def plot_lda():

    ax = plt.subplot(111)
    for label,marker,color in zip(
        range(len(checks)),('^', 's'),('teal', 'brown')):

        plt.scatter(x=lda[:,0].real[checks[label]],
                y=lda[:,1].real[checks[label]]*-1,
                marker=marker,
                color=color,
                alpha=0.5,
                label=labels[label]
                )

    plt.xlabel('LD1')
    plt.ylabel('LD2')

    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title('LDA: Projection onto the first 2 linear discriminants')

    # hide axis ticks
    plt.tick_params(axis="both", which="both", bottom="off", top="off",  
            labelbottom="on", left="off", right="off", labelleft="on")

    # remove axis spines
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)    

    plt.grid()
    plt.tight_layout
    plt.show()

plot_lda()
```


![png](/assets/images/kdd3/lda.png)


Although LDA does not completely seperate the two classes, a significant amount of seperation is observed.


```python
lda_features = eigens.Feature[0]
```

### Wrapper Methods

Wrapper methods are based on measuring the performance of a model built with different combinations of predictors to select the optimum model. Basically, if the model's performance is not improved or made worse-of by the addition of a feature, the feature is removed from the model.

There numerous ways of achieving this:
- Forward selection: Start with one feature in the model and iteratively test performance with every possible feature, keeping only features that increase model performance.

- Backward Elimination: Start with all possible features in the model and iteratively test performance removing any feature in which the model is better-off without.

- Bidirectional elimination: A strategic combination of the two steps above, performed at each iteration

Firstly, we would implement a backward elimination feature selection metjod on our dataset and the we use a library 'mlxtend' to perform a forward feature selection on the data. Given that we have a 119 features, this is a computationally expensive process and may take a while.


#### Backward Elimination

The process does exactly as the name suggests. We feed all possible features to the model and iteratively remove the worst performing features, selecting only the best model. The 'best' depends on some model evaluation criteria (AUC, accuracy, precision, recall, RMSE etc). We will use a simple OLS linear regression model to test and select our features. We build a single OLS equation that contains all 119 features, on each iteration, we take out the feature that least contributes to the model. We will build a simple OLS equation to demonstrate this process. We select a naive performance metric (the pvalue) with a 95% confidence. If the pvalue is greater than 0.05, we will eliminate the fature. First let's prepare the data. We will use our OHE function from the previous tutorial.


```python
from sklearn.model_selection import train_test_split
```


```python
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
```


```python
cols = list(train_ohe.columns)
max_p = 1
while len(cols) > 0:
    x = train_ohe[cols]
    print("Fitting Model with ", x.shape[1], " Features")
    lr1 = sm.add_constant(train_ohe)
    model = sm.OLS(outcome, x).fit()
    p = pd.Series(model.pvalues.values, index=cols)
    max_p = max(p)
    max_feature = p.idxmax()
    if (max_p>0.05):
        print("Droping Feature ", max_feature)
        cols.remove(max_feature)
    else:
        break
```

    Fitting Model with  93  Features
    Droping Feature  flag_RSTOS0
    Fitting Model with  92  Features
    Droping Feature  flag_OTH
    Fitting Model with  91  Features
    Droping Feature  is_host_login
    Fitting Model with  90  Features
    Droping Feature  flag_S3
    Fitting Model with  89  Features


```python
train_ohe = train_ohe[cols]
```

We are down to 89 features. Our linear regression-based backward elimination, leaves us with 89 features for prediction and drops 4 features from the previous step. While this is good for demonstration purposes, note that our target variable is categorical, thus a Linear Regression based evaluation may not be an appropriate metric to measure performance and a feature's predictability of the target. Now let's perform a forward selection feature selection process with Sebastain Raschka's mlxtent library to test sequential feature selection to test the performance of features selected.

#### Forward Selection

Next, we will define a random forest classifier, as well as a step forward feature selector, and then perform our feature selection. mlxtend is computationlly expensive and we have a relatively large dataset. So this should take a while to execute.


```python
from sklearn.svm import LinearSVC
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
```

```python
svm = LinearSVC(verbose=False)
sfs1 = sfs(svm,forward=True,floating=False,verbose=1,scoring='accuracy')
sfs1.fit(train_ohe,trainy2)
```

    SequentialFeatureSelector(clone_estimator=True, cv=5,
                              estimator=LinearSVC(C=1.0, class_weight=None,
                                                  dual=True, fit_intercept=True,
                                                  intercept_scaling=1,
                                                  loss='squared_hinge',
                                                  max_iter=1000, multi_class='ovr',
                                                  penalty='l2', random_state=None,
                                                  tol=0.0001, verbose=False),
                              floating=False, forward=True, k_features=1, n_jobs=1,
                              pre_dispatch='2*n_jobs', scoring='accuracy',
                              verbose=1)




```python
fs_features = train_ohe.columns[sfs1.k_feature_idx_]
```

So our forward feature selection tells us that only 'the number of bytes recieved per connection' gives the best model predicting bad connections, given 'accuracy' as the evaluation criteria.

## Recursive Feature Elimination

The Recursive Feature Elimination (RFE) method is a feature selection approach. It works by recursively removing attributes and building a model on those attributes that remain. It uses the model accuracy to identify which attributes (and combination of attributes) contribute the most to predicting the target attribute. Features are ranked by the model’s coefficients or feature importances attributes, and by recursively eliminating a small number of features per loop, RFE attempts to eliminate dependencies and collinearity that may exist in the model.

RFE requires a specified number of features to keep, however it is often not known in advance how many features are valid. To find the optimal number of features cross-validation is used with RFE to score different feature subsets and select the best scoring collection of features. The RFECV visualizer plots the number of features in the model along with their cross-validated test score and variability and visualizes the selected number of features.


```python
# Recursive Feature Elimination
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


# create a base classifier used to evaluate a subset of attributes
model = LogisticRegression(solver='lbfgs',max_iter=2000)


# create the RFE model and select 3 attributes
rfe = RFE(model, 3)
rfe = rfe.fit(train_ohe, trainy2)

# summarize the selection of the attributes
print(rfe.support_)
print(rfe.ranking_)
```

    [False False False False False False False False False False False False
     False False False False False False False False False False False  True
     False False False False False False False False False False False False
     False False False False False False False False False False False False
     False False False  True False False False False False False False False
     False False  True False False False False False False False False False
     False False False False False False False False False False False False
     False False False False False]
    [18  2  3 11 58  6 36 44 28 53 62 59 77 73 60 13 42 67  5  8  4 39 49  1
     54 29 35 37 56 52 47 65 34 51 55 48 41 57 45 63 80 79 82 50 10 61 17 26
     32 30 72  1 43 70 64 40 31  9 84 21 12  7  1 66 46 33 38 27 16 15 24 22
     69 85 86 20 14 76 75 81 23 78 71 25 87 68 74 83 19]



```python
rfe_features = train_ohe.columns[rfe.support_]
```


```python
d2 = pd.DataFrame([train_ohe.columns, rfe.ranking_], index=["x","y"]).transpose().sort_values("y")
sns.set(rc={'figure.figsize':(15.7,8.27)})
sns.barplot(d2["x"], d2["y"])
plt.xlabel("Features")
plt.ylabel("Feature Ranking")
plt.title("Feature Ranking")
plt.xticks(rotation=90)
```




    (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
            34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
            51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
            68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
            85, 86, 87, 88]), <a list of 89 Text xticklabel objects>)




![png](/assets/images/kdd3/FeatureRankings.png)


We can also plot the feature rankings to view the top features or best features for predicting the outcome class. The feature rankings are such that rankings[i] corresponds to the ranking position of the i-th feature. Selected (i.e., estimated best) features are assigned rank 1. Therefore, there are only three useful features returned by the FRE - 'service_http', 'service_smtp' and 'service_urp_i'.


```python
##combine features selected from both feature selection methods
selected = list(set(list(rfe_features)+[fs_features]+[lda_features]+list(pearsons_features)))[:-1]
selected = [i for i in selected if "target" not in i]
selected = train_ohe[selected]
## add clusters column to selected features
selected['cluster'] = clusters

##write out data
selected.to_csv("data/features.csv")
```

## Conclusion

- Filter methods do not incorporate a machine learning model in order to determine if a feature is good or bad.
- Wrapper methods use a machine learning model and train it the feature to decide if it is essential or not.
- Wrapper methods are computationally costly as they involve training a new model on each iteration.
- Features determined from wrapper methods tend to overfit the data as the model is already 'trained' with the data. Essentailly, wrapper methods are simply creating multiple experiments with the training data to find the best 'given the current data'.

In the next and final article, we will use this subset of selected features, with our clusters returned from the previous article to build a model that can tag connections as 'good' or 'bad'.
