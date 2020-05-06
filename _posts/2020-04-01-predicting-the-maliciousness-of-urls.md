---
layout: post
title:  "PREDICTING THE MALICIOUSNESS OF URLS"
date: "2020-04-01"
author: "Ruth Ikwu"
---

## Introduction
In this article, we walk through developing a simple feature set representation for identifying malicious URLs. We will create feature vectors for URLs and use these to develop a classification model for identifying malicious URLs. To evaluate how good our features are in separating malicious URLs from benign URLs, we build a Decision-Tree based machine learning model to predict the maliciousness of a given URL.

* TOC
 {:toc}

Malicious websites are well-known threats in cybersecurity. They act as an efficient tool for propagating viruses, worms, and other types of malicious codes online and are responsible for over 60% of most cyber attacks. Malicious URLs can be delivered via email links, text messages, browser pop-ups, page advertisements, etc. These URLs may be links to dodgy websites or most likely have embedded 'downloadables'. These embedded downloads can be spy-wares, key-loggers, viruses, worms, etc. As such it has become a priority for cyber defenders to detect and mitigate the spread of malicious codes within their networks promptly. Various techniques for malicious URL detectors have previously relied mainly on URL blacklisting or signature blacklisting. Most of these techniques offer 'after-the-fact' solutions. To improve the timeliness and abstraction of malicious URL detection methods, machine learning techniques are increasingly being accepted.

To develop a machine learning model, we need a feature extraction framework for featurizing URLs or converting URLs into feature vectors. In this article, We will collect samples of known malicious URLs and known benign URLs. We then develop a fingerprinting framework and extract a given set of <strong><em>M</em></strong> features for all URLs in the sample. We test the usefulness of these features in separating malicious URLs from benign URLs by developing a simple predictive model with these features. Finally, we measure the model's ability to predict the maliciousness of our URLs as the effectiveness of our features in separating malicious URLs from benign URLs.

The image below is an overview of the methodological process in this article.

<br>

<figure>
<img src="/assets/images/pmurls/methodology.png"/>
<figcaption style="text-align:center;">Analysis Process</figcaption>
</figure>

## The Data
We collected data from two sources: Alexa Top 1000 sites and phishtank.com. 1000 assumed benign URLs were crawled from Alexa top 1000 websites and 1000 suspected malicious URLs were crawled from [phishtank.com](phishtank.com). Due to virustotal API limit rates, we randomly sample 500 assumed benign URLs and 500 assumed malicious URLs. The URLs were then scanned through [virustotal](virustotal.com). URLs with 0 malicious detections were labeled as benign (b_urlX) and URLs with at least 8 detections were labeled as malicious (m_urlX). we dumped the JSON results of each scan in corresponding files 'b_urlX.json', 'm_urlX.json'. You can find these files [Here](https://github.com/eneyi/dataarchive/tree/master/pmurls/data/featurized).

```python
from requests import get
from os import listdir
import pandas as pd
import numpy as np
from pandas.io.json import json_normalize
import seaborn as sns
import matplotlib.pyplot as plt
import math
from datetime import datetime
plt.rcParams["figure.figsize"] = (20,20)
```
## Handling API Rate Limits and IP Blocking
To confirm that malicious URLs in our sample are malicious, we need to send multiple requests to VirusTotal. VirustTotal provides aggregated results from multiple virus scan engines. Also, we pass URLs through (Shodan)[shodan.io]. Shodan is a search engine for all devices connected to the internet providing service-based features of the URL's server. VirusTotal and Shodan currently have API rate limits of 4 requests per minute and at least 10,000 requests per month respectively per API key. While the number of URL requests for our data fell within the Shodan API request limits, VirusTotal proved a little more difficult. This is addressed by creating several VT API Keys (be kind, 4 at most) and randomly sampling them in each request. In addition to limits on the number of API requests, sending multiple requests within a short period will lead to IP blocking from VT and Shodan servers. We write a small crawler to get the latest set of elite IP addresses from https://free-proxy-list.net/ and create a new proxy-list on each request given the very short lifespan of free proxies. In addition to IP pooling, we use Python's FakeUserAgent library to switch User-Agents on each request.

Finally, For each request, we can send 16 requests per minute as opposed to the previous 4, with a new proxy and user agent. Each request has the following request parameters:

- 1 VirusTotal Key: Sample from VT API keys pool.
- 1 Shodan Key: Sample from Shodan API keys pool.
- 1 IP: Send a request to https://free-proxy-list.net/ to get the latest free elite proxy.
- 1 User-Agent: Sample useable user agents from Python's (Fake User-Agent)[https://pypi.org/project/fake-useragent/]

Our scanning from Shodan and VT produced the following [dataset](). From shodan, we extract the following features:
- numServices: Total number of services (open ports) running on the host
- robotstxt: Is the site has robots txt enabled


<script src="https://gist.github.com/eneyi/5c0b33129bcbfa366eb9fe79e96c1996.js?file=pooling.py"></script>

The final dataset after scanning is available [here](). You can download this data and run your analysis.

```python
data = pd.read_csv('scanned_data.csv')
data=pd.DataFrame(data)
```

## Fingerprinting URLS (Featurizing URLs for Malware URL Detection)

Our goal is to extract URL characteristics that are important in separating malicious URLs from good URLs. First, let’s look at the relevant parts in the structure of a URL.

<figure>
<img src="/assets/images/pmurls/typicalurl.png"/>
<figcaption style="text-align:center;"> A Typical URL</figcaption>
</figure>

A URL (short for Uniform Resource Locator) is a reference that specifies the location of a web resource on a computer network and a mechanism for retrieving it. The URL is made up of different components as shown in the figure below. The protocol or scheme specifies how (or what is needed for) information is to be transferred. The hostname is a human-readable unique reference of the computer’s IP address on the computer network. The Domain Name Service (DNS) naming hierarchy maps an IP address to a hostname. Compromised URLs are used to perpetrate cyber-attacks online. These attacks may be in any or more forms of phishing emails, spam emails, and drive-by downloads.

Regarding domains, owners buy domains that people find easier to remember. Owners would normally want names that are specific to a brand, product, or service which they are delivering. An attacker can register any domain with a combination of any legitimate sequence of characters. This part of the URL cannot be changed once set. Malicious domain owners may opt for multiple cheap domain names for example ‘xsertyh.com’. Also, malicious domains use additional and unnecessary text to disguise the URL as legitimate. The free URL parameters are parts of a URL that can be changed to create new URLs. These include directory names, file paths, and URL parameters. These free URL parameters are usually manipulated by attackers to form new URLs and propagate them.

There are many techniques for malicious URL detection, two main techniques being a) Blacklisting Techniques, and b) Machine Learning Techniques. Blacklisting involves maintaining a database of known malicious domains and comparing the hostname of a new URL to hostnames in that database. This has an ‘after-the-fact’ problem. It will be unable to detect new and unseen malicious URL, which will only be added to the blacklist after it has been observed as malicious from a victim. Machine learning approaches, on the other hand, provide a predictive approach that is generalizable across platforms and independent of prior knowledge of known signatures. Given a sample of malicious and benign malware samples, ML techniques will extract features of known good and bad URLs and generalize these features to identify new and unseen good or bad URLs.

Our URL fingerprinting process targets 4 types of URL features:
 - URL String Characteristics: Features derived from the URL string itself.
 - URL Domain Characteristics: Domain characteristics of the URLs domain. These include whois information and shodan information.
 - Page Content Characteristics: Features extracted from the URL's page (if any)

A summary of all features extracted are shown in the table below:

<table class="stripped" style="margin:auto;">
    <tr>
       <th>Feature Name</th>
       <th>Feature Group</th>
       <th>Feature Description</th>
   </tr>
    <tbody>
        <tr>
            <td>URL Entropy</td>
            <td>URL String Characteristics</td>
            <td>Entropy of URL</td>
        </tr>
        <tr>
            <td>numDigits</td>
            <td>URL String Characteristics</td>
            <td>Total number of digits in URL string</td>
        </tr>
        <tr>
            <td>URL Length</td>
            <td>URL String Characteristics</td>
            <td>Total number of characters in URL string</td>
        </tr>
        <tr>
            <td>numParameters</td>
            <td>URL String Characteristics</td>
            <td>Total number of query parameters in URL</td>
        </tr>
        <tr>
            <td>numFragments</td>
            <td>URL String Characteristics</td>   
            <td>Total Number of Fragments in URL</td>
        </tr>
        <tr>
            <td>domainExtension</td>
            <td>URL String Characteristics</td>
            <td>Domian extension</td>
        </tr>
        <tr>
            <td>hasHTTP</td>
            <td>URL domain features</td>
            <td>Website domain has http protocol</td>
        </tr>
        <tr>
            <td>hasHTTPS</td>
            <td>URL domain features</td>
            <td>Website domain has https protocol</td>
        </tr>
        <tr>
            <td>urlIsLive</td>
            <td>URL domain features</td>
            <td>The page is online</td>
        </tr>
        <tr>
            <td>daysSinceRegistration</td>
            <td>URL domain features</td>
            <td>Number of days from today since domain was registered</td>
        </tr>
        <tr>
            <td>daysSinceExpired</td>
            <td>URL domain features</td>
            <td>Number of days from today since domain expired</td>
        </tr>
        <tr>
            <td>bodyLength</td>
            <td>URL page features</td>
            <td>Total number of characters in URL's HTML page</td>
        </tr>
        <tr>
            <td>numTitles</td>
            <td>URL page features</td>
            <td>Total number of H1-H6 titles in URL's HTML page</td>
        </tr>
        <tr>
            <td>numImages</td>
            <td>URL page features</td>
            <td>Total number of images embedded in URL's HTML page</td>
        </tr>
        <tr>
            <td>numLinks</td>
            <td>URL page features</td>
            <td>Total number of links embedded in URL's HTML page</td>
        </tr>
        <tr>
            <td>scriptLength</td>
            <td>URL page features</td>
            <td>Total number of characters in embedded scripts in URL's HTML page</td>
        </tr>
        <tr>
            <td>specialCharacters</td>
            <td>URL page features</td>
            <td>Total number of special characters in URL's HTML page</td>
        </tr>
        <tr>
            <td>scriptToSpecialCharacterRatio</td>
            <td>URL page features</td>
            <td>The ratio of total length of embedded scripts to special characters in HTML page</td>
        </tr>
        <tr>
            <td>scriptToBodyRatio</td>
            <td>URL page features</td>
            <td>The ratio of total length of embedded scripts to total number of characters in HTML page</td>
        </tr>
    </tbody>
</table>
<br>
<script src="https://gist.github.com/eneyi/5c0b33129bcbfa366eb9fe79e96c1996.js?file=URLFeaturizer.py"></script>


Running the script above produces [the following](https://github.com/eneyi/dataarchive/blob/master/pmurls/data/scanned_data.csv) data with 23 features. We will separate integers, booleans, and object column names into separate lists for easier data access.


```python
objects = [i for i in data.columns if 'object' in str(data.dtypes[i])]
booleans = [i for i in data.columns if 'bool' in str(data.dtypes[i])]
ints = [i for i in data.columns if 'int' in str(data.dtypes[i]) or 'float' in str(data.dtypes[i])]
```

## Removing Highly Correlated Features

The most linear analysis assumes non-multicollinearity between predictor variables i.e pairs of predictor features must not be correlated. The intuition behind this assumption is that there is no additional information added to a model with multiple correlated features as the same information is captured by one of the features.

Multi-correlated features are also indicative of redundant features in the data and dropping them is a good first step for data dimension reduction. By removing correlated features (and only keeping, one of the groups of observed correlated features), we can address the issues of feature redundancy and collinearity between predictors.

Let’s create a simple correlation heat-map to observe the correlation between our derived features for malicious and benign URL and remove one or more of highly correlated features.


```python
corr = data[ints+booleans].corr()
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=np.bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(20, 15))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
```

<figure>
<img src="/assets/images/pmurls/correlation.png"/>
<figcaption style="text-align:center;"> Feature Cross-Correlation</figcaption>
</figure>


However, we do not want to remove all correlated variables-only those with a very strong correlation that do not add extra information to the model. For this, we define a certain ‘threshold’ (0.7) for positive and negative correlation observed.

The diagram above shows a cross-correlation grid between numeric features of the data. Here, we have selected all features that have at least a 70% positive or negative linear effect on the outcome feature. We see that most of the highly correlated features are negatively correlated. For example, there is a 0.56 negative correlation coefficient between the number of characters in a URL and the entropy of the URL which suggests that shorter URLs have

Here we will create a function to identify and drop one of multiple correlated features.

<script src="https://gist.github.com/eneyi/5c0b33129bcbfa366eb9fe79e96c1996.js?file=drop_multi_correlated.py"></script>

```
data2 = data[corr.columns].drop(dropMultiCorrelated(corr, 0.7), axis=1)
```
    Dropping urlLength....
    Dropping scriptLength....
    Dropping specialChars....
    Dropping bscr....
    Dropping hasHttps....


## Predicting Maliciousness of URLs (Decision Trees)

Modeling builds a blueprint for explaining data, from previously observed patterns in the data. Modeling is often predictive in that it tries to use this developed ‘blueprint’ in predicting the values of future or new observations based on what it has observed in the past.

Based on our extracted features, we want the best predictive model that tells us if an unseen URL is malicious or benign.  Therefore, we seek a unique combination of useful features to accurately separate malicious from benign URLs. We will go through two stages, feature selection, where we select only features useful in predicting the target variable and modeling with decision trees to develop a predictive model for malicious and benign URLs.

### Feature Selection

What variables are most useful in identifying a URL as ‘malicious’ or ‘benign’? Computationally, we can automatically select what variables are most useful by testing which ones ‘improves’ or ‘fails to improve’ the overall performance of our prediction model. This process is called ‘Feature Selection’. Feature selection also serves the purpose of reducing the dimension of data, addressing issues of computational complexity and model performance. The goal of feature selection is to obtain a useful subset of the original data that is predictive of the target feature in such a way that useful information is not lost (considering all predictors together).

There several [methods for feature selection](https://eneyi.github.io/2019/06/11/Introduction-to-intrusion-detection-with-python-feature-selection.html) including filter methods, wrapper-based methods, and recursive feature elimination. Here, we use the recursive feature elimination. RFE recursively removes attributes and builds a model on those attributes that remain. It uses the model accuracy to identify which attributes (and combination of attributes) contribute the most to predicting the target attribute. Features are ranked by the model’s coefficients or feature importances attributes, and by recursively eliminating a small number of features per loop, RFE attempts to eliminate dependencies and collinearity that may exist in the model.

Let's create a subset of our original data that contain only uncorrelated features.

```python
predictor_columns = data2.columns
d = data2[predictor_columns]
x, y = d[predictor_columns], data['vt_class']
```

We keep only features that are unique in their contribution to the model. We can now start developing our model with 70% of our original sample and these 14 features. We will keep 30% of our sample to evaluate the model's performance on new data.

  - numServices
  - entropy
  - numDigits
  - numParams
  - bodyLength
  - numTitles
  - numImages
  - numLinks
  - dsr
  - dse
  - sscr
  - sbr
  - robots
  - hasHttp

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 100)
```

### Decision Trees

```python
from sklearn import tree
from sklearn.metrics import accuracy_score

from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
```

There are multiple machine learning algorithms (classification) algorithms that can be applied to identifying malicious URLs. After converting URLs to a representative feature vector, we model our 'malicious URL identification problem' as a binary classification problem. A binary classification model trains a predictive model for a class with only two outcomes 'Malicious' and 'Benign'. Batch learning algorithms are machine learning algorithms that work under the following assumptions:

    - the entire training data is available before model development and
    - the target variable is known before the model training task.

Batch algorithms are ideal and effective in that they are explainable discriminative learning models that use simple loss minimization between training data points. Decision trees are one such batch learning algorithms in machine learning.

In decision analysis, a decision tree is a visual representation of a model's decision-making process to arrive at certain conclusions. The basic idea behind decision trees is an attempt to understand what factors influence class membership or why a data point belongs to a class label. A decision tree explicitly shows the conditions on which class members are made. Therefore they are a visual representation of the decision-making process.

Decision tree builds predictive models by breaking down the data set into smaller and smaller parts. The decision to split a subset is based on maximizing the information gain or minimizing information loss from splitting. Starting with the root node (the purest feature with no uncertainty), the tree is formed by creating various leaf nodes based on the purity of the subset.

In our case, the decision tree will explain class boundaries for each feature to classify a URL as malicious or benign. There are two main factors to consider when building a decision tree:

    - a) What criteria to use in splitting or creating leaf nodes and
    - b) tree pruning to control how long a tree is allowed to grow to control the risk of over-fitting.

The criterion parameter of the decision tree algorithm specifies what criteria (Gini or entropy) to control for while the max_depth parameter controls how far the tree is allowed to grow. Gini measurement is the probability of a random sample being classified incorrectly if we randomly pick a label according to the distribution in a branch. Entropy is a measurement of information (or rather lack thereof).

Unfortunately, since there is no prior knowledge of the right combination of criteria and tree depth, we would have to iteratively test for the optimal values of these two parameters. We test a max_depth for 50 iterations for both criteria and visualize the model accuracy scores.



<script src="https://gist.github.com/eneyi/5c0b33129bcbfa366eb9fe79e96c1996.js?file=parameterTuning.py"></script>

<figure>
<img src="/assets/images/pmurls/parameter_tuning.png"/>
<figcaption style="text-align:center;">Parameter Tuning</figcaption>
</figure>



It seems the best model is the simplest one with the Gini index and a max depth of 4. Also, maximizing the entropy does not seem to produce good results suggesting that new parameters added to the model do not necessarily give new information but may produce improved node probability purity. So we can fit and visualize the tree with max_depth = 4 and Gini criteria to identify which features are most important in separating malicious and benign URLs.

Build the model....

```python
###create decision tree classifier object
DT = tree.DecisionTreeClassifier(criterion="gini", max_depth=4)

##fit decision tree model with training data
DT.fit(X_train, y_train)

##test data prediction
DT_expost_preds = DT.predict(X_test)
```

Visualize the tree ...

```python
dot_data = StringIO()
export_graphviz(DT, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names=X_train.columns, class_names=DT.classes_)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
```


<figure>
<img src="/assets/images/pmurls/tree.png"/>
<figcaption style="text-align:center;">Parameter Tuning</figcaption>
</figure>


The accuracy of prediction models is very sensitive to parameter tuning of the max_depth (tree pruning) and split quality criteria (node splitting). This also helps in achieving the simplest parsimonious model that prevents over-fitting and performs just as well on unseen data. These parameters are specific to different data problems and it is good practice to test a combination of different parameter values.

Our model shows that malicious URLs have a lower script to special character ratio (sscr) and URL characters that are relatively more 'ordered' or more monotonous. Additionally, malicious URLs may have domains that have expired somewhere between 5 - 9 months ago. As we know, non-malicious URL domain owners will have domain names that are representative of their brands. Additionally, we also know of issues of 'malvertising' where scammers take ownership of expired legitimate domains to distribute downloadable malicious codes. Finally, the most distinguishing features of benign URLs is longevity.  They seem to moderate script to special character ratio in HTML body content with longer domain lifetime of 4 - 8 years.
