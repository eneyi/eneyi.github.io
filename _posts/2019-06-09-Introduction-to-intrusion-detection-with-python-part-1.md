---
layout: post
title:  "Introduction To Intrusion Detection With Python - Part 1"
date: "2019-06-09"
author: "Ruth Ikwu"
---
## Introduction

This is a four part series on implementing intrusion detection techniques to network traffic data, using python. In this series, we will use benchmarked KDDCup dataset to demonstrate how simple machine learning techniques such as unsupervised and supervised learning can be applied to network defence. Although, simplistic in its nature, the purpose is to build a machine learning model that identifies 'bad' and 'good' connections. We will follow a very similar pattern to all other machine learning techniques, but discuss model evaluation as useful in network defence.

The series is split as thus:

 - Part 1: Introduction to Intrusion Detection and the Data
 - Part 2: Unsupervised learning for clustering network connections
 - Part 3: Feature Selection
 - Part 4: Connection Classification

## Table of Content
* TOC
 {:toc}


## What is an intrusion?

Assuming you are familiar with what a [computer network](https://en.wikipedia.org/wiki/Computer_network) is, a network intrusion is a malicious or unexpected activity in any part of a computer network. Network intrusion detection is a methodology for monitoring and detecting these malicious activities on the network. Intrusion detection and prevention are two broad terms describing application of security practices used in mitigating attacks and blocking new threats.

Therefore, applying specialised intelligent analysis to security events through statistics, machine learning and AI is generally termed ‘Anomaly Detection’ (Detection of malicious activities by monitoring things that do not fit into the network’s ‘normal’ behaviour). For example, if a network has an established mean of expected incoming connections over a period of time-and this amount suddenly spikes to 250% the normal-or if the mean number of packets sent from the internal network spikes-or if number of unacknowledged 'SYN' requests suddenly spikes.

<img src="/assets/images/kdd1/anomaly.jpg" alt="drawing" width="300" height="200"/>

Anomaly based intrusion detection approaches are mainly statistical, supervised or unsupervised. Statistical anomaly analysis identify correlations and significant deviations from the normal network behaviour. Supervised anomaly detection uses past network behaviour to guess what future behaviour. There is no blanket definition for a threshold of what a ‘malicious activity’ may be, since the idea of an anomaly has to be put in context of a cyber-attack and the design of the network.

In this tutorial, we will address the classic KDD 1999 intrusion detection challenge by building a model that distinguishes 'bad' connections, called intrusions or attacks, and 'good' normal connections. We use a combination of unsupervised and supervised learning techniques to identify attack connections.

By applying unsupervised learning before classification, we are able to find hidden patterns in attack packets that improves the identification of 'bad' and 'good' connections.

## Setup

This article has the following python dependencies installed.

- Python 3.6
- Pandas: For data manipulation
- Numpy: For array and matrix Operations
- Requests: For accessing the web
- Matplotlib: For visualizing Data
- Seaborn: For data visualization

## The Data

The data used in this tutorial is the KDD Cup 1999 dataset. This benchmark dataset has been set used for the Third International Knowledge Discovery and Data Mining Tools Competition, held in conjunction with KDD-99, the Fifth International Conference on Knowledge Discovery and Data Mining. The competition task was to build a network anomaly detector, a predictive model capable of distinguishing between 'bad' connections, called intrusions or attacks, and 'good' normal connections.

This data contains a standard set of data, which includes a wide variety of intrusions simulated in a military network environment. The experimental environment set up an environment to acquire nine weeks of raw TCP dump data for a local-area network (LAN) simulating a typical U.S. Air Force LAN.  

The raw [training data](http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz) was processed into about five million connection records.  There is also [two weeks](http://kdd.ics.uci.edu/databases/kddcup99/kddcup.testdata.unlabeled.gz) of test data yielded around two million connection records. In this article, we use a subset (about 10%) of the [training data](http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz) and the [test data](http://kdd.ics.uci.edu/databases/kddcup99/kddcup.newtestdata_10_percent_unlabeled.gz) to build our clustering and classification models.

A connection is a sequence of TCP packets starting and ending at some well defined times, between which data flows to and from a source IP address to a target/destination IP address under communication protocol.  Each row in the data set represents a single connection and each connection is labelled as either normal, or as an attack, with exactly one specific attack type.  Each connection record consists of about 100 bytes. More of the task description and the data features can be read [here](http://kdd.ics.uci.edu/databases/kddcup99/task.html).


To read in the datasets, let's define the location of our datasets on the web. These links were taken from the KDD cup 1999 Website [here](http://kdd.ics.uci.edu/databases/kddcup99/task.html)


To read in the datasets, let's define the location of our datasets on the web. These links were taken from the KDD cup 1999 Website [here](http://kdd.ics.uci.edu/databases/kddcup99/task.html)


```python
train_data_page = "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz"
test_data_page = "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.testdata.unlabeled_10_percent.gz"
labels ="http://kdd.ics.uci.edu/databases/kddcup99/kddcup.names"
datadir = "data"
```


Let's start with some basic imports. Make sure dependencies are installed.


```python
from pandas import DataFrame, read_csv, Series
import numpy as np
import os
from requests import get
```

This handy class downloads, unzips, cleans, formats and labels our data. This calss is also defined in 'getKdd.py' and can be imported with the statement 'from getKdd import KDD_Data'.


```python
os.chdir("/Users/crypteye/Documents/Github/cyda/kdd_series")
```


```python
class KDD_Data():
    '''Initializes the class with needed inputs, a link for data location (webpage) and a link for labels location (labels)'''
    def __init__(self, webpage, labels, datadir = datadir):
        self.webpage = webpage
        self.labels = labels
        self.outfile = self.webpage.split("/")[-1].replace(".gz","")
        self.datadir = datadir

        if not os.path.exists(self.datadir):
            os.mkdir(self.datadir)
        os.chdir(datadir)

    '''Downloads data from webpage, unzips it and stores in outfile'''
    def download_data(self):

        #download data from web
        print("Downloading Data Files")
        try:
          os.system("wget "+self.webpage)
        except:
          print("Failed to download data files stoping program...")

        #unzip data
        os.system("gunzip "+self.outfile)

    '''Reads the content of outfile and returns a list for each line in file'''
    def read_data(self):
        with open(self.outfile, "r+") as ff:
            lines = [i.strip().split(",") for i in ff.readlines()]
        ff.close()
        return lines

    '''Reads the webcontent of labels url'''
    def readlabels(self):
        response = get(self.labels)
        labels = response.text
        labels = [i.split(",")[0].split(":") for i in labels.split("\n")]
        labels = [i for i in labels if i[0]!='']
        return labels[1::]

    '''Runs class functions and returns a tuple, target vector and predictor matrix'''
    def run(self):
        self.download_data()
        data = DataFrame(self.read_data())
        labels = self.readlabels()
        data.columns = [i[0] for i in labels]+['target']

        for i in range(len(labels)):
            if labels[i][1] == ' continuous.':
                data.iloc[:,i] = data.iloc[:,i].astype(float)

        target, predictors = data.iloc[:,-1], data.iloc[:,:-1]
        predictors.columns=data.columns[:-1]

        with open("target.txt", "w+") as ff:
            for i in target: ff.write(str(i)+"\n")
        predictors.to_csv("predictors.csv")
        os.system("rm -rd "+self.outfile)
        os.system("rm -rd "+self.outfile.replace(".gz",""))
        os.chdir("../")

```


```python
KDD_Data(train_data_page, labels).run()
```

    Downloading Data Files



```python
with open("data/target.txt","r+") as ff: trainy = Series([i.strip() for i in ff.readlines()])
trainx=read_csv("data/predictors.csv")
```


```python
trainx.info()
```


    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 494021 entries, 0 to 494020
    Data columns (total 41 columns):
    duration                       494021 non-null float64
    protocol_type                  494021 non-null object
    service                        494021 non-null object
    flag                           494021 non-null object
    src_bytes                      494021 non-null float64
    dst_bytes                      494021 non-null float64
    land                           494021 non-null object
    wrong_fragment                 494021 non-null float64
    urgent                         494021 non-null float64
    hot                            494021 non-null float64
    num_failed_logins              494021 non-null float64
    logged_in                      494021 non-null object
    num_compromised                494021 non-null float64
    root_shell                     494021 non-null float64
    su_attempted                   494021 non-null float64
    num_root                       494021 non-null float64
    num_file_creations             494021 non-null float64
    num_shells                     494021 non-null float64
    num_access_files               494021 non-null float64
    num_outbound_cmds              494021 non-null float64
    is_host_login                  494021 non-null object
    is_guest_login                 494021 non-null object
    count                          494021 non-null float64
    srv_count                      494021 non-null float64
    serror_rate                    494021 non-null float64
    srv_serror_rate                494021 non-null float64
    rerror_rate                    494021 non-null float64
    srv_rerror_rate                494021 non-null float64
    same_srv_rate                  494021 non-null float64
    diff_srv_rate                  494021 non-null float64
    srv_diff_host_rate             494021 non-null float64
    dst_host_count                 494021 non-null float64
    dst_host_srv_count             494021 non-null float64
    dst_host_same_srv_rate         494021 non-null float64
    dst_host_diff_srv_rate         494021 non-null float64
    dst_host_same_src_port_rate    494021 non-null float64
    dst_host_srv_diff_host_rate    494021 non-null float64
    dst_host_serror_rate           494021 non-null float64
    dst_host_srv_serror_rate       494021 non-null float64
    dst_host_rerror_rate           494021 non-null float64
    dst_host_srv_rerror_rate       494021 non-null float64
    dtypes: float64(34), object(7)
    memory usage: 154.5+ MB


Let's inspect the percentage of the various attack traffics.

```python
DataFrame(round((trainy.value_counts()/trainy.count())*100,3))
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
      <th>smurf.</th>
      <td>56.838</td>
    </tr>
    <tr>
      <th>neptune.</th>
      <td>21.700</td>
    </tr>
    <tr>
      <th>normal.</th>
      <td>19.691</td>
    </tr>
    <tr>
      <th>back.</th>
      <td>0.446</td>
    </tr>
    <tr>
      <th>satan.</th>
      <td>0.322</td>
    </tr>
    <tr>
      <th>ipsweep.</th>
      <td>0.252</td>
    </tr>
    <tr>
      <th>portsweep.</th>
      <td>0.211</td>
    </tr>
    <tr>
      <th>warezclient.</th>
      <td>0.206</td>
    </tr>
    <tr>
      <th>teardrop.</th>
      <td>0.198</td>
    </tr>
    <tr>
      <th>pod.</th>
      <td>0.053</td>
    </tr>
    <tr>
      <th>nmap.</th>
      <td>0.047</td>
    </tr>
    <tr>
      <th>guess_passwd.</th>
      <td>0.011</td>
    </tr>
    <tr>
      <th>buffer_overflow.</th>
      <td>0.006</td>
    </tr>
    <tr>
      <th>land.</th>
      <td>0.004</td>
    </tr>
    <tr>
      <th>warezmaster.</th>
      <td>0.004</td>
    </tr>
    <tr>
      <th>imap.</th>
      <td>0.002</td>
    </tr>
    <tr>
      <th>rootkit.</th>
      <td>0.002</td>
    </tr>
    <tr>
      <th>loadmodule.</th>
      <td>0.002</td>
    </tr>
    <tr>
      <th>ftp_write.</th>
      <td>0.002</td>
    </tr>
    <tr>
      <th>multihop.</th>
      <td>0.001</td>
    </tr>
    <tr>
      <th>phf.</th>
      <td>0.001</td>
    </tr>
    <tr>
      <th>perl.</th>
      <td>0.001</td>
    </tr>
    <tr>
      <th>spy.</th>
      <td>0.000</td>
    </tr>
  </tbody>
</table>
</div>

The traffic percentage shows approximately 20% normal traffic and 80% attack traffic. The attack traffic is dominated by smurf and neptune attacks.
- Smurf attacks are a variation of distributed denial of service attacks (DDOS) where ICMP packets with the intended target's spoofed source IP are broadcast to a computer network. The goal is to take down a single target by tricking computers on a network to receive and respond to these packets. If the number of responding hosts is relatively large, the target will be flooded with traffic.
- Neptune attack is another variation of DDOS attacks that generates a SYN flood attack against a network host by sending session synchronisation packets using forged source IPs.

## Data Attributes

The Bechmark KDDCup dataset contains 41 attributesdivided into 4 groups. The following are a decription of these attributes.

* Intrinsic Attributes: These attributes are extracted from the headers of the network packets
![Intrinsic Attributes](/assets/images/kdd1/IntrinsicAttributes.PNG)


* Content Attrinutes: These attributes are extracted from the contents area of network packetss
![Intrinsic Attributes](/assets/images/kdd1/ContentAttributes.PNG)

* Time Traffic Attributes: These are traffic attributes calculated relative to the number of conenctions in the last 2 seconds.
![Intrinsic Attributes](/assets/images/kdd1/TimeTrafficAttributes.PNG)

* Machine Traffic Attributes: These are traffic attributes calculated relative to the previous 100 connections
![Intrinsic Attributes](/assets/images/kdd1/MachineTrafficAttributes.PNG)

## Visualizing The Data

We can firther explore the data with some visualizations. First we create a correlation plot of all continous features and create line plots of correlated features to spot points of anomalies.


```python
import seaborn as sns
import matplotlib.pyplot as plt

##compute correlatiopn matrix of continous features
cormat = trainx.corr()

##generate a mask for upper triangle
mask = np.zeros_like(cormat, dtype=np.bool)
mask[np.triu_indices_from(mask)] =True

##setup a diverging color map
cmap = sns.diverging_palette(210,10,as_cmap=True)

##create figure and axis
fig,ax=plt.subplots(figsize=(10,10))

##draw heatmap
sns.heatmap(cormat,mask=mask,cmap=cmap,square=True, linewidths=.5,center=0, ax=ax)
plt.show()
```


![png](/assets/images/kdd1/CorrelationPlot.png)

Much of the correlated features are observed in the lower triangle of the correlation heatmap above. We see strong positive and negative correlations between destination host and server features. Most of the little observed inter-correlation between the derived features are expected.

We define the 'get_corr_vars' function to get features that are highly positively or negatively correlated to one or more features in the dataset beyond a certain threshold correlation coefficient (here 0.5).


```python
'''Takes in a correlation matrix and returns a list of correlated features'''
def get_corr_vars(cormat=cormat):
    correlated = []
    rows,cols = cormat.shape
    ##loops through rows and columns
    for i in range(cols):
        for j in range(i+1, cols):
            ##defines positive and negative correlation thresho;d of 0.5 and -0.5 respectively
            if cormat.values[i,j] > 0.5 or cormat.values[i,j] < -0.5:
                correlated.append(cormat.columns[i])
    return list(set(correlated))

sc=get_corr_vars()
```


```python
DataFrame(sc)
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
      <td>srv_rerror_rate</td>
    </tr>
    <tr>
      <th>1</th>
      <td>same_srv_rate</td>
    </tr>
    <tr>
      <th>2</th>
      <td>dst_host_rerror_rate</td>
    </tr>
    <tr>
      <th>3</th>
      <td>rerror_rate</td>
    </tr>
    <tr>
      <th>4</th>
      <td>dst_host_same_srv_rate</td>
    </tr>
    <tr>
      <th>5</th>
      <td>hot</td>
    </tr>
    <tr>
      <th>6</th>
      <td>serror_rate</td>
    </tr>
    <tr>
      <th>7</th>
      <td>diff_srv_rate</td>
    </tr>
    <tr>
      <th>8</th>
      <td>dst_host_serror_rate</td>
    </tr>
    <tr>
      <th>9</th>
      <td>srv_count</td>
    </tr>
    <tr>
      <th>10</th>
      <td>count</td>
    </tr>
    <tr>
      <th>11</th>
      <td>su_attempted</td>
    </tr>
    <tr>
      <th>12</th>
      <td>num_compromised</td>
    </tr>
    <tr>
      <th>13</th>
      <td>dst_host_srv_count</td>
    </tr>
    <tr>
      <th>14</th>
      <td>logged_in</td>
    </tr>
    <tr>
      <th>15</th>
      <td>dst_host_same_src_port_rate</td>
    </tr>
    <tr>
      <th>16</th>
      <td>srv_serror_rate</td>
    </tr>
  </tbody>
</table>
</div>



Now let's create a matrix of subplots to visualize correlated features. Note these features being correlated no not, at the moment, imply any prioroty usefulness in identifying good or bad connections.


```python
fig, axs = plt.subplots(6,3, figsize=(15, 15))
axs = axs.ravel()

for i in range(0,len(sc)):
    axs[i].plot(trainx[sc[i]].astype(float).diff(),color='orange')
    axs[i].set_title(sc[i])
fig.tight_layout()
```


![png](/assets/images/kdd1/traffic_matrix.png)


Notice the similarity in outlier points across all correlated features. Now let's begin our learning task with unsupervised learning.

In the next part of this series, we will explore various unsupervised learning approaches to extract hidden patterns in our attack traffic. We will create a group of clusters with the predictor features of our attack traffic to create an attack taxonomy for grouping attacks.
