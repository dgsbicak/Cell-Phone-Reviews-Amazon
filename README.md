# Cell-Phone-Reviews-Amazon

## Introduction
I decided to do a NLP project for three main reasons:
```
1-Text data is the cheapest and most abundant resource in the internet.
2-I wanted to gather my own data and use my knowledge on a real case.
3-I will use the models here on other projects.
```

I managed to gather almost 200k samples, and after preprocessing I was left with 180k samples in total.
Both stemming and lemmatizing didn't yield better results due to diminished information in the data, that is why they are not presented in current project.
Sample size numbers for review stars are shown below:
```
    5    100006
    1     40217
    4     25010
    3     13296
    2     11450
```
Note: In the previous version, the goal was to predict whether someone liked or disliked a phone (0 or 1). Achieved over 0.91 AUC and 94% F1 score on predicting.


## Importing Libraries
```
import re
import csv
import pandas as pd
import numpy as np
import pickle
import time
import string
import os
os.chdir(r"C:\Users\dogus\OneDrive\Masaüstü\DgsPy\DgsPy_DBOX\Amazon Project")

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import matplotlib.pyplot as plt
plt.style.use('ggplot')
%matplotlib inline

from nltk.corpus import stopwords
from contextlib import contextmanager
import eli5

from collections import Counter

from imblearn.under_sampling import NearMiss, RandomUnderSampler
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import ShuffleSplit, learning_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import f1_score, log_loss,precision_score, recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.utils import class_weight

from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

import lightgbm as lgb
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

import gc

import tensorflow as tf
from tensorflow.keras import backend as K
num_cores = 4

CPU=False
GPU=True
if GPU:
    num_GPU = 1
    num_CPU = 1
if CPU:
    print("Keras will run on CPU")
    num_CPU = 1
    num_GPU = 0

config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
                        inter_op_parallelism_threads=num_cores, 
                        allow_soft_placement=True,
                        device_count = {'CPU' : num_CPU,
                                        'GPU' : num_GPU}
                       )

session = tf.Session(config=config)
K.set_session(session)

from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras import optimizers
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

import warnings
warnings.filterwarnings("ignore")
```


## Functions Used
```
def combine_CSVs_infolder():
    os.chdir(r"C:\Users\dogus\Dropbox\DgsPy_DBOX\Amazon Project\comments_AMAZON")
    filenames = os.listdir()
    comb = pd.concat( [pd.read_csv(f) for f in filenames])
    comb.to_csv('AMAZON_comments_yuge.csv', index=False)

def clean_str(text):
    try:
        text = ' '.join( [w for w in text.split()] )        
        text = text.lower()
        text = re.sub(u"é", u"e", text)
        text = re.sub(u"ē", u"e", text)
        text = re.sub(u"è", u"e", text)
        text = re.sub(u"ê", u"e", text)
        text = re.sub(u"à", u"a", text)
        text = re.sub(u"â", u"a", text)
        text = re.sub(u"ô", u"o", text)
        text = re.sub(u"ō", u"o", text)
        text = re.sub(u"ü", u"u", text)
        text = re.sub(u"ï", u"i", text)
        text = re.sub(u"ç", u"c", text)
        text = re.sub(u"\u2019", u"'", text)
        text = re.sub(u"\xed", u"i", text)
        text = re.sub(u"w\/", u" with ", text)
        
        text = re.sub(u"[^a-z0-9]", " ", text)  # delete unnecessary characters
        text = u" ".join(re.split('(\d+)',text) )
        text = re.sub( u"\s+", u" ", text ).strip()
        text = ''.join(text)
    except:
        text = np.NaN
    return text

def clean_int2(text):
    output = re.sub(r'\d+', '', str(text))
    return output

def sorttext(x):
    iscolor = x[3][:5]=='Color'
    isprovider = x[3][:17]=='Service Provider:'
    issize = x[3][:5]=='Size:'
    isstyle = x[3][:6]=='Style:'
    
    if iscolor | isprovider | issize:
        if 'helpful' in x[-3]:
            x = " ".join(x[4:-3])
        else:
            x = " ".join(x[4:-2])
    else:
        if 'helpful' in x[-3]:
            x = " ".join(x[4:-3])
        else:
            x = " ".join(x[4:-2])
    return x

def goBar(variables,name,text=None):
    trace= go.Bar(
            x=variables.index,
            y=variables.values,
            text=text,
            textposition='auto',
            marker=dict(
                color=list(range(len(variables)))
                ),
            )
    layout = go.Layout(
        title = name
        )

    data = [trace]
    fig = go.Figure(
                    data=data,
                    layout=layout
                   )
    py.iplot(fig)

def plot_learning_curve(estimator, title, X, y, scoring=None, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.subplots(figsize=(12,6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y,
                                                            cv=cv, n_jobs=n_jobs,
                                                            scoring=scoring,
                                                            train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.savefig('learningcurves.png')
    return plt
    
def model_eval(model, k=5, seed=0):
    kfold = StratifiedKFold(k, shuffle=True,random_state=seed)
    oof = np.zeros(y.shape[0])
    for nfold, (train_ix, valid_ix) in enumerate(kfold.split(X,y)):
        trainX, validX = X[train_ix], X[valid_ix]
        trainy, validy = y[train_ix], y[valid_ix]
        
        model.fit(trainX, trainy)
        p = model.predict(validX)
        oof[valid_ix] = p
        print('Fold{}, F1_micro : {:.2%}'.format(nfold+1,f1_score(validy, p, average='micro')))
    
    print(confusion_matrix(y, oof))
    print(classification_report(y, oof, target_names=["1 Star", "2 Stars", "3 Stars", "4 Stars", "5 Stars"]))
    print("F1_micro : {:.2%} ".format(f1_score(test_y, p, average='micro')))
    return model, oof

def simple_eval(model,xtrain,ytrain):
    model.fit(xtrain, ytrain)
    p = model.predict(test_X)

    print(classification_report(test_y, p, target_names=["1 Star", "2 Stars", "3 Stars", "4 Stars", "5 Stars"]))
    print("F1_micro/Acc : {:.2%} ".format(f1_score(test_y, p, average='micro')))
    return model, p

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(18, 8))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

def pickle_model(model, filename):
    try: os.mkdir("models")
    except:pass
    pickle.dump(model, open("models/"+filename, 'wb'))
```

## Read the merged data
```
df = pd.read_csv('AMAZON_comments_yuge.csv')
df = df.drop_duplicates(subset='Text', keep=False)
df.reset_index(inplace=True,drop=True)
print(df.info())
```
Output:
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 199672 entries, 0 to 199671
Data columns (total 3 columns):
Text           199672 non-null object
Phone Title    199672 non-null object
Stars          199672 non-null int64
dtypes: int64(1), object(2)
memory usage: 4.6+ MB
None
```

## Gather phone model names from the data
```
modellist = list(set(df['Phone Title'].values))
modellist = list(map(lambda model: ''.join([e for e in model if e not in set(string.punctuation)]),modellist))
brands = []
models_1 = []
models_2 = []
for model in modellist:
    if len(model.split())>3:
        brand = model.split()[0]
        brands.append(brand)

        m = model.split()[1]
        models_1.append(m)

        m = model.split()[2]
        models_2.append(m)
    else:pass
models_1 = set(map(lambda x: x.lower().strip(),set(models_1)))
models_2 = set(map(lambda x: x.lower().strip(),set(models_2)))
models = list(models_1.union(models_2))# Merge the two models set, because they contain some duplicate names.
brands = list(map(lambda x: x.lower().strip(),set(brands)))
```

## Brand Review Frequency
Imbalanced, high tier brand review data might damage the generalization capability of models. However, I was limited with what Amazon presented to me, and choosed not to download ready made data.
```
brands = df['Phone Title'].apply(lambda x: x.upper().split()[0]).value_counts()[:8]
brands
```
Output:
```
SAMSUNG       81579
APPLE         56107
LG            31870
BLACKBERRY    17989
VERIZON        3333
HTC            1365
XIAOMI         1028
ONEPLUS         940
Name: Phone Title, dtype: int64
```

```
goBar(brands, 'Brand Review Frequency')
```
Output:

![newplot](https://user-images.githubusercontent.com/23128332/61597821-924c1700-ac1d-11e9-8d9f-0af4e09c14f9.png)

## Gender Ratio
The data doesn't contain information about genders of the reviewers. A firstname database will be used to estimate genders from firstnames. This approach can provide an approximate gender ratio information.
```
"""
https://github.com/MatthiasWinkelmann/firstname-database
F female
1F female if first part of name, otherwise mostly male
?F mostly female
M male
1M male if first part of name, otherwise mostly female
?M mostly male
? unisex
"""
gnames = pd.read_csv("firstnames.csv",sep=';', usecols=['name', 'gender'])
gnames.dropna(axis=0,inplace=True)
gnames['name'] = gnames['name'].apply(lambda x: x.lower())
gnames.columns = ['fname', 'gender']
gnames.drop_duplicates(subset=['fname'],inplace=True)

gnames.gender = gnames.gender.apply(lambda x: 'F' if 'F' in x else ('M' if 'M' in x else ('U' if x=='?' else '???')))

df['fname'] = df['Name'].apply(lambda x: x.lower() if len(x.split())<2 else x.split()[0].lower())  # Leave the first name (for the most cases)

df = pd.merge(df,gnames, how='left', on='fname')

gg = df['gender'].value_counts()
#M : 56%
#F : 42%
#U : 2%

goBar(gg,'Gender Bar',text=['56%','42%','2%'])
```
![newplot (1)](https://user-images.githubusercontent.com/23128332/63115439-dbcf1e00-bf9f-11e9-859b-e2924698b23a.png)

### A sample from the text data
```
df['Text'][85101]
```
output:
```
'Kenneth B.\r\r\nFive Stars\r\r\nJune 5, 2016\r\r\nStyle: U.S. Version (LGUS991)Verified Purchase\r\r\nNice phone\r\r\nHelpful\r\r\nComment Report abuse'
```

If we split the example from the parts where '\r\r\n' exists, we sort the Text data much easier.
```
df['Text'][85101].split('\r\r\n')
```
output:
```
['Kenneth B.',
 'Five Stars',
 'June 5, 2016',
 'Style: U.S. Version (LGUS991)Verified Purchase',
 'Nice phone',
 'Helpful',
 'Comment Report abuse']
```

## Preprocessing
Stemming, Lemmatization wasn't used and stopwords wasn't discarted because it leads to decrease in information and accuracy (~-5%)
```
df['Text'] = df['Text'].apply(lambda x: x.split('\r\r\n'))  # Seperate text into parts
df['Name'] = df['Text'].apply(lambda x: x[0])  # Commentator Names
df['Title'] = df['Text'].apply(lambda x: x[1]) # Review Titles

# Get the actual customer review by discarding unnecessary parts.
df['Text'] = df['Text'].apply(sorttext) 

# Make everything lowercase for the simplicity.
df['Text'] = df['Text'].apply(lambda x: x.lower())
df['Title'] = df['Title'].apply(lambda x: x.lower())

# Clean any integer value
df['Text'] = df['Text'].apply(clean_int2)  

# Titles contain the summary information of customer reviews, titles should be added into Text column but some contain Target info.
df['Title'] = df['Title'].apply(lambda x: "" if ('star' in x.split())|('stars' in x.split()) else x)

# Merge Comments and Titles
df['Text'] = df['Text']+' '+df['Title']
df = df[['Text','Stars','Phone Title']]

# Some reviews contain target information i.e. 'gave two stars'
leakage_list = "gave one two three four five star stars".split()
df['Text'] = df['Text'].apply(lambda text: ' '.join([word for word in text.split() if word not in leakage_list]))

# Drop NANs if any exist
if df.isna().any()!=0:
    print("NANs in DF:\n",df.isna().sum())
    df.dropna(axis=0,inplace=True)
    df.reset_index(inplace=True,drop=True)

# Delete Unnecessary Text
df['Text'] = df['Text'].apply(lambda x: x.replace('verified purchase',""))
df['Text'] = df['Text'].apply(lambda x: x.replace('amazon',""))
df['Text'] = df['Text'].apply(lambda x: x.replace('verizon',""))

print(df['Text'][0],"\n")
print('Deleting model informations')
df['Text'] = df['Text'].apply(lambda text: ' '.join([word for word in text.split() if word not in models]))

print('Deleting brand names...')
df['Text'] = df['Text'].apply(lambda text: ' '.join([word for word in text.split() if word not in brands]))

#print('Deleting stopwords...')
#df['Text'] = df['Text'].apply(lambda text: ' '.join([word for word in text.split() if word not in set(stopwords.words('english'))]))
#print(df['Text'][0],"\n")

print("Adjusting characters, trimming escape characters, symbols...\n")
df['Text'] = df['Text'].apply(clean_str)
print(df['Text'][0],"\n")

print('Deleting punctuations...')
df['Text'] = df['Text'].apply(lambda ftext: ''.join([e for e in ftext if e not in set(string.punctuation)]))
print(df['Text'][0],"\n")
```
Output:
```
Preprocessing Comments.. 

# Make everything lowercase for the simplicity.
# Merge Comments and Titles
it works really well a couple times it turned off and tried to reboot itelf. it could've just been like updating i'm not sure, this is my first iphone but other than that it works really well. it rebooted more time but it hasn't done it again. there were no scratches or anything. it looks really nice and everything works so far. the seller even emailed me and said if i wasn't 100% satisfied to contact them. i am very happy with the phone and it even worked with straight talk. would buy again 😊 

# Delete Unnecessary Text

it works really well a couple times it turned off and tried to reboot itelf. it could've just been like updating i'm not sure, this is my first iphone but other than that it works really well. it rebooted more time but it hasn't done it again. there were no scratches or anything. it looks really nice and everything works so far. the seller even emailed me and said if i wasn't 100% satisfied to contact them. i am very happy with the phone and it even worked with straight talk. would buy again 😊 

# Deleting model informations

# Deleting brand names...

# Adjusting characters, trimming escape characters, symbols...

it works really well a couple times it turned off and tried reboot itelf it couldve just been like updating im not sure this is my but other than that it works really well it rebooted more time but it hasnt done it again there were scratches anything it looks really nice and everything works so far the seller even emailed me and said if i wasnt satisfied contact them i am very happy the and it even worked talk would buy again 

# Discard rows with empty Text...

# Discard rows that have less than 2 characters...

```

## Outliers
```
# Measure Text Lengths
df['textlength'] = df['Text'].apply(lambda x: len(x))
df['textlength'].describe()
```
Output:
```
count    199672.000000
mean        220.878125
std         494.201899
min           0.000000
25%          33.000000
50%         102.000000
75%         224.000000
max       25699.000000
Name: textlength, dtype: float64
```

```
# Plot the length of every comment, and discard outliers
trace = go.Scattergl(
    x = df['textlength'].index,
    y = df['textlength'].values,
    mode='markers',
    marker=dict(opacity=0.1)
)
data = [trace]
py.iplot(data, filename='Text Length')
```
Output:
![newplot](https://user-images.githubusercontent.com/23128332/58370096-3873ed00-7f0b-11e9-825d-e67a50fe1b72.png)

Some of the comments don't reflect the population; some of them are spams, some of them are overenthusiastic people writing too long comments, and so on. In our refined data, average length of reviews is 142, which means 142 characters found (including blank space characters) on average, in every review. I decided to leave out every comment that is longer than 500 characters.

```
df['textbool'] = df['textlength'].apply(lambda x: 1 if x<500 else 0)
df = df[df['textbool']==1]
```

## Target Distribution

```
starsfq = df['Stars'].value_counts()
starsfq
```
Output:
```
5    100006
1     40217
4     25010
3     13296
2     11450
Name: Stars, dtype: int64
```

```
goBar(starsfq, 'Star Counts')
```
Output:

![newplot (1)](https://user-images.githubusercontent.com/23128332/58369284-e9c15580-7f00-11e9-8dbb-3ddd9b03138b.png)

## Wordcloud
```
# Randomly picked 10000 sample texts merged into one big text
tt = []
for text in df['Text'].sample(frac=1.0)[:10000]:
    tt.append(text)
TEXT = " ".join(tt)
```

```
from wordcloud import WordCloud, STOPWORDS

wordcloud = WordCloud(
    background_color='white',
    stopwords=STOPWORDS,
    max_words=200,
    max_font_size=100,
    random_state=11,
    width=800,
    height=400,
    )
wordcloud.generate(TEXT)

plt.figure(figsize=(24,12))
plt.imshow(wordcloud)
```
Output:

![wordcld](https://user-images.githubusercontent.com/23128332/58369328-8edc2e00-7f01-11e9-9a87-49bda97d17e0.png)

## Word Frequency Plots
```
from collections import defaultdict
from plotly import tools

df5 = df.loc[df["Stars"]==5,'Text']
df4 = df.loc[df["Stars"]==4,'Text']
df3 = df.loc[df["Stars"]==3,'Text']
df2 = df.loc[df["Stars"]==2,'Text']
df1 = df.loc[df["Stars"]==1,'Text']

## custom function for ngram generation ##
def generate_ngrams(text, n_gram=1):
    token = [token for token in text.lower().split(" ") if token != "" if token not in STOPWORDS]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [" ".join(ngram) for ngram in ngrams]

## custom function for horizontal bar chart ##
def horizontal_bar_chart(df, color):
    trace = go.Bar(
        y=df["word"].values[::-1],
        x=df["wordcount"].values[::-1],
        showlegend=False,
        orientation = 'h',
        marker=dict(
            color=color,
        ),
    )
    return trace

## 5 Starred Texts ##
freq_dict = defaultdict(int)
for sent in df5:
    for word in generate_ngrams(sent):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace0 = horizontal_bar_chart(fd_sorted.head(50), 'green')

## 3 Starred Texts ##
freq_dict = defaultdict(int)
for sent in df3:
    for word in generate_ngrams(sent):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace1 = horizontal_bar_chart(fd_sorted.head(50), 'blue')

## 1 Starred Texts ##
freq_dict = defaultdict(int)
for sent in df1:
    for word in generate_ngrams(sent):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace2 = horizontal_bar_chart(fd_sorted.head(50), 'orange')

# Creating two subplots
fig = tools.make_subplots(rows=1, cols=3, vertical_spacing=0.04,
                          subplot_titles=["5 Starred: Word Frequency",
                                          "3 Starred: Word Frequency", 
                                          "1 Starred: Word Frequency"])
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig.append_trace(trace2, 1, 3)
fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots")
py.iplot(fig, filename='word-plots')
```
Output:

![newplot (1)](https://user-images.githubusercontent.com/23128332/65023401-6a3a1500-d93b-11e9-9bc4-6a52c5dee813.png)

## Bigram Count Plots

```
freq_dict = defaultdict(int)
for sent in df5:
    for word in generate_ngrams(sent,2):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace0 = horizontal_bar_chart(fd_sorted.head(50), 'orange')


freq_dict = defaultdict(int)
for sent in df1:
    for word in generate_ngrams(sent,2):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace1 = horizontal_bar_chart(fd_sorted.head(50), 'orange')

# Creating two subplots
fig = tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04,horizontal_spacing=0.15,
                          subplot_titles=["5 Starred: Word Bigram Frequency", 
                                          "1 Starred: Word Bigram Frequency"])
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Bigram Count Plots")
py.iplot(fig, filename='word-plots')
```
Output:

![newplot](https://user-images.githubusercontent.com/23128332/65023366-568eae80-d93b-11e9-91ff-45f1fe7bd626.png)

# Part 2: Algorithm Training

```
# Load the Data
df = pd.read_csv('cleaned_amazon_yuge.csv')
df = df.sample(frac=1.0, random_state=13).reset_index(drop=True)

X, test_X, y, test_y = train_test_split(df['Text'], df['Stars'], test_size=0.2, random_state=10, shuffle=True)
```

## Vectorization
### TfidfVectorizer
```
tfidf = TfidfVectorizer(token_pattern=r'\w{1,}',
                        ngram_range=(1, 3),
                        max_df=0.5,
                        min_df=3,
                        max_features=100000,
                        strip_accents='unicode'
                        #decode_error='ignore',
                       )

tfidf.fit(X)
X_tf = tfidf.transform(X)
test_X_tf = tfidf.transform(test_X)
```
### CountVectorizer
```
from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer(ngram_range=(1, 3),
                            max_df=0.5,
                            analyzer='word',
                            token_pattern = r'\w{1,}',
                       max_features=100000)
count.fit(X)
X_cv = count.transform(X)
test_X_cv = count.transform(test_X)
```

## Model Building
### Tfidf & CountVectorizer & Scaling
Model shows better performance with Tfidf method. Tfidf will be used later on.
```
# Tfidf Not Scaled
# Tfidf method scales the data, usually additional scaling isn't needed.
logr = LogisticRegression(solver='sag',multi_class="multinomial",n_jobs=-1)
a = simple_eval(logr, X_tf, y, test_X_tf)

>>
F1_micro/Acc : 73.54% 
Log Loss: 0.72

# CountVectorizer Not Scaled
logr = LogisticRegression(solver='sag',multi_class="multinomial",n_jobs=-1)
a = simple_eval(logr, X_cv, y, test_X_cv)

>>
F1_micro/Acc : 72.10% 
Log Loss: 0.85
```

```
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize

X_tf_l2 = normalize(X_tf)
X_tf_l2_te = normalize(test_X_tf)

X_cv_l2 = normalize(X_cv)
X_cv_l2_te = normalize(test_X_cv)

# Scaled Data
# Tfidf Used
logr = LogisticRegression(solver='sag',multi_class="multinomial",n_jobs=-1)
a = simple_eval(logr, X_tf_l2, y, X_tf_l2_te)

>>
tfidf
F1_micro/Acc : 73.54% 
Log Loss: 0.72

# CountVectorizer Used
logr = LogisticRegression(solver='sag',multi_class="multinomial",n_jobs=-1)
a = simple_eval(logr, X_cv_l2, y, X_cv_l2_te)

>>
CVec
F1_micro/Acc : 73.44% 
Log Loss: 0.72
```

## Dealing with Imbalanced Classes
Logistic regression gives good performance. After testing various over & under sampling techniques, the method that gives the best results will be chosen.

The model's ability to predict 1 & 5 starred reviews is very good, however 2nd,3rd and 4th classes has very low recall values(High false negatives), and low precision values (Low true positives).
Reasons for this:

1-) Imbalanced data can cause the model to ignore minority classes and overfit the majority class.
2-) Needs more data. 10-20k text data is not enough to help the model successfully distinguish a 2 or 3 starred review from other classes.
3-) It would be very hard even for a human to distinguish between 2 or 3 starred review from one another or 4 and 5 starred review. A fair error rate is expected.
4-) 1 & 5 starred reviews are two opposite end of the spectrum. Those reviews contain distinctive words more often, such as "garbage","sucks","excellent", etc. 

### Assigning Class Weights
Assigning higher class weights causes the model to be more punishing on false predictions. 
There is a trade off between majority and minority class scores, when minority (2,3,4) recall values get higher, majority (1 and 5) recall values get lower which causes the model accuracy to be lower, because of the amount of support is much higher and lower recall has a greater impact on overall score.
Trying out different weights didn't help as well.
```
class_weights = class_weight.compute_class_weight('balanced',np.unique(y),y)
class_weights = {clss+1: weight for clss, weight in enumerate(class_weights)}
print(class_weights)

# Training on the standard data with class weights.
logr = LogisticRegression(solver='sag',multi_class="multinomial",class_weight=class_weights,n_jobs=-1)
a = simple_eval(logr, X_tf, y, test_X_tf)
```
Output:
```
{1: 0.9452688782060852, 2: 3.191008970869872, 3: 2.780187933608501, 4: 1.4746599590087572, 5: 0.38596491228070173}
              precision    recall  f1-score   support

      1 Star       0.76      0.72      0.74      8390
     2 Stars       0.23      0.32      0.27      2450
     3 Stars       0.28      0.38      0.32      2807
     4 Stars       0.34      0.45      0.39      5166
     5 Stars       0.90      0.75      0.82     20760

    accuracy                           0.65     39573
   macro avg       0.50      0.53      0.51     39573
weighted avg       0.71      0.65      0.68     39573

F1_micro/Acc : 65.26% 
Log Loss: 0.87
```

### Over-Sampling Using SMOTE, ADASYN,RandomOverSampler
SMOTE and other over-sampling methods didn't help as well. Pretty much resulted the same as before, recall trade-off between classes.
The major issue is the scarcity of information in the data. Over sampling doesn't increase the overall information.
```
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
# Standard Data Class Distribution
print("Class Distribution for Standard Data:\n",Counter(y))

# Apply SMOTE on Tfidf vector
smote = SMOTE(sampling_strategy='auto', random_state=10,n_jobs=-1)
X_smote, y_smote = smote.fit_sample(X_tf,y)
print("Class Distribution for SMOTE:\n",Counter(y_smote))

ada = ADASYN(sampling_strategy='not majority', random_state=10,n_jobs=-1)
X_ada, y_ada = ada.fit_sample(X_tf,y)
print("Class Distribution for ADASYN:\n",Counter(y_ada))

rsamp = RandomOverSampler(random_state=10)
X_ra, y_ra = rsamp.fit_sample(X_tf,y)
print("Class Distribution for RandomOverSampler:\n",Counter(y_ra))
```
Output:
```
Class Distribution for Standard Data:
 Counter({5: 82023, 1: 33491, 4: 21468, 3: 11387, 2: 9921})
Class Distribution for SMOTE:
 Counter({5: 82023, 1: 82023, 3: 82023, 4: 82023, 2: 82023})
Class Distribution for ADASYN:
 Counter({2: 84952, 4: 82561, 3: 82370, 5: 82023, 1: 78497})
Class Distribution for RandomOverSampler:
 Counter({5: 82023, 1: 82023, 3: 82023, 4: 82023, 2: 82023})
```

#### Synthetic Minority Over-sampling (SMOTE) Method Result
SMOTE and other over-sampling methods didn't help as well. Pretty much resulted the same as before, recall trade-off between classes.
The major issue is the scarcity of information in the data. Over sampling doesn't increase the overall information.


#### Apply SMOTE on Tfidf vector
```
logr = LogisticRegression(solver='sag',multi_class="multinomial",n_jobs=-1)
a = simple_eval(logr, X_smote, y_smote, test_X_tf)
```
Output:
```
      1 Star       0.74      0.76      0.75      7480
     2 Stars       0.21      0.24      0.22      2132
     3 Stars       0.28      0.34      0.31      2474
     4 Stars       0.33      0.37      0.35      4667
     5 Stars       0.87      0.78      0.82     19142

    accuracy                           0.66     35895
   macro avg       0.48      0.50      0.49     35895
weighted avg       0.69      0.66      0.67     35895
F1_micro/Acc : 66.32% 
Log Loss: 0.85
```

#### Adaptive Synthetic (ADASYN) Method Result
```
logr = LogisticRegression(solver='sag',multi_class="multinomial",n_jobs=-1)
a = simple_eval(logr, X_ada, y_ada, test_X_tf)
```
Output:
```
F1_micro/Acc : 66.35% 
Log Loss: 0.85
```

#### Random Over-Sampler Method Result
```
logr = LogisticRegression(solver='sag',multi_class="multinomial",n_jobs=-1)
a = simple_eval(logr, X_ra, y_ra, test_X_tf)
```
Output:
```
F1_micro/Acc : 65.55% 
Log Loss: 0.86
```

### Under-sampling Using NearMiss, RandomUnderSampler

```
from imblearn.under_sampling import NearMiss, RandomUnderSampler

nm1 = NearMiss(version=1,random_state=10,n_jobs=-1)
nm2 = NearMiss(version=2,random_state=10,n_jobs=-1)
nm3 = NearMiss(version=3,random_state=10,n_jobs=-1)
rus = RandomUnderSampler(random_state=10)
```

NearMiss Version-1
```
X_nm1, y_nm1 = nm1.fit_sample(X_tf,y)
print("Class Distribution for Nearmiss Version 1:\n",Counter(y_nm1))

logr = LogisticRegression(solver='sag',multi_class="multinomial",n_jobs=-1)
a= simple_eval(logr,X_nm1, y_nm1, test_X_tf)
del X_nm1, y_nm1

>> 
F1_micro/Acc : 37.16% 
Log Loss: 1.37
```

NearMiss Version-2
```
X_nm2, y_nm2 = nm2.fit_sample(X_tf,y)
print("Class Distribution for Nearmiss Version 2:\n",Counter(y_nm2))

logr = LogisticRegression(solver='sag',multi_class="multinomial",n_jobs=-1)
a = simple_eval(logr,X_nm2, y_nm2, test_X_tf)
del X_nm2, y_nm2

>> 
F1_micro/Acc : 61.65% 
Log Loss: 0.98
```

NearMiss Version-3
```
>>
F1_micro/Acc : 15.94% 
Log Loss: 2.61
```

Random Under Sampler
```
>>
F1_micro/Acc : 61.54% 
Log Loss: 0.96
```

#### Under-sampling Results
RandomUnderSampler and Nearmiss version-2 yields best results among other under sampling methods, but fails to pass the success of the first model that trained on the original data mainly due to the scarcity of training data and inability to distinguish between similar classes. ~40% recall ~25% precision on 2-3-4 Star classes indicates the model is trying hard to find them, but fails.
This model can be used in stacking and blending, because of the relative success in finding minority classes.


### Moving on with the original data and no class weight assignment

### LogisticRegression

#### Tuning Hyperparameters with GridSearch
```
from sklearn.model_selection import GridSearchCV
def HyperOpti(trainX, trainY, model, params, k=5):
    """
    params: Takes 1D dictionary
    """
    grid = GridSearchCV(model, cv=k, param_grid=params)
    grid.fit(trainX, trainY)
    
    results = {}
    for x,y in zip(*params.values(), grid.cv_results_['mean_test_score']):
        results[str(x)] = y
    results = pd.DataFrame(results,index=[0])

    
    ax = sns.barplot(data=results)
    ax.set_ylabel('Accuracy', size=15)
    ax.set_xlabel('Parameter', size=15)
    ax.tick_params(labelsize=14)
    
    best=grid.best_params_.popitem()
    print("Best parameter is {}".format(best))
    
    return best
```

```
%%time
# Training on the standard data with class weights.
params = {'C': [1e-5, 1e-3, 1e-1, 1e0, 1e1, 1e2]}
logr = LogisticRegression(solver='sag', multi_class="multinomial", n_jobs=-1)
p, a = HyperOpti(X_tf, y, logr, params)
logr.C = a
>>
```
![logbestp](https://user-images.githubusercontent.com/23128332/64973413-a5910100-d8b3-11e9-9c34-273af1193052.JPG)


```
logr, logr_probs = simple_eval(logr, X_tf, y, test_X_tf)

>>
              precision    recall  f1-score   support

      1 Star       0.70      0.89      0.79      7480
     2 Stars       0.28      0.05      0.08      2132
     3 Stars       0.39      0.20      0.26      2474
     4 Stars       0.46      0.21      0.29      4667
     5 Stars       0.80      0.95      0.87     19142

    accuracy                           0.74     35895
   macro avg       0.53      0.46      0.46     35895
weighted avg       0.68      0.74      0.69     35895

F1_micro/Acc : 73.54% 
Log Loss: 0.72
```


Image below shows the top Tfidf features and their weights measured by the first logistic regression model, which trained on the original data and had no class weights assigned.

At first glance, we can see that there is no leakage left in the data. Some review texts contained target information, such as "gave two stars"; also phone brand and model information could had a similar affect.

It's hard to distinguish close classes from one another. You can see common words in different classes, such as "perfect","awesome","love", etc. are common in 4th and 5th classes. 1 and 2 starred reviews share features like "defective", "not", "not a good", "bad", etc.

Learning graph shows high bias and high variance, the model needs much more high quality data to improve the results
-We can't see from the top features but there are few garbage data in the reviews (spam, misleading review, etc.) which decreases the overall quality of the data a bit.

```
eli5.show_weights(logr, vec=tfidf, top=40)
```
Output:
![logwords](https://user-images.githubusercontent.com/23128332/63623212-b1194100-c601-11e9-961c-500926cf4b2f.JPG)

```
plot_learning_curve(logr,'Logistic Regression', X=X, y=y, n_jobs=4)
```
Output:
![LogR](https://user-images.githubusercontent.com/23128332/63638115-c5eee680-c68c-11e9-9f67-e55edea3ad8d.JPG)

## Stochastic Gradient Descent Classifier
```
params = {'alpha':[1e-10, 1e-7, 1e-5, 1e-3, 1e-1, 1e-0]}
sgd = SGDClassifier(loss='modified_huber', penalty='l2', n_jobs=-1)
p, a= HyperOpti(X_tf, y, sgd, params)
sgd.alpha = a

#sgd, sgd_preds = model_eval(sgd)
sgd, sgd_preds = simple_eval(sgd,X,y)
>>
```
![image](https://user-images.githubusercontent.com/23128332/64973652-0fa9a600-d8b4-11e9-9bd4-0f461c176beb.png)

```
              precision    recall  f1-score   support

      1 Star       0.71      0.85      0.78      7480
     2 Stars       0.24      0.08      0.12      2132
     3 Stars       0.33      0.21      0.26      2474
     4 Stars       0.40      0.22      0.28      4667
     5 Stars       0.81      0.93      0.87     19142

    accuracy                           0.72     35895
   macro avg       0.50      0.46      0.46     35895
weighted avg       0.67      0.72      0.69     35895

F1_micro/Acc : 72.32% 
Log Loss: 2.27
```

```
plot_learning_curve(sgd,'SGD', X=X, y=y, scoring='accuracy', n_jobs=4)
```
Output:
![SGD](https://user-images.githubusercontent.com/23128332/63623221-b1b1d780-c601-11e9-9982-b811c1ae08c0.JPG)

```
eli5.show_weights(sgd, vec=tfidf, top=40)
```
Output:
![SGDwords](https://user-images.githubusercontent.com/23128332/63623208-b080aa80-c601-11e9-982c-dc232d7e16fa.JPG)


# MultinomialNB()
```
MNB = MultinomialNB()
#MNB, MNB_preds = model_eval(MNB)
MNB, MNB_probs = simple_eval(MNB, X_tf, y, test_X_tf)
```
Output:
```
              precision    recall  f1-score   support

      1 Star       0.65      0.91      0.76      7480
     2 Stars       0.00      0.00      0.00      2132
     3 Stars       0.53      0.01      0.02      2474
     4 Stars       0.45      0.10      0.17      4667
     5 Stars       0.76      0.97      0.85     19142

    accuracy                           0.72     35895
   macro avg       0.48      0.40      0.36     35895
weighted avg       0.64      0.72      0.64     35895

F1_micro/Acc : 72.11% 
Log Loss: 0.84
```

## RandomForestClassifier
```
tree = RandomForestClassifier(max_depth=300, n_estimators=50, verbose=1, n_jobs=-1)
#tree , tree_preds = model_eval(tree, k=3)
tree , tree_probs = simple_eval(tree,X_tf, y, test_X_tf)
```
Output:
```
      1 Star       0.65      0.85      0.73      7480
     2 Stars       0.26      0.01      0.02      2132
     3 Stars       0.34      0.02      0.04      2474
     4 Stars       0.43      0.04      0.07      4667
     5 Stars       0.73      0.97      0.83     19142

    accuracy                           0.70     35895
   macro avg       0.48      0.38      0.34     35895
weighted avg       0.62      0.70      0.61     35895

F1_micro/Acc : 70.19% 
Log Loss: 0.91
```

```
std = np.std([ree.feature_importances_ for ree in tree.estimators_], axis=0).astype(np.float32)
imps = pd.DataFrame({
    'Features':tfidf.get_feature_names(),
    'Importances':tree.feature_importances_.astype(np.float32),
    'Stds':std
    }
            ).sort_values(by='Importances', ascending=False).reset_index(drop=True)
imps.to_csv('important_features.csv',index=False)

trace = go.Bar(x=imps.Features[:20],
                y=imps.Importances[:20],
                marker=dict(color='red'),
                error_y = dict(visible=True, arrayminus=imps.Stds[:20]),
                opacity=0.5
               )
layout = go.Layout(title="RandomForest Feature Importances")
fig = go.Figure(data=[trace], layout=layout)
py.iplot(fig)
```
Output:
![image](https://user-images.githubusercontent.com/23128332/65023077-cb151d80-d93a-11e9-99e4-771bce42069c.png)


## Lightgbm
```
model = LGBMClassifier(objective='softmax')
lgb_model, lgb_probs = simple_eval(model,X_tf,y,test_X_tf)
```
Output:
```
              precision    recall  f1-score   support

      1 Star       0.67      0.84      0.75      7480
     2 Stars       0.31      0.03      0.06      2132
     3 Stars       0.38      0.13      0.20      2474
     4 Stars       0.44      0.18      0.26      4667
     5 Stars       0.77      0.95      0.85     19142

    accuracy                           0.71     35895
   macro avg       0.51      0.43      0.42     35895
weighted avg       0.65      0.71      0.66     35895

F1_micro/Acc : 71.49% 
Log Loss: 0.78
```

# Neural Networks
Embedding, LSTM, GRU and other tactics resulted in limited success. 

Transfer Learning might help (Spacy, GloVe embeddings).

```
df = pd.read_csv('cleaned_amazon_yuge.csv')
df = df.dropna(axis=0).sample(frac=1, random_state=13).reset_index(drop=True)
X = df['Text']
y = df['Stars']

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
ohe.fit(y.values.reshape(-1,1))
max_len=200
tokenizer = text.Tokenizer(num_words=max_len)
tokenizer.fit_on_texts(X)

X = tokenizer.texts_to_sequences(X)
X = sequence.pad_sequences(X, maxlen = 200).astype(np.int32)
y = ohe.transform(y.values.reshape(-1,1)).astype(np.uint8)

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=10)
del df, X, y
gc.collect()
```
## Transfer Learning

### Pretrained GloVe 300D Embedding
```
f = open('glove.840B.300d/glove.840B.300d.txt', encoding='utf-8')
embeddings_index={}
for line in f:
    values = line.split()
    word = values[0]
    try:
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    except ValueError:
        pass
f.close()

word_index = tokenizer.word_index
embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
```
```
from keras.callbacks import EarlyStopping
earlystop = EarlyStopping(monitor='val_loss',patience=3, min_delta=0, verbose=0, mode='auto')

nlabels=5

model = Sequential()
model.add(layers.Embedding(len(word_index)+1,
                           300,
                           weights=[embedding_matrix],
                           input_length=max_len,
                           trainable=False))
model.add(layers.SpatialDropout1D(0.3))
model.add(layers.GRU(300, dropout=0.3, recurrent_dropout=0.3, return_sequences=True))
model.add(layers.GRU(300, dropout=0.3, recurrent_dropout=0.3))

model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.8))

model.add(layers.Dense(nlabels, activation='sigmoid')) #'sigmoid'
model.compile(loss='categorical_crossentropy',
                optimizer='adam', #'adam',# optimizer.SGD(lr=1e-3)
                metrics=['acc'])
model.summary()

batch=512
epoch=30

history = model.fit(xtrain, ytrain,
                    validation_split=0.2,
                    epochs=epoch,
                    batch_size=batch,
                   verbose=1,
                   callbacks=[earlystop])
score, acc = model.evaluate(xtest, ytest,
                            batch_size=batch)
model.save('gru_model.h5')
#model.save_weights('gru_weights.h5')

NN_preds = model.predict_proba(xtest,batch_size=batch)
print(confusion_matrix(np.argmax(ytest, axis=1)+1, np.argmax(NN_preds, axis=1)+1))
print(classification_report(np.argmax(ytest, axis=1)+1, np.argmax(NN_preds, axis=1)+1))
print('Log Loss: {:.2f}'.format(log_loss(ytest, NN_preds)))
plot_history(history)

>>
[[ 6297     5   207    96   875]
 [ 1418     8   229   126   351]
 [  919     7   400   464   684]
 [  375     3   232   849  3208]
 [  640     0    89   461 17952]]
              precision    recall  f1-score   support

           1       0.65      0.84      0.74      7480
           2       0.35      0.00      0.01      2132
           3       0.35      0.16      0.22      2474
           4       0.43      0.18      0.25      4667
           5       0.78      0.94      0.85     19142

    accuracy                           0.71     35895
   macro avg       0.51      0.43      0.41     35895
weighted avg       0.65      0.71      0.66     35895

Log Loss: 0.78
```

![image](https://user-images.githubusercontent.com/23128332/65022977-9dc86f80-d93a-11e9-8702-9a801b37cad6.png)


