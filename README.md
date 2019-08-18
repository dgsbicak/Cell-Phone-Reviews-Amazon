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
os.chdir(r"C:\Users\dogus\Dropbox\DgsPy_DBOX\Amazon Project")

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import matplotlib.pyplot as plt
%matplotlib inline

from nltk.corpus import stopwords
from contextlib import contextmanager
import eli5

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import ShuffleSplit, learning_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import f1_score, log_loss,roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_log_error

from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
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
    import re
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

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.subplots(figsize=(12,6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
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

def model_eval(model, k=5, seed=0):
    kfold = KFold(k, random_state=seed)
    oof = np.zeros(y.shape[0])
    for nfold, (train_ix, valid_ix) in enumerate(kfold.split(X,y)):
        trainX, validX = X[train_ix], X[valid_ix]
        trainy, validy = y[train_ix], y[valid_ix]
        
        model.fit(trainX, trainy)
        p = model.predict(validX)
        oof[valid_ix] = p
        if len(y.unique())==2:
            print('Fold{}, Valid AUC: {:.4f}'.format(nfold,roc_auc_score(validy, p)))
        else:
            print('Fold{}, F1_score : {:.2%}'.format(nfold,f1_score(validy, p, average='micro')))
        
    print(confusion_matrix(y, oof))
    print(classification_report(y, oof))
    print('Valid RMSLE: {:.3f}'.format(np.sqrt(mean_squared_log_error(y, oof))))
    print("F1_score : {:.2%} ".format(f1_score(y, oof, average='micro')))  # 'samples', 'weighted', 'macro', 'micro', 'binary'
    
    if len(y.unique())==2:
        print('AUC: {:.4f}'.format(roc_auc_score(y, oof)))

    return model, oof
```

## Read the merged data
```
data = pd.read_csv('AMAZON_comments_yuge.csv')
print(data.info())
data = data.drop_duplicates(subset='Text', keep=False)
data.reset_index(inplace=True,drop=True)
df = data.copy()
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

### A sample from the data
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

## Gender Ratio
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

## Preprocessing
```
# Clean some of the repetitive words.
df['Text'] = df['Text'].apply(clean_int2)  # Clean any integer value
df['Text'] = df['Text'].apply(lambda x: x.split('\r\r\n'))  # Seperate text into parts
df['Name'] = df['Text'].apply(lambda x: x[0])  # Commentator Names
df['Title'] = df['Text'].apply(lambda x: x[1]) # Review Titles

# Get the actual customer review by discarding unnecessary parts.
df['Text'] = df['Text'].apply(sorttext) 
```

```
# Titles contain the summary information of customer reviews (e.g. 'Five Stars')
# Titles should be added into Text column but they shouldn't contain Target info.Otherwise, our algorithm wouldn't perform well.
df['Title'] = df['Title'].apply(lambda x: "" if ('Star' in x.split())|('Stars' in x.split()) else x)
```

```
# Gather all of the phone model names from the data
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

# Merge the two models set, because they contain some duplicate names.
models = list(models_1.union(models_2))
brands = list(map(lambda x: x.lower().strip(),set(brands)))
```

```
t0 = time.time()
print("Preprocessing Comments..","\n")

# Merge Comments and Titles
df['Text'] = df['Text']+' '+df['Title']
df = df[['Text','Stars','Phone Title']]

# Drop NANs
if df.isna().sum().sum()!=0:
    print("NANs in DF:\n",df.isna().sum())
    df.dropna(axis=0,inplace=True)

# Delete Unnecessary Text
df['Text'] = df['Text'].apply(lambda x: x.replace('Verified Purchase',""))

# Make the text lowercase for the simplicity.
df['Text'] = df['Text'].apply(lambda x: x.lower())

print(df['Text'][0],"\n")
print('Deleting model informations...')
df['Text'] = df['Text'].apply(lambda text: ' '.join([word for word in text.split() if word not in models]))

print('Deleting brand names...')
df['Text'] = df['Text'].apply(lambda text: ' '.join([word for word in text.split() if word not in brands]))

print("Adjusting characters, skipping symbols, trimming escape characters...\n")
df['Text'] = df['Text'].apply(clean_str)
print(df['Text'][0],"\n")

print('Deleting stopwords')
df['Text'] = df['Text'].apply(lambda text: ' '.join([word for word in text.split() if word not in set(stopwords.words('english'))]))

print('Deleting punctuations')
df['Text'] = df['Text'].apply(lambda ftext: ''.join([e for e in ftext if e not in set(string.punctuation)]))

# Delete Unnecessary Text
df['Text'] = df['Text'].apply(lambda x: x.replace('amazon',""))

print("Completed within %0.1f minutes." % ((time.time() - t0)/60)) # 13.7
print(df['Text'][0],"\n")
```
Output:
```
Really really good! It's like a new phone, no one scratches, I'm really happy with it, just the charger is not original but I don't care, it's good enough Nice! 


Deleting model informations...
Deleting brand names...
Adjusting characters, skipping symbols, trimming escape characters...


really really good it s like a phone scratches i m really happy it just the charger is not but i don t care it s good enough nice 

Deleting stopwords...
Deleting punctuations...
Completed within 30.8 minutes.

really really good like phone scratches really happy charger care good enough nice 
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
mean        142.354912
std         316.887600
min           0.000000
25%          23.000000
50%          66.000000
75%         144.000000
max       16055.000000
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

![newplot (2)](https://user-images.githubusercontent.com/23128332/58369285-ea59ec00-7f00-11e9-8f3f-6a9808905e13.png)

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

![newplot (3)](https://user-images.githubusercontent.com/23128332/58369286-ea59ec00-7f00-11e9-8a84-60b4ccedca22.png)

# Algorithm Training

## Vectorization
### TfidfVectorizer
```
text = df['Text']
tfidf = TfidfVectorizer(token_pattern=r'\w{2,}',
                        ngram_range=(1, 3),
                        max_df=0.5,
                        #encoding='utf-8',
                        #decode_error='ignore',
                        #strip_accents='unicode'
                       )

tfidf.fit(text)
trans_tfidf = tfidf.transform(text)
#trans_tfidf = tfidf_fitted.transform(feature_test)

from sklearn.model_selection import train_test_split
X = trans_tfidf
y = df['Satisfaction'].astype(float)

feature_train, feature_test, label_train, label_test = train_test_split(X, y, test_size=0.3, shuffle=True)
```

# Model Building
## Logistic Regression
```
logr = LogisticRegression()
logr, logr_preds = model_eval(logr)
```
Output:
```
Fold0, F1_score : 73.80%
Fold1, F1_score : 69.89%
Fold2, F1_score : 67.15%
Fold3, F1_score : 70.31%
Fold4, F1_score : 68.89%
[[32844  1790  1477   588  3129]
 [ 6381  1179  1586   614  1600]
 [ 3645   982  2826  2026  3675]
 [ 1690   375  2162  4954 15558]
 [ 2610   318  1173  4945 89678]]
              precision    recall  f1-score   support

           1       0.70      0.82      0.76     39828
           2       0.25      0.10      0.15     11360
           3       0.31      0.21      0.25     13154
           4       0.38      0.20      0.26     24739
           5       0.79      0.91      0.84     98724

    accuracy                           0.70    187805
   macro avg       0.48      0.45      0.45    187805
weighted avg       0.65      0.70      0.67    187805

Valid RMSLE: 0.289
F1_score : 70.01% 
```

```
plot_learning_curve(logr,'Logistic Regression', X=X, y=y, n_jobs=4)
```
Output:

![LogR](https://user-images.githubusercontent.com/23128332/63229717-1d640100-c20c-11e9-9f62-1f322d8e4031.JPG)
```
eli5.show_weights(logr, vec=tfidf, top=40)
```
Output:

![logwords](https://user-images.githubusercontent.com/23128332/63229718-1d640100-c20c-11e9-8957-80b329a74117.JPG)

## Stochastic Gradient Descent Classifier
```
model = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, max_iter=5, tol=None)
sgd, sgd_preds = model_eval(model)
```
Output:
```
Fold0, F1_score : 61.67%
Fold1, F1_score : 60.03%
Fold2, F1_score : 55.43%
Fold3, F1_score : 62.26%
Fold4, F1_score : 61.76%
[[14683    16     6     5 25118]
 [ 2239    11     8     6  9096]
 [ 1136     7    26    15 11970]
 [  323     6    19     8 24383]
 [  305     5    13    14 98387]]
              precision    recall  f1-score   support

           1       0.79      0.37      0.50     39828
           2       0.24      0.00      0.00     11360
           3       0.36      0.00      0.00     13154
           4       0.17      0.00      0.00     24739
           5       0.58      1.00      0.74     98724

    accuracy                           0.60    187805
   macro avg       0.43      0.27      0.25    187805
weighted avg       0.53      0.60      0.49    187805

Valid RMSLE: 0.456
F1_score : 60.23% 
```

```
plot_learning_curve(sgd,'Logistic Regression', X=X, y=y, n_jobs=4)
```
Output:
![SGD](https://user-images.githubusercontent.com/23128332/63229719-1d640100-c20c-11e9-8937-5c790e01eb15.JPG)

```
eli5.show_weights(sgd, vec=tfidf, top=40)
```
Output:
![sgdWords](https://user-images.githubusercontent.com/23128332/63229716-1d640100-c20c-11e9-90f4-1fedcf9c9710.JPG)


## Support Vector Machine Classifier
```
from sklearn.svm import LinearSVC
svc = LinearSVC()
svc, svc_preds = model_eval(svc)
```
Output:
```
Fold0, F1_score : 74.92%
Fold1, F1_score : 70.66%
Fold2, F1_score : 68.32%
Fold3, F1_score : 71.24%
Fold4, F1_score : 70.38%
[[34423   522   481   501  3901]
 [ 7557   469   545   557  2232]
 [ 4804   396  1147  1796  5011]
 [ 2063   155   679  3419 18423]
 [ 2193    92   284  2074 94081]]
              precision    recall  f1-score   support

           1       0.67      0.86      0.76     39828
           2       0.29      0.04      0.07     11360
           3       0.37      0.09      0.14     13154
           4       0.41      0.14      0.21     24739
           5       0.76      0.95      0.85     98724

    accuracy                           0.71    187805
   macro avg       0.50      0.42      0.40    187805
weighted avg       0.64      0.71      0.65    187805

Valid RMSLE: 0.296
F1_score : 71.11% 
```

# MultinomialNB()
```
from sklearn.naive_bayes import MultinomialNB
MNB = MultinomialNB()
MNB, MNB_preds = model_eval(MNB)
```
Output:
```
Fold0, F1_score : 64.68%
Fold1, F1_score : 62.46%
Fold2, F1_score : 59.67%
Fold3, F1_score : 64.65%
Fold4, F1_score : 63.92%
[[19969     0     0     0 19859]
 [ 3022     0     0     0  8338]
 [ 1262     0     0     0 11892]
 [  278     0     0     1 24460]
 [  234     0     0     0 98490]]
              precision    recall  f1-score   support

           1       0.81      0.50      0.62     39828
           2       0.00      0.00      0.00     11360
           3       0.00      0.00      0.00     13154
           4       1.00      0.00      0.00     24739
           5       0.60      1.00      0.75     98724

    accuracy                           0.63    187805
   macro avg       0.48      0.30      0.27    187805
weighted avg       0.62      0.63      0.53    187805

Valid RMSLE: 0.415
F1_score : 63.08% 
```


## RandomForestClassifier
```
tree = RandomForestClassifier(verbose=1, n_jobs=-1)
tree , tree_preds = model_eval(tree,k=2)
```
Output:
```
Fold0, F1_score : 66.86%
Fold1, F1_score : 66.16%
[[30886   588   587   433  7334]
 [ 6966   282   382   271  3459]
 [ 5177   268   575   651  6483]
 [ 3298   155   451  1137 19698]
 [ 4503   184   366  1636 92035]]
              precision    recall  f1-score   support

           1       0.61      0.78      0.68     39828
           2       0.19      0.02      0.04     11360
           3       0.24      0.04      0.07     13154
           4       0.28      0.05      0.08     24739
           5       0.71      0.93      0.81     98724

    accuracy                           0.67    187805
   macro avg       0.41      0.36      0.34    187805
weighted avg       0.57      0.67      0.59    187805

Valid RMSLE: 0.367
F1_score : 66.51% 
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
![Forest](https://user-images.githubusercontent.com/23128332/63217651-d7973200-c152-11e9-8bcf-2d8cebd924fe.JPG)

## Lightgbm
```
import lightgbm as lgb
from lightgbm import LGBMClassifier

model = LGBMClassifier()
lgb_model, lgb_preds = model_eval(model)
```
Output:
```
Fold0, F1_score : 73.40%
Fold1, F1_score : 69.64%
Fold2, F1_score : 66.97%
Fold3, F1_score : 69.97%
Fold4, F1_score : 69.09%
[[32049   354   518   673  6234]
 [ 6715   384   633   680  2948]
 [ 4184   284  1361  1790  5535]
 [ 1999   133   673  3569 18365]
 [ 2709    79   278  1908 93750]]
              precision    recall  f1-score   support

           1       0.67      0.80      0.73     39828
           2       0.31      0.03      0.06     11360
           3       0.39      0.10      0.16     13154
           4       0.41      0.14      0.21     24739
           5       0.74      0.95      0.83     98724

    accuracy                           0.70    187805
   macro avg       0.51      0.41      0.40    187805
weighted avg       0.63      0.70      0.64    187805

Valid RMSLE: 0.326
F1_score : 69.81% 
```
```
eli5.show_weights(lgb_model, vec=tfidf, top=50)
```
Output:

![lgweights](https://user-images.githubusercontent.com/23128332/58370208-7d4c5380-7f0c-11e9-8193-3d90984094d3.JPG)

# Neural Networks
```
from keras import layers
from keras.models import Sequential
from keras.optimizers import Adam
import tensorflow as tf

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
gpu_name = tf.test.gpu_device_name()
session_conf = tf.ConfigProto(intra_op_parallelism_threads=44,
                             inter_op_parallelism_threads=44,
                             allow_soft_placement=True,
                             gpu_options=gpu_options)
sess = tf.Session(graph=tf.get_default_graph(),
                 config=session_conf)
session_conf.gpu_options.allow_growth=True

def build_model():
    embed_dim = 50
    nlabels=5
    nlayer = 1
    hidden_u = 128
    attention_u = 64

    model = Sequential()
    model.add(layers.embeddings.Embedding(xtrain.shape[1], embed_dim))
    model.add(layers.GRU(units=hidden_u, dropout=0.5, recurrent_dropout=0.2))
    model.add(layers.Dense(attention_u, activation = 'tanh'))
    """
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(1, activation = None))
    model.add(layers.Dense(attention_u, activation = 'softmax'))
    """
    model.add(layers.Dense(nlabels, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy',
                 optimizer='adam',
                 metrics=['categorical_accuracy'])
    return model

batch=512
epoch=5
k = 5
kfold = KFold(k, random_state=0)

tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(X)

ohe = OneHotEncoder()
ohe.fit(y.values.reshape(-1,1))

NN_preds = np.zeros((y.shape[0], len(y.unique())))
for nfold, (train_ix, test_ix) in enumerate(kfold.split(X,y)):
    print(f"\nFold {nfold+1}/{k}, ... ")
    xtrain, xtest = X[train_ix], X[test_ix]
    ytrain, ytest = y[train_ix], y[test_ix]
    
    xtrain = tokenizer.texts_to_sequences(xtrain)
    xtest  = tokenizer.texts_to_sequences(xtest)
    xtrain = sequence.pad_sequences(xtrain)
    xtest = sequence.pad_sequences(xtest)
    
    ytrain = ohe.transform(ytrain.values.reshape(-1,1))
    ytest = ohe.transform(ytest.values.reshape(-1,1))
    
    model = build_model()
    history = model.fit(xtrain, ytrain,
                        validation_split=0.1,
                       epochs=epoch,
                       batch_size=batch)
    score, acc = model.evaluate(xtest, ytest,
                               batch_size=batch)
    print(f"{nfold+1}th fold, Score: %.2f, Accuracy: %.2f" % (score,acc))
    NN_preds[test_ix] = model.predict(xtest)

ytest = ohe.transform(y.values.reshape(-1,1)).toarray()
print('Valid RMSLE: {:.3f}'.format(np.sqrt(mean_squared_log_error(ytest, NN_preds))))
print("F1_score : {:.2%} ".format(f1_score(np.argmax(ytest, axis=1)+1, np.argmax(NN_preds, axis=1)+1, average='micro')))
print('ROC AUC: {:.2f}'.format(roc_auc_score(ytest, NN_preds)))
print("Accuracy: {:.2f}".format(accuracy_score(np.argmax(ytest, axis=1)+1, np.argmax(NN_preds, axis=1)+1)))
print(confusion_matrix(np.argmax(ytest, axis=1)+1, np.argmax(NN_preds, axis=1)+1))
print(classification_report(np.argmax(ytest, axis=1)+1, np.argmax(NN_preds, axis=1)+1))
```
Output:
```
[[29159     0    76   353 10240]
 [ 6675     1   100   329  4255]
 [ 5008     2   130   679  7335]
 [ 3162     0   107  1024 20446]
 [ 5956     0    45   723 92000]]
              precision    recall  f1-score   support

           1       0.58      0.73      0.65     39828
           2       0.33      0.00      0.00     11360
           3       0.28      0.01      0.02     13154
           4       0.33      0.04      0.07     24739
           5       0.69      0.93      0.79     98724

    accuracy                           0.65    187805
   macro avg       0.44      0.34      0.31    187805
weighted avg       0.57      0.65      0.56    187805

Valid RMSLE: 0.238
F1_score : 65.13% 
ROC AUC: 0.78
Accuracy: 0.65
```

## Blending
```
# Simple Averaging
blend = (svc_preds + sgd_preds + MNB_preds + tree_preds + lgb_preds + (np.argmax(NN_preds,axis=1)+1))/6
blend = pd.Series(map(round,blend))
print('Valid RMSLE: {:.2f}'.format(np.sqrt(mean_squared_log_error(y, blend))))
print('F1_score : {:.2%}'.format(f1_score(y, blend, average='micro')))

hoter = lambda array: ohe.transform(array.values.reshape(-1,1)).toarray()
print('AUC: {:.2f}'.format(roc_auc_score(hoter(y), hoter(blend))))
```
Output:
```
Valid RMSLE: 0.30
F1_score : 58.95%
AUC: 0.63
```

### Voting Classifier
```
from sklearn.ensemble import VotingClassifier

voter = VotingClassifier(estimators=[('SGD',sgd),
                                     ('SVC',svc),
                                     ('MNB',MNB),
                                     ('tree',tree),
                                     ('lgb',lgb_model),
                                    ('GRU',model)],
                            voting='hard')

voter, voter_preds = model_eval(voter)
```

Output:
```

```

## Stacking
```
stacked_preds = pd.DataFrame({
    'logr_preds':logr_preds,
    'sgd_preds':sgd_preds,
    'svc_preds':svc_preds,
    'MNB_preds':MNB_preds,
    'tree_preds':tree_preds,
    'lgb_preds':lgb_preds,
    'blend_preds':blend,
    'NN_preds':np.argmax(NN_preds, axis=1)+1
})

stacked_preds.to_csv('stacked_preds.csv',index=False)
```
### Stacking with XGBoost Classifier
```
from xgboost import XGBClassifier
s_train, s_test, y_train, y_test = train_test_split(stacked_preds, y, test_size=0.2, random_state=10)

xgb = XGBClassifier(n_jobs=-1)
xgb.fit(s_train,y_train)
stack_pred = xgb.predict(s_test)

print(confusion_matrix(stack_pred, y_test))
print(classification_report(stack_pred, y_test))
print('Valid RMSLE: {:.3f}'.format(np.sqrt(mean_squared_log_error(stack_pred, y_test))))
print("F1_score : {:.2%} ".format(f1_score(y_test, stack_pred, average='micro')))
print('ROC AUC: {:.2f}'.format(roc_auc_score(onehotter(np.array(y_test)),onehotter(np.array(stack_pred)))))
```
Output:
```
[[ 7027  1550   964   431   475]
 [   24    44    15     6     1]
 [  188   233   434   244    87]
 [   87   103   376   803   491]
 [  669   359   858  3472 18620]]
              precision    recall  f1-score   support

           1       0.88      0.67      0.76     10447
           2       0.02      0.49      0.04        90
           3       0.16      0.37      0.23      1186
           4       0.16      0.43      0.24      1860
           5       0.95      0.78      0.85     23978

    accuracy                           0.72     37561
   macro avg       0.43      0.55      0.42     37561
weighted avg       0.86      0.72      0.78     37561

Valid RMSLE: 0.290
F1_score : 71.69% 
ROC AUC: 0.67
```

Output:
![stacked](https://user-images.githubusercontent.com/23128332/63217656-d82fc880-c152-11e9-8081-50a643e768cb.JPG)


## Things to do:
```
1- Gather more data.
2- Make research for better NLP techniques.
3- Feature engineering
4- Model optimizations
5- More models for stacking

- Sentimental Analysis
```
