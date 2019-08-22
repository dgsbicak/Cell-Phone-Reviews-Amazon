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
        print('Fold{}, F1_score : {:.2%}'.format(nfold,f1_score(validy, p, average='micro')))
    
    print(confusion_matrix(y, oof))
    print(classification_report(y, oof))
    print('Valid RMSLE: {:.3f}'.format(np.sqrt(mean_squared_log_error(y, oof))))
    print("F1_score : {:.2%} ".format(f1_score(y, oof, average='micro')))  # 'samples', 'weighted', 'macro', 'micro', 'binary'
    return model, oof
    
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
really really good! it's like a new phone, no scratches, i'm really happy with it, just the charger is not original but i don't care, it's good enough nice! 

Deleting model informations
Deleting brand names...
Adjusting characters, trimming escape characters, symbols...

really really good it s like a phone scratches i m really happy it just the charger is not original but i don t care it s good enough nice 

Deleting punctuations...
really really good it s like a phone scratches i m really happy it just the charger is not original but i don t care it s good enough nice 
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

## Problem of Imbalanced Classes
### Upsampling Minority Classes
```
from sklearn.utils import resample

def upsampledf(df,target):
    """
    Divides classes into different dataframes,
    Appends upsampled minority class samples onto majority class samples
    """
    majority_class_name = df[target].value_counts().index[0]
    features = df[target].unique().tolist()
    majority_df = df[df['Stars']==majority_class_name]
    threshold = len(majority_df)
    for feature in features[1:]:
        print(f"Class '{feature}' is being upsampled. ")
        one_df = df[df[target]==feature]
        upsampledclass = resample(one_df,
                             replace=True, # Sample with replacement
                              n_samples=threshold,
                              random_state=10)
        majority_df = pd.concat([majority_df,upsampledclass],axis=0)
        majority_df = majority_df.sample(frac=1.0).reset_index(drop=True)
    print("Final class distribution:")
    print(majority_df.Stars.value_counts())
    return majority_df

df = pd.read_csv('cleaned_amazon_yuge.csv')
df = df.dropna(axis=0).reset_index(drop=True)
df = upsampledf(df, 'Stars')
X = df['Text']
y = df['Stars']
```

Output:
```
Feature '1' is being upsampled. 
Feature '3' is being upsampled. 
Feature '4' is being upsampled. 
Feature '2' is being upsampled. 
Final class distribution:
5    102783
4    102783
3    102783
2    102783
1    102783
```

## Vectorization
### TfidfVectorizer
```
tfidf = TfidfVectorizer(token_pattern=r'\w{1,}',
                        ngram_range=(1, 3),
                        max_df=0.5,
                        #encoding='utf-8',
                        #decode_error='ignore',
                        #strip_accents='unicode'
                       )

tfidf.fit(X)
X = tfidf.transform(X)
y = df['Stars']
```

# Model Building
## Logistic Regression
```
logr = LogisticRegression(solver='lbfgs',multi_class="multinomial",C=1e5)
logr, logr_preds = model_eval(logr)

```
Output:
---

```
plot_learning_curve(logr,'Logistic Regression', X=X, y=y, n_jobs=4)
```
Output:

---
```
eli5.show_weights(logr, vec=tfidf, top=40)
```
Output:

---

## Stochastic Gradient Descent Classifier
```
sgd = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-10)
sgd, sgd_preds = model_eval(sgd)
# Save Model 
pickle.dump(sgd ,open("SGD_model.sav", 'wb'))
```
Output:
```
Fold0, F1_score : 92.23%
Fold1, F1_score : 92.83%
Fold2, F1_score : 92.18%
Fold3, F1_score : 91.71%
Fold4, F1_score : 92.45%
[[ 97989   1884   1136    677   1097]
 [   626 101386    338    249    184]
 [   251    629  99168   1373   1362]
 [   234    652   1797  92712   7388]
 [  1595    798   2756  14651  82983]]
              precision    recall  f1-score   support

           1       0.97      0.95      0.96    102783
           2       0.96      0.99      0.97    102783
           3       0.94      0.96      0.95    102783
           4       0.85      0.90      0.87    102783
           5       0.89      0.81      0.85    102783

    accuracy                           0.92    513915
   macro avg       0.92      0.92      0.92    513915
weighted avg       0.92      0.92      0.92    513915

Valid RMSLE: 0.120
F1_score : 92.28%
```

```
plot_learning_curve(sgd,'SGD', X=X, y=y, scoring='neg_mean_squared_log_error', n_jobs=4)
```
Output:
![SGD](https://user-images.githubusercontent.com/23128332/63510611-ab2f3d00-c4e7-11e9-9043-2c24ab7f4372.JPG)

```
plot_learning_curve(sgd,'SGD', X=X, y=y, scoring='accuracy', n_jobs=4)
```
Output:
![SGDacc](https://user-images.githubusercontent.com/23128332/63510612-abc7d380-c4e7-11e9-8f4b-318e697d1f9f.JPG)

```
eli5.show_weights(sgd, vec=tfidf, top=40)
```
Output:
[sgdWords](https://user-images.githubusercontent.com/23128332/63510609-ab2f3d00-c4e7-11e9-8581-563d87548455.JPG)

## Linear Support Vector Classifier
```
from sklearn.svm import LinearSVC
svc = LinearSVC()
svc, svc_preds = model_eval(svc)
pickle.dump(svc ,open("linearSVC109.sav", 'wb'))  # Save the model
```
Output:
```
Fold0, F1_score : 93.70%
Fold1, F1_score : 93.66%
Fold2, F1_score : 93.52%
Fold3, F1_score : 93.54%
Fold4, F1_score : 93.58%
[[ 99378   1458    764    364    819]
 [   820 101308    338    225     92]
 [   348    286  99452   1803    894]
 [   314    183   1451  93074   7761]
 [  1703    386   1714  11170  87810]]
              precision    recall  f1-score   support

           1       0.97      0.97      0.97    102783
           2       0.98      0.99      0.98    102783
           3       0.96      0.97      0.96    102783
           4       0.87      0.91      0.89    102783
           5       0.90      0.85      0.88    102783

    accuracy                           0.94    513915
   macro avg       0.94      0.94      0.94    513915
weighted avg       0.94      0.94      0.94    513915

Valid RMSLE: 0.109
F1_score : 93.60% 
```

```
eli5.show_weights(svc, vec=tfidf, top=40)
```
Output:
![SVCwords](https://user-images.githubusercontent.com/23128332/63510979-a919ae00-c4e8-11e9-9b56-b932c82929d7.JPG)

# MultinomialNB()
```
from sklearn.naive_bayes import MultinomialNB
MNB = MultinomialNB()
MNB, MNB_preds = model_eval(MNB)
pickle.dump(MNB ,open("MultinomialNB.sav", 'wb'))
```
Output:
```
Fold0, F1_score : 83.20%
Fold1, F1_score : 83.33%
Fold2, F1_score : 82.88%
Fold3, F1_score : 83.10%
Fold4, F1_score : 83.29%
[[87558 10732  3420   775   298]
 [ 2554 98448  1093   488   200]
 [ 1955  1601 94962  2787  1478]
 [ 1245  2778  4467 84349  9944]
 [ 1897  3287  5178 30383 62038]]
              precision    recall  f1-score   support

           1       0.92      0.85      0.88    102783
           2       0.84      0.96      0.90    102783
           3       0.87      0.92      0.90    102783
           4       0.71      0.82      0.76    102783
           5       0.84      0.60      0.70    102783

    accuracy                           0.83    513915
   macro avg       0.84      0.83      0.83    513915
weighted avg       0.84      0.83      0.83    513915

Valid RMSLE: 0.169
F1_score : 83.16% 
```


## RandomForestClassifier
```
tree = RandomForestClassifier(verbose=1, n_jobs=-1)
tree , tree_preds = model_eval(tree,k=2)
pickle.dump(svc ,open("bigoltree.sav", 'wb'))
```
Output:
```
Fold0, F1_score : 89.21%
Fold1, F1_score : 89.19%
[[ 96271   1657   1225   1244   2386]
 [  1568 100168    405    364    278]
 [  1344    388  97485   2177   1389]
 [  2492    455   2165  86530  11141]
 [  7745    835   2390  13860  77953]]
              precision    recall  f1-score   support

           1       0.88      0.94      0.91    102783
           2       0.97      0.97      0.97    102783
           3       0.94      0.95      0.94    102783
           4       0.83      0.84      0.84    102783
           5       0.84      0.76      0.80    102783

    accuracy                           0.89    513915
   macro avg       0.89      0.89      0.89    513915
weighted avg       0.89      0.89      0.89    513915

Valid RMSLE: 0.195
F1_score : 89.20% 
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
![image](https://user-images.githubusercontent.com/23128332/63511166-1f1e1500-c4e9-11e9-9cd0-d67d9d561637.png)


## Lightgbm
```
import lightgbm as lgb
from lightgbm import LGBMClassifier

model = LGBMClassifier()
lgb_model, lgb_preds = model_eval(model)
```
Output:
```
```

```
eli5.show_weights(lgb_model, vec=tfidf, top=40)
```
Output:


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
    embed_dim = 10
    nlabels=5
    hidden_u = 1
    conv_u = 4
    attention_u = 10

    model = Sequential()
    model.add(layers.embeddings.Embedding(xtrain.shape[1], embed_dim, trainable=True))
    #model.add(layers.GRU(units=hidden_u))#, dropout=0.1, recurrent_dropout=0.1))
    model.add(layers.GlobalMaxPool1D())
    model.add(layers.Dense(attention_u, activation = 'relu'))
    """
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(1, activation = None))
    model.add(layers.Dense(attention_u, activation = 'softmax'))
    """
    model.add(layers.Dense(nlabels, activation='sigmoid')) #'sigmoid'
    model.compile(loss='categorical_crossentropy',
                 optimizer='adam',# optimizer.SGD(lr=1e-3)
                 metrics=['categorical_accuracy'])
    return model

xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size=0.2,random_state=10,shuffle=False)

from keras.preprocessing import text, sequence
tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(X)
del X,y

xtrain = tokenizer.texts_to_sequences(xtrain)
xtest  = tokenizer.texts_to_sequences(xtest)
xtrain = sequence.pad_sequences(xtrain)
xtest = sequence.pad_sequences(xtest)
    
ytrain = ohe.transform(ytrain.values.reshape(-1,1))
ytest = ohe.transform(ytest.values.reshape(-1,1))

batch=10000
epoch=40
model = build_model()
history = model.fit(xtrain, ytrain,
                    validation_split=0.2,
                    epochs=epoch,
                    batch_size=batch)
score, acc = model.evaluate(xtest, ytest,
                            batch_size=batch)

NN_preds = model.predict(xtest,batch_size=batch)
print(confusion_matrix(np.argmax(ytest, axis=1)+1, np.argmax(NN_preds, axis=1)+1))
print(classification_report(np.argmax(ytest, axis=1)+1, np.argmax(NN_preds, axis=1)+1))
print('Valid RMSLE: {:.3f}'.format(np.sqrt(mean_squared_log_error(ytest.toarray(), NN_preds))))
print("F1_score : {:.2%} ".format(f1_score(np.argmax(ytest, axis=1)+1, np.argmax(NN_preds, axis=1)+1, average='micro')))
print("Accuracy: {:.2f}".format(accuracy_score(np.argmax(ytest, axis=1)+1, np.argmax(NN_preds, axis=1)+1)))
plot_history(history)
model.save_weights('wo_gru_weights.h5')
```
Output:
```

```


## Blending
```
# Simple Averaging
blend = (svc_preds + sgd_preds + MNB_preds + tree_preds + lgb_preds + (np.argmax(NN_preds,axis=1)+1))/6
blend = pd.Series(map(round,blend))
print('Valid RMSLE: {:.2f}'.format(np.sqrt(mean_squared_log_error(y, blend))))
print('F1_score : {:.2%}'.format(f1_score(y, blend, average='micro')))

hoter = lambda array: ohe.transform(array.values.reshape(-1,1)).toarray()

```
Output:
```

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
```
Output:
```

```

Output:


## Things to do:
```
1- Gather more data.
2- Make research for better NLP techniques.
3- Feature engineering
4- Model optimizations
5- More models for stacking

- Sentimental Analysis
```
