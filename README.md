# Cell-Phone-Reviews-Amazon

I decided to do a NLP project for three main reasons:
```
1-It is fun
2-I decided not to use a ready made data from the internet
3-I wanted to gather my own data and improve my data mining knowledge.
```
I managed to gather almost 200k samples, and after preprocessing I was left with 180k samples in total. Since the amount of data I have was not enough for training an algorithm to successfully predict amongst 5 review stars I decided to create 'Positive Review' class by merging 5-4 starred comments, and a 'Negative Review' class by merging 1-2 starred comments. I didn't want to create a 'Neutral Review' class because 13296 sample size was not enough to train an algorithm to successfully predict, and It would further decrease the performance by increasing the complexity. Sample size numbers for review stars are shown below:
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

def model_eval(model, k=5, seed=0):
    kfold = StratifiedKFold(k, random_state=seed)
    
    oof = np.zeros(len(y))
    for nfold, (train_ix, valid_ix) in enumerate(kfold.split(X,y)):
        trainX, validX = X[train_ix], X[valid_ix]
        trainy, validy = y[train_ix], y[valid_ix]

        model.fit(trainX, trainy)
        p = model.predict(validX)
        oof[valid_ix] = p
        
        print('Fold{}, Valid AUC: {:.4f}'.format(nfold,roc_auc_score(validy, p)))
        
    print(confusion_matrix(y, oof))
    print(classification_report(y, oof))
    print('Valid RMSLE: {:.3f}'.format(np.sqrt(mean_squared_log_error(y, oof))))
    print("F1_score : {:.2%} ".format(f1_score(y, oof, average='micro')))  # 'samples', 'weighted', 'macro', 'micro', 'binary'
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

# If the 'Title' column contain NaNs, replace them.
if df['Title'].isna().sum()==0:
    print("NaNs in Title: ",df['Title'].isna().sum())
    df['Title'] = df['Title'].apply(lambda x: "" if isinstance(x,float) else x)


# Merge Comments and Titles
df['Text'] = df['Text']+' '+df['Title']

# Leave only two columns
df = df[['Text','Stars']]

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

## Target Distribution

```
starsfq = df['Stars'].value_counts()
df['Stars'].value_counts()
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
trace = go.Bar(
    x=starsfq.index,
    y=starsfq.values,
    marker=dict(
        color=starsfq.values
        ),
    )

layout = go.Layout(
    title='Star Counts',
    )

data = [trace]
fig = go.Figure(
    data=data,
    layout=layout
    )
py.iplot(fig, filename='StarCounts')
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

## Target Feature: Satisfaction 
Since the data I mined doesn't contain enough sample to train algorithms to predict 2,3,4 starred comments, I decided to seperate the data into two segments, positive and negative comments, by naming 5 and 4 starred comments "1" and 1 and 2 starred comments "0".
```
df['Stars'] = df['Stars'].astype(float)
# Negative, Positive, Neutr Comments
df['Satisfaction'] = df['Stars'].apply(lambda x: 0 if x<=2 else (1 if x>=4 else 2))
df = df[df['Satisfaction']!=2]
len(df[df['Satisfaction']==0]),len(df[df['Satisfaction']==1])
df.reset_index(drop=True, inplace=True)
```

## Vectorization
### CountVectorizer
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
Fold0, Valid AUC: 0.9176
Fold1, Valid AUC: 0.8852
Fold2, Valid AUC: 0.8683
Fold3, Valid AUC: 0.8976
Fold4, Valid AUC: 0.8917
[[ 42680   8992]
 [  5226 119792]]
              precision    recall  f1-score   support

         0.0       0.89      0.83      0.86     51672
         1.0       0.93      0.96      0.94    125018

   micro avg       0.92      0.92      0.92    176690
   macro avg       0.91      0.89      0.90    176690
weighted avg       0.92      0.92      0.92    176690

Valid RMSLE: 0.197
F1_score : 91.95% 
AUC: 0.8921
```

```
plot_learning_curve(logr,'Logistic Regression', X=X, y=y, n_jobs=4)
```
Output:

![logrLC](https://user-images.githubusercontent.com/23128332/58369824-b5519780-7f08-11e9-8884-6ab247d0eeee.png)

## Stochastic Gradient Descent Classifier
```
model = SGDClassifier(loss='modified_huber')
sgd, sgd_preds = model_eval(model)
```
Output:
```
Fold0, Valid AUC: 0.9173
Fold1, Valid AUC: 0.8821
Fold2, Valid AUC: 0.8668
Fold3, Valid AUC: 0.8969
Fold4, Valid AUC: 0.8891
[[ 42512   9160]
 [  5231 119787]]
              precision    recall  f1-score   support

         0.0       0.89      0.82      0.86     51672
         1.0       0.93      0.96      0.94    125018

   micro avg       0.92      0.92      0.92    176690
   macro avg       0.91      0.89      0.90    176690
weighted avg       0.92      0.92      0.92    176690

Valid RMSLE: 0.198
F1_score : 91.86% 
AUC: 0.8904
```
Output:

![sgdfeaturew](https://user-images.githubusercontent.com/23128332/58369814-97843280-7f08-11e9-8ad8-94783c986363.JPG)

```
plot_learning_curve(sgd,'Logistic Regression', X=X, y=y, n_jobs=4)
```
Output:

![SGDLR](https://user-images.githubusercontent.com/23128332/58369823-b5519780-7f08-11e9-9eb1-0fa0a2a2d34c.png)

## RandomForestClassifier
```
tree = RandomForestClassifier(verbose=1, n_jobs=-1)
tree , tree_preds = model_eval(tree,k=2)
```
```
Fold1, Valid AUC: 0.8536
[[ 38910  12762]
 [  7222 117796]]
              precision    recall  f1-score   support

         0.0       0.84      0.75      0.80     51672
         1.0       0.90      0.94      0.92    125018

   micro avg       0.89      0.89      0.89    176690
   macro avg       0.87      0.85      0.86    176690
weighted avg       0.89      0.89      0.88    176690

Valid RMSLE: 0.233
F1_score : 88.69% 
AUC: 0.8476
Wall time: 19min 14s
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

![RF_FI](https://user-images.githubusercontent.com/23128332/58369287-ea59ec00-7f00-11e9-80a3-7d19effdd328.png)

## Lightgbm
```
import lightgbm as lgb
from lightgbm import LGBMClassifier

model = LGBMClassifier()
lgb_model, lgb_preds = model_eval(model)
```
Output:
```
Fold0, Valid AUC: 0.8959
Fold1, Valid AUC: 0.8629
Fold2, Valid AUC: 0.8474
Fold3, Valid AUC: 0.8698
Fold4, Valid AUC: 0.8636
[[ 40526  11146]
 [  6054 118964]]
              precision    recall  f1-score   support

         0.0       0.87      0.78      0.82     51672
         1.0       0.91      0.95      0.93    125018

   micro avg       0.90      0.90      0.90    176690
   macro avg       0.89      0.87      0.88    176690
weighted avg       0.90      0.90      0.90    176690

Valid RMSLE: 0.216
F1_score : 90.27% 
AUC: 0.8679
```
```
eli5.show_weights(lgb_model, vec=tfidf, top=50)
```
Output:

![lgweights](https://user-images.githubusercontent.com/23128332/58370208-7d4c5380-7f0c-11e9-8193-3d90984094d3.JPG)

## Blending
```
# Simple Averaging
blend = (logr_preds + sgd_preds + tree_preds + lgb_preds)/4
print('Valid RMSLE: {:.4f}'.format(np.sqrt(mean_squared_log_error(y, blend))))
print('AUC: {:.4f}'.format(roc_auc_score(y, blend)))
```
Output:
```
Valid RMSLE: 0.1860
AUC: 0.9222
```

### Voting Classifier
```
from sklearn.ensemble import VotingClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

blender = VotingClassifier(estimators=[('logr',logr),
                                     ('SGD',sgd),
                                     ('tree',tree),
                                     ('lgb',lgb_model)],
                         voting='hard'                         
                        )
blender, blend_preds = model_eval(blender)
```
Simple averaging gives much better performance than voting classifier. Simple blend predictions will be used in the Stacking stage.
Output:
```
Fold0, Valid AUC: 0.9202
Fold1, Valid AUC: 0.8893
Fold2, Valid AUC: 0.8712
Fold3, Valid AUC: 0.9018
Fold4, Valid AUC: 0.8949
[[ 43341   8331]
 [  5970 119048]]
              precision    recall  f1-score   support

         0.0       0.88      0.84      0.86     51672
         1.0       0.93      0.95      0.94    125018

   micro avg       0.92      0.92      0.92    176690
   macro avg       0.91      0.90      0.90    176690
weighted avg       0.92      0.92      0.92    176690

Valid RMSLE: 0.197
F1_score : 91.91% 
AUC: 0.8955
```

## Stacking
```
stacked_df = pd.DataFrame({
    'logr_preds':logr_preds,
    'sgd_preds':sgd_preds,
    'tree_preds':tree_preds,
    'lgb_preds':lgb_preds,
    'blend_preds':blend
})
```
### Stacking with XGBoost Classifier
```
from xgboost import XGBClassifier

s_train, s_test, y_train, y_test = train_test_split(stacked_df, y, test_size=0.3, random_state=1)

xgb = XGBClassifier()
xgb.fit(s_train,y_train)
stack_pred = xgb.predict(s_test)

print(confusion_matrix(stack_pred, y_test))
print(classification_report(stack_pred, y_test))
print('Valid RMSLE: {:.3f}'.format(np.sqrt(mean_squared_log_error(stack_pred, y_test))))
print("F1_score : {:.2%} ".format(f1_score(stack_pred, y_test)))
print('ROC AUC: {:.4f}'.format(roc_auc_score(stack_pred, y_test))
```
Output:
```
[[12808  1582]
 [ 2741 35876]]
              precision    recall  f1-score   support

         0.0       0.82      0.89      0.86     14390
         1.0       0.96      0.93      0.94     38617

   micro avg       0.92      0.92      0.92     53007
   macro avg       0.89      0.91      0.90     53007
weighted avg       0.92      0.92      0.92     53007

Valid RMSLE: 0.198
F1_score : 94.32% 
ROC AUC: 0.9095
```
Output:

![stackweights](https://user-images.githubusercontent.com/23128332/58369815-98b55f80-7f08-11e9-8b65-d4eac6cb3f26.JPG)


## Things to do:
```
1- Make research for better NLP techniques
2- Feature engineering
3- Model optimizations via gridsearch
4- Apply neural networks
5- More models for stacking
```
