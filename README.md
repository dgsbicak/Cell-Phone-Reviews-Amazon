# Cell-Phone-Reviews-Amazon
```
import re
import csv
import pandas as pd
import numpy as np
import pickle
import time
import string
from nltk.corpus import stopwords
import os
os.chdir(r"...\Amazon Project")

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import matplotlib.pyplot as plt
%matplotlib inline

```
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
    return output  # 'hello world'

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

data = pd.read_csv('AMAZON_comments_yuge.csv')
print(data.info())
data = data.drop_duplicates(subset='Text', keep=False)
data.reset_index(inplace=True,drop=True)
df = data.copy()
print(df.info())

```


```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 214993 entries, 0 to 214992
Data columns (total 3 columns):
Text           214993 non-null object
Phone Title    214993 non-null object
Stars          214993 non-null int64
dtypes: int64(1), object(2)
memory usage: 4.9+ MB
None
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

#### A sample from the data
```
df['Text'][85101]
```
output:
```
'Kenneth B.\r\r\nFive Stars\r\r\nJune 5, 2016\r\r\nStyle: U.S. Version (LGUS991)Verified Purchase\r\r\nNice phone\r\r\nHelpful\r\r\nComment Report abuse'
```

If we split the example from the parts where '\r\r\n' exists:
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

### Preprocessing
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
    mode='markers'
)
data = [trace]
py.iplot(data, filename='Text Length')
```

PLOTT

Some of the comments don't reflect the population; some of them are spams, some of them are overenthusiastic people, and so on. In our refined data, average length of reviews is 142, which means 142 characters found (including blank space characters) on average, in every review. I decided to leave out every comment that is longer than 500 characters.

```
df['textbool'] = df['textlength'].apply(lambda x: 1 if x<500 else 0)
df = df[df['textbool']==1]
```

# Target Distribution

```
starsfq = df['Stars'].value_counts()
df['Stars'].value_counts()
```

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

PLOTT

### Wordcloud
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
    background_color='black',
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
CLOUDIMG

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

WORDFQPLOT



# Vectorization
### CountVectorizer
```
text = df['Text']
count_vec = CountVectorizer(ngram_range=(1, 3),
                            #max_df=0.50,
                            analyzer='word',
                           token_pattern = r'\w{2,}'
                           )
feature_train_counts = count_vec.fit(text)
bag_of_words = feature_train_counts.transform(text)

X = bag_of_words
#y = df['Stars'].astype(float)
y = df['Stars']
```

# Model Building

## Logistic Regression
```
feature_train, feature_test, label_train, label_test = train_test_split(X, y, test_size=0.33, shuffle=True)
clf = LogisticRegression().fit(feature_train, label_train)
preds_lr = clf.predict(feature_test)
preds_proba = clf.predict_proba(feature_test)

print(confusion_matrix(label_test, preds_lr))
print(classification_report(label_test, preds_lr))
print('Valid RMSLE: {:.4f}'.format(np.sqrt(mean_squared_log_error(label_test, preds_lr))))
print("F1_score : {:.2%} ".format(f1_score(label_test, preds_lr, average='micro')))  # 'samples', 'weighted', 'macro', 'micro', 'binary'
print("Log_loss : {:.4f} ".format(log_loss(label_test, preds_proba)))
```

## Stochastic Gradient Descent Classifier
```
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import f1_score, log_loss

clf = SGDClassifier(loss='log').fit(feature_train, label_train)
preds_sgd = clf.predict(feature_test)
preds_proba = clf.predict_proba(feature_test)

print(confusion_matrix(label_test, preds_sgd))
print(classification_report(label_test, preds_sgd))
print('Valid RMSLE: {:.4f}'.format(np.sqrt(mean_squared_log_error(label_test, preds_sgd))))
print("F1_score : {:.2%} ".format(f1_score(label_test, preds_sgd, average='micro')))  # 'samples', 'weighted', 'macro', 'micro', 'binary'
print("Log_loss : {:.4f} ".format(log_loss(label_test, preds_proba)))

"""
Valid RMSLE: 0.2126
F1_score : 80.53% 
Log_loss : 1.3345 

Valid RMSLE: 0.2194
F1_score : 80.47% 
Log_loss : 0.7738 

Valid RMSLE: 0.2286
F1_score : 80.48% 
Log_loss : 0.7692  # trimmed a little
"""
```
```
[[3292  101   64   53  122]
 [ 452  520   96   57   90]
 [ 258  103  762  211  207]
 [  88   32  106 1703  730]
 [ 104   22   40  326 7113]]
             precision    recall  f1-score   support

        1.0       0.78      0.91      0.84      3632
        2.0       0.67      0.43      0.52      1215
        3.0       0.71      0.49      0.58      1541
        4.0       0.72      0.64      0.68      2659
        5.0       0.86      0.94      0.90      7605

avg / total       0.79      0.80      0.79     16652

Valid RMSLE: 0.2221
F1_score : 80.41% 
Log_loss : 0.7619 
```
![sgdc](https://user-images.githubusercontent.com/23128332/43723046-6626896c-999f-11e8-9dd5-4909a720a8d5.PNG)

## LogisticRegression
```
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(C=10).fit(feature_train, label_train)
preds_lr = clf.predict(feature_test)
preds_proba = clf.predict_proba(feature_test)

print(confusion_matrix(label_test, preds_lr))
print(classification_report(label_test, preds_lr))
print('Valid RMSLE: {:.4f}'.format(np.sqrt(mean_squared_log_error(label_test, preds_lr))))
print("F1_score : {:.2%} ".format(f1_score(label_test, preds_lr, average='micro')))  # 'samples', 'weighted', 'macro', 'micro', 'binary'
print("Log_loss : {:.4f} ".format(log_loss(label_test, preds_proba)))
"""
Valid RMSLE: 0.2063
F1_score : 81.20% 
Log_loss : 0.8746 

Valid RMSLE: 0.2207
F1_score : 80.30% 
Log_loss : 0.8388 

Valid RMSLE: 0.2270
F1_score : 80.54% 
Log_loss : 0.8218 # trimmed a little
"""
```
```
[[3271  107   64   56  134]
 [ 420  529   99   61  106]
 [ 258   87  786  185  225]
 [  83   37  107 1649  783]
 [ 104   17   46  260 7178]]
             precision    recall  f1-score   support

        1.0       0.79      0.90      0.84      3632
        2.0       0.68      0.44      0.53      1215
        3.0       0.71      0.51      0.59      1541
        4.0       0.75      0.62      0.68      2659
        5.0       0.85      0.94      0.90      7605

avg / total       0.80      0.81      0.79     16652

Valid RMSLE: 0.2245
F1_score : 80.55% 
Log_loss : 0.8122 
```
![logisticr](https://user-images.githubusercontent.com/23128332/43723039-627e5eac-999f-11e8-8538-aa6786856695.PNG)
