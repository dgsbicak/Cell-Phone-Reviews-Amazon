# Cell-Phone-Reviews-Amazon
```
import pandas as pd
import os
import numpy as np
from sklearn.metrics import mean_squared_log_error
import string
from nltk.corpus import stopwords
import time
os.chdir(r"...\Amazon Project")
```
```
def combine_CSVs_infolder():
    os.chdir(r"C:\Users\dogus\Dropbox\DgsPy_DBOX\Amazon Project\comments_AMAZON")
    filenames = os.listdir()
    comb = pd.concat( [pd.read_csv(f) for f in filenames])
    comb.to_csv('AMAZON_comments1.csv', index=False)
```

```
data = pd.read_csv('AMAZON_comments1.csv',dtype=str)
df = data.copy()
df = df.dropna(axis=0)
text = df['Text']
df.info()
```

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 43533 entries, 0 to 43532
Data columns (total 3 columns):
Text           43533 non-null object
Phone Title    43530 non-null object
Stars          43513 non-null float64
dtypes: float64(1), object(2)
memory usage: 1020.4+ KB
```
```
data['Text'][0]
```
#### Review Example Before Process
output:

'You won\'t believe what you get for 50$ bucks!\r\r\nByLuisP8on June 18, 2016\r\r\nColor: Black\r\r\nVerified Purchase\r\r\nI must say and admit I was a little concerned about this choice after I bought it but I said "Well what the heck, let\'s give it a try!" It was for my mom anyway but on paper it looked like a nice budget phone, and also I had extra concerns cause I\'m not from usa so retuning this was out of the table. I had to accept whatever came on the box. Fortunately I was wrong.. The product came in perfect condition and brand new of course, no problems with packaging or shipping. And after first hands on the phone, you won\'t believe you pay 60 bucks (after taxes and handling) for this phone.. I MEAN IT! This guys (BLU Inc. ) are doing an awesome job with budget phones. This phone is really awesome. Of course it doesn\'t look fancy or premium because it\'s obviously plastic but you fell like you have a 500$ phone in your hands..\r\r\nSo if you are looking for a budget phone DO NOT hesitate of this one.. you won\'t be sorry!!\r\r\n410 people found this helpful\r\r\nHelpful\r\r\nNot Helpful\r\r\n5 comments\r\r\nReport abuse'


```
def stop_punch(text):
    # Clean out all the stopwords
    stext = [word for word in text.lower().split() if word not in set(stopwords.words('english'))]
    ftext = ' '.join(stext)
    
    # Clean out all the punctuations
    stext = [word for word in ftext if word not in set(string.punctuation)]
    ftext = ''.join(stext)
    return ftext
```

```
# Clean some of the repetitive words.
text = text.apply(lambda x: x.replace('Format',""))
text = text.apply(lambda x: x.replace('Wireless Phone Accessory',""))
text = text.apply(lambda x: x.replace('Color',""))
text = text.apply(lambda x: x.replace('Verified Purchase',""))
text = text.apply(lambda x: x.replace('people found this helpful',""))
```

```
t0 = time.time()
text = text.apply(stop_punch)
print("Completed within %0.1f seconds." % (time.time() - t0))
```
```
output:
Completed within 902.2 seconds.
```
```
text[0]
```
#### Review Example After Process
output:

'believe get 50 bucks byluisp8on june 18 2016  black must say admit little concerned choice bought said well heck lets give try mom anyway paper looked like nice budget phone also extra concerns cause im usa retuning table accept whatever came box fortunately wrong product came perfect condition brand new course problems packaging shipping first hands phone believe pay 60 bucks after taxes handling phone mean it guys blu inc  awesome job budget phones phone really awesome course look fancy premium obviously plastic fell like 500 phone hands looking budget phone hesitate one sorry 410 helpful helpful 5 comments report abuse'



# Vectorization
### CountVectorizer
```
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
count_vec = CountVectorizer(ngram_range=(1, 3),
                            max_df=0.50,
                            analyzer='word',
                           token_pattern = r'\w{2,}'
                           )
feature_train_counts = count_vec.fit(text)
bag_of_words = feature_train_counts.transform(text)

from sklearn.model_selection import train_test_split
X = bag_of_words 
y = df['Stars'].astype(float)
feature_train, feature_test, label_train, label_test = train_test_split(X, y, test_size=0.33, shuffle=True)
```
```
bag_of_words
```


# Model Building
## SGDClassifier
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
