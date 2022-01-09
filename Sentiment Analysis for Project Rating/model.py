import pandas as pd
import numpy as np
import pickle
import spacy
import re
# nlp = spacy.load("en_core_web_sm")

df=pd.read_csv('amazon_baby.csv',usecols=['review','rating'])
df=df.dropna()

# from IPython import get_ipython

# get_ipython().system('pip install git+https://github.com/laxmimerit/preprocess_kgptalkie.git')

import preprocess_kgptalkie as ps

print('ps line done')
for i in range(0,len(df)-1):
    if type(df.iloc[i]['review']) != str:
        df.iloc[i]['review'] = str(df.iloc[i]['review'])

# import preprocess_kgptalkie as ps



print('re line done')
def get_clean(x):
    x = str(x).lower().replace('\\', '').replace('_', ' ')
    x = ps.cont_exp(x)
    x = ps.remove_emails(x)
    x = ps.remove_urls(x)
    x = ps.remove_html_tags(x)
    x = ps.remove_accented_chars(x)
    x = ps.remove_special_chars(x)
    x = re.sub("(.)\\1{2,}", "\\1", x)
    return x


df['review']=df['review'].apply(lambda x:get_clean(x))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC # something releated to selection vector machine
from sklearn.metrics import classification_report

tfidf= TfidfVectorizer()


X=tfidf.fit_transform(df['review'])
pickle.dump(tfidf, open("tfidf.pkl", "wb"))

print('pickle done')

y=df['rating']


print(X.shape,y.shape)


X_train,X_test, y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=200)


clf=LinearSVC(C=20,class_weight='balanced')
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

pickle.dump(clf, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
x='I think this product is normal. I like it'
x=get_clean(x)
vec=tfidf.transform([x])
print(model.predict(vec))

#print(classification_report(y_test,y_pred))

"""
x='My neighbor had the Nojo and liked it so I was going to get one too. I tried hers, but I couldnt get it tight enough so my baby hung down too low which didnt seem safe and hurt my back. I thought I could order a different'
x=get_clean(x)
vec=tfidf.transform([x])
print(clf.predict(vec))


x="I don't like it"
x=get_clean(x)
vec=tfidf.transform([x])
clf.predict(vec)


x='I think is not good product , not comfortable for use more then 20 mints . and sound is average material is not good , I suggest to you not buy this product is sell time '
x=get_clean(x)
vec=tfidf.transform([x])
clf.predict(vec)

x='I think this product is normal. I like it'
x=get_clean(x)
vec=tfidf.transform([x])
clf.predict(vec)


x='worst product i have ever seen'
x=get_clean(x)
vec=tfidf.transform([x])
clf.predict(vec)


x='waste of money'
x=get_clean(x)
vec=tfidf.transform([x])
clf.predict(vec)

x='poor quality'
x=get_clean(x)
vec=tfidf.transform([x])
clf.predict(vec)


if ((clf.predict(vec))>=4 ):
  print('Good')
elif ((clf.predict(vec))==3):
  print('Normal')
else:
  print('Bad')


x='the product is amazing'
x=get_clean(x)
vec=tfidf.transform([x])
clf.predict(vec)

if ((clf.predict(vec))>=4 ):
  print('Good')
elif ((clf.predict(vec))==3):
  print('Normal')
else:
  print('Bad')


x='the product is fine'
x=get_clean(x)
vec=tfidf.transform([x])
clf.predict(vec)

"""

