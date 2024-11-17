#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Data manipulation
import pandas as pd
import numpy as np

#Data plotting
import matplotlib
import matplotlib.pyplot as plt
import squarify
import seaborn as sns

#Data cleaning
import re
import emoji

#Text analysis
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer, SnowballStemmer
from collections import Counter
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

#Sentiment analysis
from textblob import TextBlob

#Model building and checking
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score


# # Data Loading

# In[2]:


df_path = "/Users/ryambo/Desktop/Bitcoin_tweets.csv"
df = pd.read_csv(df_path,low_memory=False)

df.head(5)


# # Data preprocessing - Exploration - Text Analysis  

# In[13]:


#sorting the dataset by dates
df_timely = df.sort_values(by = 'date')


# In[107]:


pd.set_option('display.max_colwidth', None)


# In[14]:


#extracting fraction of the dataset for analysis
df_timely = df_timely.sample(frac=0.20, replace=False, random_state=1)
df_timely.reset_index(inplace=True)


# In[15]:


#checking for null values in the datset
df_timely.isnull().sum()


# In[6]:


df_timely.info()


# In[117]:


df_timely['text']


# In[16]:


df_timely = df_timely[df_timely['text'].notna()]


# In[17]:


#Cleaning the tweets for noise
def clean_tweets(tweet):
    tweet = re.sub("#bitcoin", 'bitcoin', tweet) #bitcoin - bitcoin
    tweet = re.sub("#Bitcoin", 'Bitcoin', tweet) #Bitcoin - Bitcoin
    tweet = re.sub('@[A-Za-z0-9]+', '', tweet) #strings containing '@'
    tweet = re.sub('#[A-Za-z0-9]+', '', tweet) #strings containing '#'
    tweet = re.sub('\\n', '', tweet) #'\n' 
    tweet = ''.join(c for c in tweet if c not in emoji.UNICODE_EMOJI) #Emojis
    tweet = tweet.replace("#", "").replace("_", " ") #remove hashtag retain the text
    tweet = re.sub('https:\/\/\S+', '', tweet) #URLs
    return tweet


# In[18]:


#Converting the text column to string datatype
df_timely['text']=df_timely['text'].apply(str)


# In[19]:


#cleaning noise
df_timely['text'] = df_timely['text'].apply(clean_tweets)


# In[20]:


#clearing out the emojis from tweets
def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)

df_timely['text'] = df_timely['text'].apply(deEmojify)


# # Stop words removal

# In[21]:


#extracting english stop words
stopwords_eng=set(stopwords.words("english"))


# In[22]:


#removing the negative impact words from stopwords
stopwords_eng.remove('not')
stopwords_eng.remove('down')
stopwords_eng.remove('off')
stopwords_eng.remove('no')


# In[23]:


#removing stopwords from the data
df_timely['text'] = df_timely['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords_eng)]))


# # Lemmatization/stemming and tokenization

# In[24]:


#defining tokonizer and lemmatizers
tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = PorterStemmer()


# In[25]:


#function to stem, lemmatize while tokenizing
def lemmatize_text(text):
    return [lemmatizer.stem(w) for w in tokenizer.tokenize(text)]
df_timely['lemmatized'] = df_timely.text.apply(lemmatize_text)
df_timely['text_processed'] = df_timely['lemmatized'].str.join(' ')
df_timely['text_processed'] = df_timely['text_processed'].apply(deEmojify)


# In[26]:


df_timely['text_processed'] = df_timely['text_processed'].str.replace(r',','', regex=True)
df_timely['text_processed'] = df_timely['text_processed'].str.replace(r'.','', regex=True)
df_timely['text_processed'] = df_timely['text_processed'].str.replace(r'?','', regex=True)
df_timely['text_processed'] = df_timely['text_processed'].str.replace(r'-','', regex=True)
df_timely['text_processed'] = df_timely['text_processed'].str.replace(r'|','', regex=True)
df_timely['text_processed'] = df_timely['text_processed'].str.replace(r'@','', regex=True)
df_timely['text_processed'] = df_timely['text_processed'].str.replace(r'=','', regex=True)
df_timely['text_processed'] = df_timely['text_processed'].str.replace(r'&amp;','', regex=True)


# In[27]:


df_timely['text_processed'] = df_timely['text_processed'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords_eng)]))


# # word frequency tree map

# In[28]:


#Extracting top 100 words based on their frequency
words = pd.Series(np.concatenate([x.split() for x in df_timely['text_processed']])).value_counts()
words100 = words[:100]


# In[29]:


words100


# In[80]:


w = w.drop(':')


# In[83]:


#Plotting tree of top 100 frequent words
fig = plt.gcf()
ax = fig.add_subplot()
plt.title('Top 100 frequent words')
fig.set_size_inches(16, 4.5)
squarify.plot(
sizes=w, 
label=w.index, 
alpha=.7,
bar_kwargs=dict(linewidth=1, edgecolor="#222222"),text_kwargs={'fontsize':10})


# In[32]:


#word cloud of frequent words top 200
words = ''
for i in df_timely.text_processed.values:
    words += '{} '.format(i.lower())
# Create a pandas dataframe with the word and its frequency
wd = pd.DataFrame(Counter(words.split()).most_common(200), columns=['word', 'frequency'])
# Convert the dataframe to a dictionary
data = dict(zip(wd['word'].tolist(), wd['frequency'].tolist()))

wc = WordCloud(background_color='white',
               stopwords=STOPWORDS,
               max_words=400).generate_from_frequencies(data)
fig = plt.gcf()
ax = fig.add_subplot()
fig.set_size_inches(20, 6.5)
plt.imshow(wc, interpolation='bilinear')
plt.show()


# 
# # Sentiment Analysis

# In[33]:



#function for subjectivity calculation
def subjectivity(twt):
    return TextBlob(twt).sentiment.subjectivity

# function for polarity calculation
def polarity(twt):
    return TextBlob(twt).sentiment.polarity

df_timely['subjectivity'] = df_timely['text_processed'].apply(subjectivity)
df_timely['polarity'] = df_timely['text_processed'].apply(polarity)


# In[34]:


#working dataset with necessary columns only
df_sa = df_timely.filter(['date','text_processed', 'subjectivity', 'polarity'], axis=1)


# In[85]:


#plotting subjectivity against polarity to see the spread
plt.figure(figsize=(8,5))

for i in range(0, 2000):
    plt.scatter(df_sa["polarity"].iloc[[i]].values[0], df_sa["subjectivity"].iloc[[i]].values[0], color="blue")

plt.title("Scatter Plot - Polarity vs Subjectivity")
plt.xlabel('polarity')
plt.ylabel('subjectivity')
plt.show()


# In[36]:


#function to extract sentimental value
def sentiment(val):
    if val < 0:
        return "negative"
    elif val == 0:
        return "neutral"
    else:
        return "positive"


# In[38]:


#extracting sentiments for each tweet based on polarity
df_sa['sentiment'] = df_sa['polarity'].apply(sentiment)

df_sa['text_processed'] = df_sa['text_processed'].apply(deEmojify)
df_sa.head(10)


# In[119]:


#plotting sentimets to demonstrate their count
df_sa['sentiment'].value_counts().plot(kind="bar")
plt.title("Polarity grouping bar plot")
plt.xlabel("sentiment groups")
plt.ylabel("counts")
plt.show()


# In[104]:


# Frequently occuring Negative impact words
df_sa['text_processed'] = df_sa['text_processed'].apply(clean_tweets)
df_sa['text_processed'] = df_sa['text_processed'].str.replace(r'[','', regex=True)
df_sa['text_processed'] = df_sa['text_processed'].str.replace(r']','', regex=True)
df_sa['text_processed'] = df_sa['text_processed'].str.replace(r':','', regex=True)
words = ''
negative = df_sa[df_sa['sentiment'] == 'negative' ]

for i in negative.text_processed.values:
    words += '{} '.format(i.lower())
wd1 = pd.DataFrame(Counter(words.split()).most_common(100), columns=['word', 'frequency'])
data1 = dict(zip(wd1['word'].tolist(), wd1['frequency'].tolist()))

wc1 = WordCloud(background_color='white',
               stopwords=STOPWORDS,
              max_words=400).generate_from_frequencies(data1)
plt.title('Negative sentiment', fontsize = 15)
fig = plt.gcf()
fig.set_size_inches(5, 5)
plt.imshow(wc1, interpolation='bilinear')
plt.show()


# In[101]:


negative


# In[100]:


# Frequently occuring Positive impact words
words = ''
positive = df_sa[df_sa['sentiment'] == 'positive' ]
for i in positive.text_processed.values:
    words += '{} '.format(i.lower())
wd2 = pd.DataFrame(Counter(words.split()).most_common(100), columns=['word', 'frequency'])
data2 = dict(zip(wd2['word'].tolist(), wd2['frequency'].tolist()))

wc2 = WordCloud(background_color='white',
               stopwords=STOPWORDS,
              max_words=400).generate_from_frequencies(data2)
plt.title('Positive sentiment', fontsize = 15)
fig = plt.gcf()
fig.set_size_inches(5,5)
plt.imshow(wc, interpolation='bilinear')
plt.show()


# # Analysing btc price for the observed period as tweets 

# In[42]:


#BTC data loading
df_btc = pd.read_csv("/Users/ryambo/Desktop/BTC-USD.csv")
df_btc.Date = pd.to_datetime(df_btc.Date)
df_btc.head(2)


# In[120]:


df_btc.info()


# # grouping tweets and btc-usd time series

# In[43]:


df_mt = df_timely.copy()


# In[44]:


tweets = df_mt.copy()
tweets['date'] = pd.to_datetime(tweets['date'],utc=True,errors='coerce')
tweets.date = tweets.date.dt.tz_localize(None)
tweets.index = tweets['date']
grpd_tweets = tweets.resample('1h').sum()


# In[45]:


btc_usd = df_btc.copy()
btc_usd['Date'] = pd.to_datetime(btc_usd['Date'], unit='s')
btc_usd.index = btc_usd['Date']
grpd_btc = btc_usd.resample('D')['Close'].mean()


# In[46]:


#defining starting and ending periods for observing the trend
start = max(grpd_tweets.index.min().replace(tzinfo=None), grpd_btc.index.min())
end = min(grpd_tweets.index.max().replace(tzinfo=None), grpd_btc.index.max())
grpd_tweets = grpd_tweets[start:end]
grpd_btc = grpd_btc[start:end]


# In[47]:


#Bitcoin and tweet timeseries trend for the observed period
fig, ax1 = plt.subplots(figsize=(20,10))
ax1.set_title("BTC Progression wrt tweet sentiments", fontsize=18)
ax1.tick_params(labelsize=14)
ax2 = ax1.twinx()
ax1.plot_date(grpd_tweets.index, grpd_tweets, 'black')
ax2.plot_date(grpd_btc.index, grpd_btc, 'red')
ax1.set_ylabel("Tweet Sentiments", color='black', fontsize=16)
ax2.set_ylabel(" BTC", color='red', fontsize=16)
plt.show()


# In[48]:


#normalizing tweet data

min_max_scaler = preprocessing.StandardScaler()
scld_score = min_max_scaler.fit_transform(grpd_tweets['polarity'].values.reshape(-1,1))
grpd_tweets['normalized_score'] = scld_score
grp_btc_scaled = grpd_btc / max(grpd_btc.max(), abs(grpd_btc.min()))


# In[49]:


# Derivative
grpd_tweets_derivative = pd.Series(np.gradient(grpd_tweets['normalized_score'].values), grpd_tweets.index, name='slope')
grpd_btc_derivative = pd.Series(np.gradient(grpd_btc.values), grpd_btc.index, name='slope')


# In[106]:


fig, ax1 = plt.subplots(figsize=(10,5))
ax1.set_title("Rate of change of sentiment vs bitcoin price", fontsize=18)
ax1.tick_params(labelsize=14)

ax2 = ax1.twinx()
ax1.plot_date(grpd_tweets_derivative.index, grpd_tweets_derivative, 'black')
ax2.plot_date(grpd_btc_derivative.index, grpd_btc_derivative, 'red')

ax1.set_ylabel("Derivative of Tweet sentiments", color='black', fontsize=16)
ax2.set_ylabel('Derivative of Bitcoin', color='red', fontsize=16)
plt.show()


# # Data preparation for model building

# In[51]:


df_twt_btc = df_sa.copy()
#cleaning the data 
df_twt_btc['Date'] = pd.to_datetime(df_twt_btc['date'], errors='coerce').dt.strftime('%Y-%m-%d')


# In[52]:


def btc_category(rate):
    if rate < 1:
        return 'negative'
    elif rate == 1:
        return 'neutral'
    else:
        return 'positive'
def window(duration):
    val = grpd_btc.shift(duration)/grpd_btc
    val = val.apply(btc_category)
    return val 

btc_price = window(7) 
df_twt_btc['btc_category'] = df_twt_btc.Date.apply(lambda x: btc_price[x] if x in btc_price else np.nan)


# In[53]:


df_twt_btc


# In[54]:


df_twt_btc['result'] = df_twt_btc['sentiment'] == df_twt_btc['btc_category']


# In[55]:


df_twt_btc.head(10)


# # NLP Model building

# In[56]:


#defining X and y
X=df_sa.text_processed
y=df_sa.sentiment


# In[57]:


#Splitting the dataset, train and test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.05, random_state =26105111)


# In[59]:


#TF-IDF - Dataset transformation
vectorise = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
vectorise.fit(X_train)
print('feature words: ', len(vectorise.get_feature_names_out()))


# In[60]:


#transforming training and validation
X_train = vectorise.transform(X_train)
X_test  = vectorise.transform(X_test)


# In[108]:


#Model Evaluation

def model_checking1(model):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    classes = ['Negative','Neutral', 'Positive']
    cm = confusion_matrix(y_pred,np.array(y_test))

    # plot confusion matrix
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, cmap=plt.cm.Blues, annot=True, fmt='d', 
                xticklabels=classes,
                yticklabels=classes)
    plt.title('Confusion matrix - Bernoulli NB', fontsize=16)
    plt.xlabel('Actual label', fontsize=12)
    plt.ylabel('Predicted label', fontsize=12)
    


# In[109]:


#Model Building - Bernoulli
BNBmodel = BernoulliNB()

BNBmodel.fit(X_train, y_train)
#model checking
model_checking1(BNBmodel)
y_pred1 = BNBmodel.predict(X_test)
#accuracy
print('Accuracy:\t{:0.1f}%'.format(accuracy_score(y_test,y_pred1)*100))


# In[111]:


def model_checking2(model):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    classes = ['Negative','Neutral', 'Positive']
    cm = confusion_matrix(y_pred,np.array(y_test))

    # plot confusion matrix
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, cmap=plt.cm.Blues, annot=True, fmt='d', 
                xticklabels=classes,
                yticklabels=classes)
    plt.title('Confusion matrix - Multinomial NB', fontsize=16)
    plt.xlabel('Actual label', fontsize=12)
    plt.ylabel('Predicted label', fontsize=12)


# In[112]:


from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
# Model Generation Using Multinomial Naive Bayes
clf = MultinomialNB().fit(X_train, y_train)
predicted= clf.predict(X_test)
model_checking2(clf)
print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))


# In[ ]:





# In[ ]:




