import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from wordcloud import STOPWORDS
from wordcloud import ImageColorGenerator
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from warnings import filterwarnings
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import joblib


filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

data = pd.read_csv("UpdatedResumeDataSet.csv")
nltk.download('stopwords')

print(data.head())

print(data.size)

print(data.shape)

print(data.info())

print(data.isnull().sum())

print(data['Category'].value_counts())

# Create count plot for categoris columns.

plt.figure(figsize=(25,8))
ax = sns.countplot(x = 'Category', data= data,palette = 'mako')
ax.set_title("Count of Categories.",fontweight = 'bold',size=32)
plt.xticks(rotation=80)
ax.set_ylabel('Count',fontweight='bold',size=34)
ax.set_xlabel("Categories",fontweight = 'bold',size=34)
plt.show()


# Most common words
text = " ".join(i for i in data.Resume)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.figure( figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()



data_science_resumes = data[data['Category'] == 'Data Science']['Resume']
text = ' '.join(data_science_resumes)

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(15, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# Cleaning the Resume texts
corpus = []
for i in range(0, len(data)):
  review = re.sub('[^a-zA-Z]', ' ', data['Resume'][i])
  review = review.lower()
  review = review.split()
  ps = PorterStemmer()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  review = ' '.join(review)
  corpus.append(review)

print(corpus[1])


# Creating the Bag of Words model
cv = CountVectorizer(max_features = 1500,ngram_range=(1,3))
X = cv.fit_transform(corpus).toarray()
y = data['Category'].values


# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)

# Training the Naive Bayes model on the Training set
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
print(acc)

# Creating the TF - IDF Vectorizer model
tfidf_v=TfidfVectorizer(max_features=5000,ngram_range=(1,3))
X=tfidf_v.fit_transform(corpus).toarray()
y=data['Category'].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X1_train, X1_test, y1_train, y1_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
print(X1_train.shape)

print(X1_test.shape)

print(y1_train.shape)

print(y1_test.shape)

tfidf_v.fit(corpus)

X1_train_tfidf = tfidf_v.transform(corpus)

feature_names = tfidf_v.get_feature_names_out()

count_df = pd.DataFrame(X1_train_tfidf.toarray(), columns=feature_names)
print(count_df.head())

# Naive Bayes
classifier = GaussianNB()
classifier.fit(X1_train, y1_train)
y1_pred = classifier.predict(X1_test)

cm = confusion_matrix(y1_test, y1_pred)

acc1 = accuracy_score(y1_test, y1_pred)

cm = confusion_matrix(y1_test, y1_pred)

class_names = np.unique(y1_test)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

print(acc1)

logreg = LogisticRegression()

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'penalty': ['l1', 'l2']}

grid_search = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)

grid_search.fit(X1_train, y1_train)

print("En iyi parametreler:", grid_search.best_params_)
print("En iyi doğruluk skoru:", grid_search.best_score_)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X1_test)

cm = confusion_matrix(y1_test, y_pred)

class_names = np.unique(y1_test)

# Heatmap ile confusion matrix'i görselleştirme
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix for Logistic Regression")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


joblib.dump(best_model, 'logistic_regression_model.pkl')

tfidf_v = joblib.dump(tfidf_v,'tfidf_vectorizer.pkl')