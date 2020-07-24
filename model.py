import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pickle
nltk.download('stopwords')

#https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
#link to dataset

def preprocessing(data):

	def remove_URL(sample):
	    """Remove URLs from a sample string"""
	    return re.sub(r"http\S+", "", sample)

	TAG_RE = re.compile(r'<[^>]+>')
	def remove_tags(text):
		
	    return TAG_RE.sub('', text)

	def remove_emoji(string):
	    """
	    This method removes emojis,symbols and flags in the text
	    """
	    emoji_pattern = re.compile(
	      "["
	      u"\U0001F600-\U0001F64F" #emoticons
	      u"\U0001F300-\U0001F5FF" #symbols and pictographs
	      u"\U0001F680-\U0001F6FF" #transport and map symbols
	      u"\U0001F1E0-\U0001F1FF" #flags
	      u"\U00002702-\U000027B0"
	      u"\U000024C2-\U0001F251"
	      "]+",
	      flags=re.UNICODE
	    )
	    return emoji_pattern.sub("[^a-zA-Z]",string)

	def remove_punct(text):
	    """
	    This method removes all the punctions in the text
	    """
	    table = str.maketrans("","",string.punctuation)
	    return text.translate(table)

	stopwords = set(nltk.corpus.stopwords.words("english"))

	def remove_stopwords(text):
	    """
	    This method removes the stopwords from the text
	    """
	    text = [word.lower() for word in text.split() if word.lower() not in stopwords]
	    return " ".join(text)

	data['review'] = data.review.apply(lambda x:remove_URL(x))
	data['review'] = data.review.apply(lambda x:remove_tags(x))
	data['review'] = data.review.apply(lambda x:remove_emoji(x)) 
	data['review'] = data.review.apply(lambda x:remove_punct(x))
	data['review'] = data.review.apply(lambda x:remove_stopwords(x))


	return data['review']


def main():
	data = pd.read_csv('IMDB Dataset.csv',sep=',')
	# print(data.head())
	data['sentiment'] = data['sentiment'].map({'positive':1,'negative':0})
	# print(data.head())
	data['review'] = preprocessing(data)
	X_train, X_test, y_train, y_test = train_test_split(data['review'], data['sentiment'], test_size=0.2, random_state=101)
	pipeline = Pipeline([
    	('bow',CountVectorizer())
    	,('tdidf',TfidfTransformer()),
    	('classifier',MultinomialNB())])
	pipeline.fit(np.array(X_train),np.array(y_train))
	pred = pipeline.predict(X_test)
	print(classification_report(y_test,pred))
	filename = "model.sav"
	pickle.dump(pipeline,open(filename,'wb'))

if __name__ == '__main__':
	main()