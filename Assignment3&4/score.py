import pickle
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Importing Count Vectorizer and Word Vector
vec_path = "./models/word_vec.sav"
tfidf_path = "./models/tfidf.sav"

nb_path = "./models/nb_model.sav"
lr_path = "./models/lr_model.sav"
rf_path = "./models/rf_model.sav"

word_vec = pickle.load(open(vec_path, "rb"))
tfidf = pickle.load(open(tfidf_path, "rb"))

spam_detectorNB = pickle.load(open(nb_path, "rb"))


def tokenizer(text):
	text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE) # Remove URL
	text = re.sub(r'^http?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE) 

	tokens = word_tokenize(text)
	
	stopwords = list(nltk.corpus.stopwords.words('english'))
	lemmatizer = WordNetLemmatizer()
	final_tokens = []

	for token in tokens:
		if token.lower() not in stopwords:
			token = lemmatizer.lemmatize(token.lower())
			final_tokens.append(token.lower())

	return final_tokens

# Defining Score Function
def score(text: str, model, threshold: float, word_vec = word_vec, tfidf = tfidf): 
	tokens = tokenizer(text)
	text = ' '.join(tokens)
	
	bow_transformer = word_vec.transform([text])
	text_tfidf = tfidf.transform(bow_transformer)[0]
	propensity = model.predict_proba(text_tfidf)[0][1]

	if propensity >= threshold:
		return True, propensity
	else:
		return False, propensity

if __name__ == '__main__':
	print(score("Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005", spam_detectorNB, 0.75))
