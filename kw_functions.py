# Library for Keywords Extraction
# Methods included: Yake, Rake, TextRank, TFIDF
# !pip install yake rake_nltk spacy nltk scikit-learn textract

from spacy_textrank import TextRank4Keyword
from sklearn.feature_extraction.text import TfidfVectorizer
from rake_nltk import Rake
import yake
import pandas as pd

# Functions For Keyword Extraction
def yake_extractor(text):
	kw_extractor = yake.KeywordExtractor()
	my_keyword_with_yake = kw_extractor.extract_keywords(text)
	return my_keyword_with_yake

def rake_extractor(text):
	r = Rake()
	r.extract_keywords_from_text(text)
	return r.get_ranked_phrases_with_scores()

def spacy_txtrank_extractor(text,num_of_words=30):
	txtRank = TextRank4Keyword()
	txtRank.analyze(text,candidate_pos=['NOUN','PROPN'],window_size=4)
	return txtRank.get_keywords(num_of_words)	

def tfidf_extractor(text):
	tfidf = TfidfVectorizer(ngram_range=(2,2))
	tfidf.fit(text)
	df = pd.DataFrame({'Keyword':tfidf.get_feature_names(),'Scores':tfidf.idf_})
	return df 

# Convert to Dataframe
def convert_to_df(my_dict):
	df = pd.DataFrame(list(my_dict.items()),columns=['Keyword','Scores'])
	return df 

def process_file_to_string(file):
	text = textract.process(file)
	return text.decode('utf-8')
