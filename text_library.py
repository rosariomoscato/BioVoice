import streamlit as st

# NLP Pkgs
#python -m spacy download en_core_web_sm
#python -m textblob.download_corpora
import nltk
import spacy
nlp  = spacy.load('en_core_web_sm')
from spacy import displacy

import neattext as nt 
import neattext.functions as nfx 
from textblob import TextBlob

from collections import Counter

# Reading Files
import docx2txt
import pdfplumber

# Viz Pkgs
from wordcloud import WordCloud 
import matplotlib.pyplot as plt 
import matplotlib 
matplotlib.use('Agg')
import altair as alt 

# Custom Pkgs
from kw_functions import *


# Functions
def get_most_common_tokens(docx,num=10):
	word_freq = Counter(docx.split())
	most_common_tokens = word_freq.most_common(num)
	return dict(most_common_tokens)



def kw_processing(text_file,kw_method="TextRank"):
	# Keywords Processing of Text
	if kw_method == 'TextRank':
		keyword_result = spacy_txtrank_extractor(text_file)
		resulting_keywords_as_df = convert_to_df(dict(keyword_result))
	
	elif kw_method == 'TFIDF':
		processed_text = nfx.remove_stopwords(text_file.lower())
		keyword_result = tfidf_extractor(processed_text.split('.'))
		resulting_keywords_as_df = keyword_result

	elif kw_method == 'Yake':
		keyword_result = yake_extractor(text_file)
		inverted_dict = {v:k for k,v in dict(keyword_result).items()}
		resulting_keywords_as_df = convert_to_df(inverted_dict)

	elif kw_method == 'Rake':
		keyword_result = rake_extractor(text_file)
		inverted_dict = {v:k for k,v in dict(keyword_result).items()}
		resulting_keywords_as_df = convert_to_df(inverted_dict)

	else:
		# Remove the stopwords
		processed_text = nfx.remove_stopwords(text_file.lower())
		keyword_result = Counter(processed_text.split(' '))
		#st.write(keyword_result.most_common(10))
		resulting_keywords_as_df = convert_to_df(dict(keyword_result.most_common(30)))

	# Printing Bar Char
	my_chart = alt.Chart(resulting_keywords_as_df).mark_bar().encode(x='Keyword',y='Scores')
	st.altair_chart(my_chart)

	# General
	#st.dataframe(resulting_keywords_as_df)


TAGS = {
            'NN'   : 'green',
            'NNS'  : 'green',
            'NNP'  : 'green',
            'NNPS' : 'green',
            'VB'   : 'blue',
            'VBD'  : 'blue',
            'VBG'  : 'blue',
            'VBN'  : 'blue',
            'VBP'  : 'blue',
            'VBZ'  : 'blue',
            'JJ'   : 'red',
            'JJR'  : 'red',
            'JJS'  : 'red',
            'RB'   : 'cyan',
            'RBR'  : 'cyan',
            'RBS'  : 'cyan',
            'IN'   : 'darkwhite',
            'POS'  : 'darkyellow',
            'PRP$' : 'magenta',
            'PRP$' : 'magenta',
            'DET'   : 'black',
            'CC'   : 'black',
            'CD'   : 'black',
            'WDT'  : 'black',
            'WP'   : 'black',
            'WP$'  : 'black',
            'WRB'  : 'black',
            'EX'   : 'yellow',
            'FW'   : 'yellow',
            'LS'   : 'yellow',
            'MD'   : 'yellow',
            'PDT'  : 'yellow',
            'RP'   : 'yellow',
            'SYM'  : 'yellow',
            'TO'   : 'yellow',
            'None' : 'off'
        }

def get_pos_tags(docx):
	blob = TextBlob(docx)
	tagged_docx = blob.tags
	#tagged_df = pd.DataFrame(tagged_docx,columns=['token','tags'])
	return tagged_docx


def mytag_visualizer(tagged_docx):
	colored_text = []
	for i in tagged_docx:
		if i[1] in TAGS.keys():
		   token = i[0]
		   color_for_tag = TAGS.get(i[1])
		   result = '<span style="color:{}">{}</span>'.format(color_for_tag,token)
		   colored_text.append(result)
	result = ' '.join(colored_text)
	return result


def plot_wordcloud(docx):
	mywordcloud = WordCloud().generate(docx)
	#plt.imshow(mywordcloud,interpolation='bilinear')
	fig, ax = plt.subplots()
	ax.imshow(mywordcloud,interpolation='bilinear')
	plt.axis('off')
	st.pyplot(fig)


def plot_mendenhall_curve(docx):
	word_length = [ len(token) for token in docx.split()]
	word_length_count = Counter(word_length)
	sorted_word_length_count = sorted(dict(word_length_count).items())
	x,y = zip(*sorted_word_length_count)
	fig = plt.figure(figsize=(20,10))
	plt.plot(x,y)
	plt.title("Plot of Word Length Distribution (Mendenhall Curve)")
	plt.show()
	st.pyplot(fig)
	

# DOCUMENT Fetching
def get_docx(docx_file):
	return docx2txt.process(docx_file)


def get_pdf(pdf_file):
	pdf_file = pdfplumber.open(pdf_file)
	p0 = pdf_file.pages[0]
	return p0.extract_text()


def get_txt(text_file):
	return str(text_file.read(),"utf-8")