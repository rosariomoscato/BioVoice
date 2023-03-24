# Core Pkgs
import streamlit as st
st.set_page_config(page_title="SETEM by RML",page_icon="./imgs/Rosario_Moscato_LAB_120_120.png")
import streamlit.components.v1 as stc

from ui_templates import HTML_BANNER, HTML_STICKER, HTML_BANNER_SKEWED, HTML_WRAPPER

# Data Pkgs
import pandas as pd


# Plotting Pkgs
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.use('Agg')
import altair as alt 


import numpy as np
from scipy.stats import norm

from PIL import Image
import os
import time
from datetime import datetime

#Prosody Pkgs
import myprosody as mysp
import pickle

#Speech Recognition Pkg
import speech_recognition as sr
#from speech_recognition import Microphone, Recognizer, AudioFile


#Import Fxs Libraries
import text_library

#Import NLP and Text Libraries
#!pip install gensim gensim_sum_ext
from gensim.summarization.summarizer import summarize

import neattext as nt 
import neattext.functions as nfx 

import nltk
import spacy
nlp  = spacy.load('en_core_web_sm')
from spacy import displacy

from google_trans_new import google_translator





@st.cache
def load_image(img):
	im = Image.open(os.path.join(img))
	return im



def voice_similarity(p1,p2):

    c = r"/home/rosario/Projects/Streamlit/BioVoice/myprosody/"

    f0mean_p1 = mysp.myspf0mean(p1,c)
    f0mean_p2 = mysp.myspf0mean(p2,c)
    f0mean_delta = round(abs(f0mean_p1-f0mean_p2),2)
    #st.write("F0 Mean Delta:",f0mean_delta)

    f0sd_p1 = mysp.myspf0sd(p1,c)
    f0sd_p2 = mysp.myspf0sd(p2,c)
    f0sd_delta = round(abs(f0sd_p1-f0sd_p2),2)
    #st.write("F0 SD Delta:",f0sd_delta)

    f0med_p1 = mysp.myspf0med(p1,c)
    f0med_p2 = mysp.myspf0med(p2,c)
    f0med_delta = round(abs(f0med_p1-f0med_p2),2)
    #st.write("F0 MED Delta:",f0med_delta)
    

    f025_p1 = mysp.myspf0q25(p1,c)
    f025_p2 = mysp.myspf0q25(p2,c)
    f025_delta = round(abs(f025_p1-f025_p2),2)
    #st.write("F0 25P Delta:",f025_delta)

    f075_p1 = mysp.myspf0q75(p1,c)
    f075_p2 = mysp.myspf0q75(p2,c)
    f075_delta = round(abs(f075_p1-f075_p2),2)
    #st.write("F0 75P Delta:",f075_delta)

    # f0min_p1 = mysp.myspf0min(p1,c)
    # f0min_p2 = mysp.myspf0min(p2,c)
    # f0min_delta = round(abs(f0min_p1-f0min_p2),2)
    #st.write("F0 75P Delta:",f075_delta)

    # f0max_p1 = mysp.myspf0max(p1,c)
    # f0max_p2 = mysp.myspf0max(p2,c)
    # f0max_delta = round(abs(f0max_p1-f0max_p2),2)
    #st.write("F0 75P Delta:",f075_delta)

    features = {'Parameter': ['Mean Delta     ','St.Dev. Delta  ','Median Delta   ', '25th Per. Delta','75th Per. Delta'],
        'Value': [f0mean_delta,f0sd_delta,f0med_delta,f025_delta,f075_delta]
        }

    df_feat = pd.DataFrame(features, columns = ['Parameter', 'Value'])


    if (f0mean_delta + f0med_delta + f0sd_delta) <= 20.0:
        return "VERY HIGH", df_feat
    elif (f0mean_delta + f0med_delta + f0sd_delta) <= 45.0:
        return "HIGH", df_feat
    elif (f0mean_delta + f0med_delta + f0sd_delta) <= 70.0:
        return "MEDIUM", df_feat
    elif (f0mean_delta + f0med_delta + f0sd_delta) <= 100.0:
        return "LOW", df_feat
    else:
        return "VERY LOW", df_feat



def language_detect(text):

	detector = google_translator()  
	detect_result = detector.detect(text)

	return detect_result



def main():
	"""A Keywords Extractor NLP APP with Streamlit"""

	print("**************************************")
	print("* SETEM                              *")
	print("* SpEech and TExt Management tool by:*")
	print("* Rosario Moscato                    *")
	print("*                                    *")
	print("* 2021-All Rights Reserved           *")
	print("*                                    *")
	print("*                                    *")
	print("*                                    *")
	print("* email:                             *")
	print("* rosariomoscatolab@gmail.com        *")
	print("**************************************")

	enter_menu = ["Voice Analysis", "Voice Similarity", "Text Analysis", "Text Summarization", "Text Translation", "Sentiment from Text", "About"]
	enter_choice = st.sidebar.selectbox("Select a Module",enter_menu)

	languages = {'Chinese':'zh-CN','English':'en-US','French':'fr','Greek':'el','Hindi':'hi','Italian':'it','Japanese':'ja','Russian':'ru','Spanish':'es','Turkish':'tr'}


	#Audio Files Directory" 
	c = r"/home/rosario/Projects/Streamlit/BioVoice/myprosody/"

	audio_files_path = "./myprosody/dataset/audioFiles/"


	# STARTING VOICE PART ---------------------------------------------------------------------------------------------------
	# Menu "Voice"
	if enter_choice == "Voice Analysis":


		#stc.html(HTML_BANNER,width=800,height=200)

		html_templ = """
		<div style="background-color:#080e4c;padding:10px;border-radius:5px">
		<h1 style="color:#3e47a5">Voice Analysis</h1>
		</div>
		"""

		st.markdown(html_templ,unsafe_allow_html=True)
		st.write("A simple proposal for Voice Biometrics")

		placeholder = st.empty()
		placeholder.image(load_image('imgs/biovoice2.jpeg'), use_column_width=True)

		#File Uploading
		uploaded_file = st.sidebar.file_uploader("Upload Audio File (Wav)",type=['wav'])

		if uploaded_file is not None:
			file_details = {"Filename":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
			#st.write(file_details)
			
			# Check File Type
			if uploaded_file.type == "audio/x-wav":
				try:

					#Reading Wav File
					bytes_data = uploaded_file.read()

					#Playing Wav File
					st.sidebar.write("Click to play: " + file_details['Filename'])
					st.sidebar.audio(bytes_data, format='audio/x-wav')

					#File Name without WAV extension
					p = file_details['Filename'][:-4]


					menu = ["Prosody","Gender ID & Mood","Silence/Speech Rate", "Recognition & Transcription"]
					choice = st.sidebar.selectbox("Voice Analysis Menu",menu)



					# Menu "Prosody"
					if choice == "Prosody":

						st.write("")
						st.write("")
						placeholder.info("PROSODIC FEATURES (F0, Fundamental Frequency Distribution)")

						try:

							#f0 mean (global mean of fundamental frequency distribution Hz)
							f0mean = mysp.myspf0mean(p,c)
							st.write("F0 Mean (global mean in Hz):", f0mean)

							#f0 SD (global standard deviation of fundamental frequency distribution Hz)
							f0sd = mysp.myspf0sd(p,c)
							st.write("F0 SD (global standard deviation):", f0sd)

							#f0 MD (global median of fundamental frequency distribution Hz)
							f0med = mysp.myspf0med(p,c)
							st.write("F0 Median (global median in Hz):", f0med)

							#f0 Min (global minimum of fundamental frequency distribution)
							f0min = mysp.myspf0min(p,c)
							st.write("F0 Minimum (global minimum in Hz):", f0min)

							#f0 Max (global maximum of fundamental frequency distribution Hz)
							f0max = mysp.myspf0max(p,c)
							st.write("F0 Maximum (global maximum in Hz):", f0max)

							#f0 quan25 (global 25th quantile of fundamental frequency distribution Hz)
							f025 = mysp.myspf0q25(p,c)
							st.write("F0 25th Quantile (global 25th quantile in Hz):", f025)

							#f0 quan75 (global 75th quantile of fundamental frequency distribution)
							f075 = mysp.myspf0q75(p,c)
							st.write("F0 75th Quantile (global 75th quantile in Hz):", f075)


						except:
							st.write("ERRORE")	
						#total = mysp.mysptotal(file_details['Filename'][:-4],c)
						#st.write(total[0])



					# Menu "Gender ID & Mood"
					elif choice == "Gender ID & Mood":

						st.write("")
						st.write("")
						placeholder.info("Gender ID and Speech Mood (Semantic Analysis)")

						result = mysp.myspgend(p,c)
						st.write("Gender:",result[0])
						st.write("Mood of speech:",result[1])



					# Menu "Silent/Speech Rate"
					elif choice == "Silence/Speech Rate":

						st.write("")
						st.write("")
						placeholder.info("Silence and Speech Mode Analysis")

						#Original Duration (sec total speaking duration with pauses)
						original_duration = mysp.myspod(p,c)
						st.write("Original Duration (total speaking duration with pauses in seconds):", original_duration)

						#Speaking Duration (sec only speaking duration without pauses)
						speaking_duration = mysp.myspst(p,c)
						st.write("Speaking Duration (only speaking duration without pauses in seconds):", speaking_duration)

						#Balance Ratio (speaking duration/original duration)
						balance = mysp.myspbala(p,c)
						st.write("Balance Ratio (Speaking Duration/Original Duration):", balance)

						#Silent Rate
						silent_rate = original_duration - speaking_duration
						st.write("Silence Duration in seconds:", round(silent_rate,2))

						#Number of Syllables
						syllables = mysp.myspsyl(p,c)
						st.write("Number of Syllables:",syllables)

						#Number of Pauses
						pauses = mysp.mysppaus(p,c)
						st.write("Number of Pauses:", pauses)

						#Rate of Speech (syllables/second)
						speech_rate = mysp.myspsr(p,c)
						st.write("Rate of Speech (syllables/second in Original Duration):", speech_rate)

						#Articulation Rate
						articulation_rate = mysp.myspatc(p,c)
						st.write("Articulation Rate (syllables/second in Speaking Duration):", articulation_rate)



					# Menu "Recognition & Transcription"
					else: #choice == "Recognition & Transcription":

						st.write("")
						st.write("")
						placeholder.info("Voice Recognition & Transcription")

						lang = st.selectbox("Select a language", list(languages.keys()))

						if lang != None:

							if st.button("Transcribe"):

								try:

									r = sr.Recognizer()
									st.write("")
									with sr.AudioFile(audio_files_path+file_details['Filename']) as source:
										audio = r.record(source)  # read the entire audio file

									st.write("")
									transcription = r.recognize_google(audio, language=languages[lang])

									# datetime object containing current date and time
									now = datetime.now()
									# dd/mm/YY H:M:S
									dt_string = now.strftime("%Y%m%d%H%M%S")

									file_name = "./Transcriptions/Transcription_"+lang+"_"+dt_string+".txt"

									with open(file_name, "w") as f:
									    f.write(transcription)

									st.write("Transcription saved as: Transcription_"+lang+"_"+dt_string+".txt")    
									st.success("Transcribed Text")
									st.write(transcription)


								except sr.UnknownValueError:
									st.write("Audio not recognized")

								except sr.RequestError as e:
									st.write("Connection problems...")




				except:
					st.warning("Problem in loading Audio File...")



	# Enter Menu "Similarity"
	elif enter_choice == "Voice Similarity":

		html_templ = """
		<div style="background-color:#080e4c;padding:10px;border-radius:5px">
		<h1 style="color:#3e47a5">Voice Similarity</h1>
		</div>
		"""

		st.markdown(html_templ,unsafe_allow_html=True)
		st.write("Voice Similarity Analysis by Prosodic Analisys (Experimental Feature)")

		placeholder = st.empty()
		placeholder.image(load_image('imgs/voice_id.jpeg'), use_column_width=True)

		similarity_files = st.file_uploader("Select 2 Audio File (Wav)",type=['wav'], accept_multiple_files=True)
		if similarity_files is not None and len(similarity_files) == 2:

			similarity_name_1 = similarity_files[0].name 
			similarity_name_2 = similarity_files[1].name

			similarity_name_1 = similarity_name_1[:-4]
			similarity_name_2 = similarity_name_2[:-4]

			st.info("F0 Functions Comparative Analysis")

			result, df_res = voice_similarity(similarity_name_1,similarity_name_2)

			st.dataframe(df_res)

			output = "Voice Similarity Rate: " + result

			if result == "VERY HIGH" or result == "HIGH":
				st.success(output)
			else:
				st.warning(output)

		else:
			st.warning("Select 2 Wav Files")

	# ENDING VOICE PART ---------------------------------------------------------------------------------------------------




	# STARTING TEXT PART ---------------------------------------------------------------------------------------------------
	#Menu "Text Analysis"
	elif enter_choice == "Text Analysis":

		html_templ = """
		<div style="background-color:#080e4c;padding:10px;border-radius:5px">
		<h1 style="color:#3e47a5">Text Analysis (only for English)</h1>
		</div>
		"""

		st.markdown(html_templ,unsafe_allow_html=True)
		st.write("A simple proposal for Text Analytics with NLP")

		placeholder = st.empty()
		placeholder.image(load_image('imgs/text_analytics2.jpeg'), use_column_width=True)



		#File Uploading
		raw_text_file = st.sidebar.file_uploader("Upload File",type=['txt','docx','pdf'])
		raw_text = ""

		if raw_text_file is not None:
			file_details = {"Filename":raw_text_file.name,"FileType":raw_text_file.type,"FileSize":raw_text_file.size}
			#st.write(file_details)
			# Check File Type
			if raw_text_file.type == "text/plain":
				try:
					raw_text = text_library.get_txt(raw_text_file)
					placeholder.write("")
				except:
					st.write("TXT File Fetching Problem...")
			elif raw_text_file.type == "application/pdf":
				try:
					raw_text = text_library.get_pdf(raw_text_file)
					placeholder.write("")
				except:
					st.write("PDF File Fetching Problem...")
			elif raw_text_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
				try:
					raw_text = text_library.get_docx(raw_text_file)
					placeholder.write("")
				except:
					st.write("DOCX File Fetching Problem...")

			if raw_text != "":

				st.info("Text from file")
				language = language_detect(raw_text)
				language_code = language[0]
				#st.write(language_code)
				#st.write (language[1])
				st.write(raw_text)

				#Language Code Condition
				if language_code == 'en':

					text_menu = ["Stats & Most Common Tokens", "Key Words Extraction","Named Entity Recognition","PoS Tagger","Visualizations"]
					text_choice = st.sidebar.selectbox("Text Analysis Menu",text_menu)

					max_limit = len(raw_text.split())
					#Text with NO STOPWORDS
					process_text = nfx.remove_stopwords(raw_text)

					if text_choice == 'Stats & Most Common Tokens':

						st.write("")
						st.info("Text Stats")
						word_desc = nt.TextFrame(raw_text).word_stats()
						result_desc = {"Length of Text":word_desc['Length of Text'],
										"Num of Vowels":word_desc['Num of Vowels'],
										"Num of Consonants":word_desc['Num of Consonants'],
										"Num of Stopwords":word_desc['Num of Stopwords']}
						st.write(result_desc)

						num_of_tokens = st.number_input("Num of Tokens to be showed",1,max_limit, value = 10)

						most_common_tokens = text_library.get_most_common_tokens(process_text,num_of_tokens)
						tk_df = pd.DataFrame({'tokens':most_common_tokens.keys(),'counts':most_common_tokens.values()})

						st.info("Plot For Most Common Tokens")
						brush = alt.selection(type='interval', encodings=['x'])
						c = alt.Chart(tk_df).mark_bar().encode(
						    x='tokens',
						    y='counts',
						    opacity=alt.condition(brush, alt.OpacityValue(1), alt.OpacityValue(0.7)),
						    ).add_selection(brush)
						
						st.altair_chart(c,use_container_width=True)

						st.info("Most Common Tokens")
						st.dataframe(tk_df)


					elif text_choice == 'Key Words Extraction':

						st.write("")
						st.info("Key Words Extractioon")
						kw_method = st.selectbox("Extraction Method",("TextRank","Yake","Rake"))
						#if st.button('Extract'):
						text_library.kw_processing(raw_text,kw_method)


					elif text_choice == 'Named Entity Recognition':

						ent_list = []
						entity = []
						start = []
						end = []
						category = []

						docx = nlp(raw_text)

						for ent in docx.ents:

							entity.append(ent.text)
							start.append(ent.start_char)
							end.append(ent.end_char)
							category.append(ent.label_)

						data = {'Entity': entity,'Start':start,'End':end,'Category': category}

						df_ner = pd.DataFrame(data, columns = ['Entity', 'Start', 'End', 'Category'])

						st.subheader("List of Entities")
						st.dataframe(df_ner)

						st.write("")

						if st.button("Click to visualize NER in the text"):

							html = displacy.render(docx,style="ent")
							html = html.replace("\n\n","\n")
							result = HTML_WRAPPER.format(html)
							stc.html(result,scrolling=True)



					elif text_choice == 'PoS Tagger':
						st.info("Part of Speech Tags Visualization (green: nouns, blue: verbs, red: adjectives, etc.)")
						processed_tags = text_library.get_pos_tags(raw_text)
						tagged_docx = text_library.mytag_visualizer(processed_tags)
						stc.html(tagged_docx,scrolling=True)



					else: #Visualizations

						st.write("")
						st.info("Text Visualization")
						visualization_kind = st.selectbox("Select a Visualization",("Wordcloud","Mendenhall Curve"))
						if visualization_kind == "Mendenhall Curve":
							#st.info("Mendenhall Curve")
							text_library.plot_mendenhall_curve(raw_text)

						else: #visualization_kind = "Wordcloud":
							#st.info("Wordcloud")
							text_library.plot_wordcloud(process_text)

				#End of Language Code Condition
					


			




	elif enter_choice == "Text Summarization":

		html_templ = """
		<div style="background-color:#080e4c;padding:10px;border-radius:5px">
		<h1 style="color:#3e47a5">Text Summarization</h1>
		</div>
		"""

		st.markdown(html_templ,unsafe_allow_html=True)
		st.write("A simple proposal for Text Summarization")

		placeholder = st.empty()
		placeholder.image(load_image('imgs/summary.png'), use_column_width=True)

		#File Uploading
		raw_text_file = st.sidebar.file_uploader("Upload File",type=['txt','docx','pdf'])
		raw_text = ""

		if raw_text_file is not None:
			file_details = {"Filename":raw_text_file.name,"FileType":raw_text_file.type,"FileSize":raw_text_file.size}
			#st.write(file_details)
			# Check File Type
			if raw_text_file.type == "text/plain":
				try:
					raw_text = text_library.get_txt(raw_text_file)
					placeholder.write("")
				except:
					st.write("TXT File Fetching Problem...")
			elif raw_text_file.type == "application/pdf":
				try:
					raw_text = text_library.get_pdf(raw_text_file)
					placeholder.write("")
				except:
					st.write("PDF File Fetching Problem...")
			elif raw_text_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
				try:
					raw_text = text_library.get_docx(raw_text_file)
					placeholder.write("")
				except:
					st.write("DOCX File Fetching Problem...")

			if raw_text != "":

				summary = ""

				language = language_detect(raw_text)
				language_code = language[0]
				#st.write(language_code)
				#st.write (language[1])


				if language_code != 'en':
					translator = google_translator()
					translate_text = translator.translate(raw_text,lang_tgt='en')  

					try:
						summary = summarize(translate_text)
						summary = translator.translate(summary,lang_tgt=language_code)
					except:
						st.warning("Input must have more than one sentence")

				else:
					try:
						summary = summarize(raw_text)
					except:
						st.warning("Input must have more than one sentence")

				if summary != "":
					st.info("Summary")
					st.write(summary)

				st.subheader("Original Text in "+language[1].capitalize())
				st.write(raw_text)






	elif enter_choice == "Text Translation":

		html_templ = """
		<div style="background-color:#080e4c;padding:10px;border-radius:5px">
		<h1 style="color:#3e47a5">Text Translation</h1>
		</div>
		"""

		st.markdown(html_templ,unsafe_allow_html=True)
		st.write("A simple proposal for Text Translation")

		placeholder = st.empty()
		placeholder.image(load_image('imgs/translation.jpeg'), use_column_width=True)

		#File Uploading
		raw_text_file = st.sidebar.file_uploader("Upload File",type=['txt','docx','pdf'])
		raw_text = ""

		if raw_text_file is not None:
			file_details = {"Filename":raw_text_file.name,"FileType":raw_text_file.type,"FileSize":raw_text_file.size}
			#st.write(file_details)
			# Check File Type
			if raw_text_file.type == "text/plain":
				try:
					raw_text = text_library.get_txt(raw_text_file)
					placeholder.write("")
				except:
					st.write("TXT File Fetching Problem...")
			elif raw_text_file.type == "application/pdf":
				try:
					raw_text = text_library.get_pdf(raw_text_file)
					placeholder.write("")
				except:
					st.write("PDF File Fetching Problem...")
			elif raw_text_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
				try:
					raw_text = text_library.get_docx(raw_text_file)
					placeholder.write("")
				except:
					st.write("DOCX File Fetching Problem...")

			if raw_text != "":

				#summary = ""

				language = language_detect(raw_text)
				language_code = language[0]
				#st.subheader("Original language: " + language[1].capitalize())

				lang = st.sidebar.selectbox("Select a language", list(languages.keys()))

				if lang != None:
					
					tran_lang = languages[lang]

					translator = google_translator()
					translate_text = translator.translate(raw_text,lang_tgt=tran_lang)

					st.info("Translation")
					st.write(translate_text)

					st.subheader("Original Text in "+language[1].capitalize())
					st.write(raw_text)









	elif enter_choice == "Sentiment from Text":

		html_templ = """
		<div style="background-color:#080e4c;padding:10px;border-radius:5px">
		<h1 style="color:#3e47a5">Sentiment Analysis from Text</h1>
		</div>
		"""

		st.markdown(html_templ,unsafe_allow_html=True)
		st.write("A simple proposal for Sentiment Analysis")

		placeholder = st.empty()
		placeholder.image(load_image('imgs/sentiment_small.jpg'), use_column_width=True)


	# ENDING TEXT PART ---------------------------------------------------------------------------------------------------


	# Menu "About"
	else:

		html_templ = """
		<div style="background-color:#080e4c;padding:10px;border-radius:5px">
		<h1 style="color:#3e47a5">About SETEM</h1>
		</div>
		"""

		st.markdown(html_templ,unsafe_allow_html=True)
		st.write("SpEech and TExt Management tool by Rosario Moscato LAB")

		stc.html(HTML_STICKER,width=700,height=800)










if __name__ == '__main__':
	main()