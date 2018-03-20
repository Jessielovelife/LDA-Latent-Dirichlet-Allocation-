#!/usr/bin/python
#_*_coding:utf-8_*_

#@Time:  27th February, 2018/ modify on 1st March
#@Author: xymlovelife
#@Function: preprocess wiki and form a dictionary
#@Corpora: from chinese wikipedia(146 contents// 150~300 topics)


import os
import time
import numpy as np
import re
import jieba
import logging
from collections import defaultdict
import warnings
from gensim import corpora, models, similarities
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
def stoplist(stopwords_path):
	"""
    load stopwords.txt and save as a list
	:param filepath:
	:return: stopwords(list)
	"""
	stopwords = [line.strip() for line in open(stopwords_path, 'rb').readlines()]
	return stopwords


def segment(raw_input_path):
	"""
	split words from input corpora
	:return: words (str)
	"""
	f = open(raw_input_path, "rb")  # "rb"
	s = f.read()
	seg_list = jieba.cut(s, cut_all=False)
	words = " ".join(seg_list)
	return words


def delstopwords(stop_words, words_corpora, outstr_path):
	"""
	delete stopwords from split words and save as a wordstr.txt
	:return: outstr (str)  & wordstr.txt
	"""
	outstr = ""
	for w in words_corpora:
		if w not in stop_words:
			outstr += str(w)
			#outstr += (w).encode('utf-8')

	with open(outstr_path, 'w+', encoding="utf-8") as f:
		f.write(outstr)
	return outstr


def changelist(outstr_path, outlist_path):
	"""
	devide str elements to list elements [] --> [[],[],[]......]
	:return: list with list elements
	"""
	removelist = ['id', 'url', 'https', 'zh', 'wikipedia', 'org', 'curid', 'wiki', 'title']
	with open(outstr_path, "r", encoding="utf-8") as f:
		originlist = f.read().split()
		for text in removelist:
			while text in originlist:
				originlist.remove(text)

	splitword = "doc"
	numwords = [i for i, v in enumerate(originlist) if v == splitword]
	lenwords = len(numwords) // 2
	wordlist = []
	for i in range(lenwords):
		temp = []
		for j in range(int(numwords[i*2]), int(numwords[i*2+1])):
			if len(originlist[j])<=1:
				pass
			else:
				temp.append(originlist[j])
		while 'doc' in temp:
			temp.remove('doc')
		wordlist.append(temp)
	with open(outlist_path, "w+", encoding="utf-8") as flist:
		flist.write(str(wordlist))
	return wordlist


def removeaword(wiki_dict_path, wdlist):
	"""
	remove single word in corpora
	:return: new corpora without single word
	"""
	frequency = defaultdict(int)
	for texts in wdlist:
		for text in texts:
			frequency[text] += 1

	wordlist = [[text for text in texts if frequency[text]>=1]for texts in wdlist]
	wordlist = [[text for text in texts if len(text)>1] for texts in wordlist]
	#pprint(wordlist)
	#wordlist.remove('[]')
	wiki_dict = corpora.Dictionary(wordlist)
	wiki_dict.save(wiki_dict_path)
	#pprint(wiki_dict)
	print(wiki_dict.token2id)
	return wiki_dict



def lda_of_wikipedia(dict_file, wd_list, mm_path, topicNo, wiki_topic_path):
	dict_len = len(dict_file)
	wiki_corpus = [dict_file.doc2bow(word) for word in wd_list]
	corpora.MmCorpus.serialize(mm_path, wiki_corpus)
	tfidModel = models.TfidfModel(wiki_corpus)
	tfidf_vectors = tfidModel[wiki_corpus]
	lda = models.LdaModel(tfidf_vectors, id2word=dict_file, num_topics=topicNo,alpha=0.1, eta='auto')
	lda.save("wiki_lda.model")	
	baike_topics = open(wiki_topic_path, "a", encoding="utf-8")
	shown = lda.show_topics(num_topics = topicNo, num_words=dict_len)
	for text in shown:
		baike_topics.write(str(text)+"\n")
		print(len(str(text).split('+')))
	print(len(shown))
	
	print("LDA in wikipedia is ok.")
"""
	#num_show_term = 30
	for topic_id in range(topicNo):
		term_distribute_all = lda.get_topic_terms(topicid=topic_id)
		#term_distribute = term_distribute_all[:num_show_term]
		term_distribute = term_distribute_all
		term_distribute = np.array(term_distribute)
		term_id = term_distribute[:, 0].astype(np.int)

		topic = []
		j = 0
		baike_topics = open(wiki_topic_path, "a", encoding="utf-8")
		for t in term_id:
			topic.append(dict_file.id2token[t]+":"+str(term_distribute[j,1]))
			j += 1
		print(topic)
		topicstr = ",".join(topic)
		baike_topics.writelines(topicstr)
		baike_topics.writelines("\n")

	print("LDA in wikipedia ok.")

"""


if __name__ == "__main__":
	start = time.time()
	stopword_path = "/data/xymcorpora/xym/wiki/stopwords_zh.txt"
	stopwords = stoplist(stopword_path)
	rawinput_path = "/data/xymcorpora/xym/wiki/chinese"
	wordstr_path = "/data/xymcorpora/xym/wiki/output/wordstr.txt"
	wordlist_path = "/data/xymcorpora/xym/wiki/output/wordlist.txt"
	wikidict_path = "/data/xymcorpora/xym/wiki/output/wiki_dict.dict"
	files = os.listdir(rawinput_path)
	segment_str = ""
	for file in files:
		if not os.path.isdir(file):
			ffile = rawinput_path + "/" + file
			segment_str += segment(ffile)

	wordstr = delstopwords(stopwords, segment_str, wordstr_path)
	wordlist = changelist(wordstr_path, wordlist_path)
	wiki_dict = removeaword(wikidict_path, wordlist)
	whole_list = wordlist
	wiki_mm_path = "/data/xymcorpora/xym/wiki/output/wiki_mm.mm"
	wiki_topic_path = "/data/xymcorpora/xym/wiki/wiki_topics.txt"
	lda_of_wikipedia(wiki_dict, whole_list, wiki_mm_path, 500, wiki_topic_path)
	print("Game Over: wikipedia done!")
	end = time.time()
	print("time: %s sec."%(end-start))

