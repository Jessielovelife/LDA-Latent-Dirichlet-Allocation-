#!/usr/bin/python
#_*_coding:utf-8_*_

#@Time:  7th February, 2018
#@Author: xymlovelife
#@corpora from chinese wikipedia


import numpy as np
import jieba
import jieba.analyse
from collections import defaultdict
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim import corpora, models, similarities
from pprint import pprint


def stoplist(filepath):
	"""
    a function to load stopwords and save as a list
	:param filepath:
	:return:
	"""
	stopwords = [line.strip() for line in open(filepath, 'r', encoding="utf-8").readlines()]
	return stopwords

def segment(input0_path):
	"""
	split words from input corpora
	:return: words (str)
	"""
	f = open(input0_path, "rb")
	s = f.read()
	seg_list = jieba.cut(s, cut_all=False)
	words = " ".join(seg_list)
	return words


def delstopwords(stopword, word_corpora, output0_path):
	"""
	delete stopwords from split words and save as a opt.txt
	:return: outstr
	"""
	outstr = ""
	for w in word_corpora:
		if w not in stopword:
			outstr += str(w)

	with open(output0_path, 'a', encoding="utf-8") as f:
		f.write(outstr)
	return outstr


input0_path = "D:\\wiki\\extracted_0201\\AA\\test.txt"
output0_path = "D:\\wiki\\Round0\\output0.txt"
stopword_path = "D:\\wiki\\Round0\\stopwords_zh.txt"
stopwords = stoplist(stopword_path)
seg_list = segment(input0_path)
output0_list = delstopwords(stopwords, seg_list, output0_path)


def changelist(list_path):
	"""
	devide str elements to list elements [] --> [[],[],[]......]
	:return: list with list elements
	"""
	with open(list_path, "r", encoding="utf-8") as f:
		originlist = f.read().split()


	splitword = "doc"
	strlist = [i for i, v in enumerate(originlist) if v == splitword]
	wordlist = []
	for i in range(len(strlist)-1):
		temp = []
		for j in range(int(strlist[i]), int(strlist[i+1])):
			temp.append(originlist[j])
		wordlist.append(temp)
	with open("D:\\wiki\\Round0\\output1.txt", "w", encoding="utf-8") as f1:
		f1.write(str(wordlist))
	return wordlist


wordlist = changelist(output0_path)

def removeaword(output1_path, wdlist):
	"""
	remove single word in corpora
	:return: new corpora without single word
	"""
	frequency = defaultdict(int)
	for texts in wdlist:
		for text in texts:
			frequency[text] += 1

	wordlist = [[text for text in texts if frequency[text]>1]for texts in wdlist]
	#pprint(wordlist)

	dictionary = corpora.Dictionary(wordlist)
	dictionary.save(output1_path)
	#pprint(dictionary)
	print(dictionary.token2id)
	return dictionary

dictionary0 = removeaword("D:\\wiki\\Round0\\dictionay.dict", wordlist)


def ldamodel(dictionary0, wdlist, topicnum, num_of_doc, num_of_topic, newdoc_path=None):
	"""
	LDA for topic model
	:param dictionary0: existing corpora
	:param topicnum: demanding topic number
	:param wordlist: corpora list
	:param num_of_doc: doc number for show
	:param num_of_topic: topic number for show
	:return: None
	"""
	dictionary = dictionary0
	dict_len = len(dictionary)
	corpus = [dictionary.doc2bow(text) for text in wdlist]
	corpus_tfidf = models.TfidfModel(corpus)[corpus]

	lda = models.LdaModel(corpus_tfidf, num_topics=topicnum, id2word=dictionary,
						  alpha=0.1, eta=0.5, minimum_probability=0.001, update_every=1,
						  chunksize=100, passes=1)


	#update corpora
	if newdoc_path:
		new_list = changelist(newdoc_path)
		new_corpus = [dictionary.doc2bow(text) for text in wdlist]
		new_corpus_tfidf = models.TfidfModel(new_corpus)[new_corpus]
		lda.update(new_corpus)


	#lda.print_topics(num_of_doc)
	#lda.print_topic(num_of_topic)


	#print existing topic distribute of num_of_doc documents
	num_show_doc = num_of_doc
	print("show existing topic distribute of %d documents" % num_of_doc)
	doc_topics = lda.get_document_topics(corpus_tfidf)
	for i in range(num_show_doc):
		topic = np.array(doc_topics[i])
		topic_distribute = np.array(topic[:, 1])
		topic_index = list(topic_distribute)
		print("第%d个文档的%d个主题概率分别为: " % (i, num_show_doc))
		print(topic_index)


	# print shownum topic word and probability
	num_show_term = num_of_topic
	for topic_id in range(num_show_term):
		print("第%d个主题词与概率如下：\t" % topic_id)
		term_distribute_all = lda.get_term_topics(topicid = topic_id)
		term_distribute = term_distribute_all[:num_show_term]
		term_distribute = np.array(term_distribute)
		term_id = term_distribute[:, 0].astype(np.int)
		print("word:\t", end = " ")
		for t in term_id:
			print(dictionary.id2token[t], end = " ")
		print("\tprobability:\t", term_distribute[:,1])





