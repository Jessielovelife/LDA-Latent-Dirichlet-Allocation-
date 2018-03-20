#!/usr/bin/python
#_*_coding:utf-8_*_

#@Time:  1st March, 2018
#@Author: xymlovelife
#@Funtion: preprocess baike corpora and form a baike dictionary
#@Corpora: from baidu baike(/data/dongly1/baidumerge.txt)


import os
import time
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim import corpora, models, similarities
from collections import defaultdict
import itertools
import numpy as np
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
def process2dict(raw_input_path, baikedict_path):
	"""
	preprocess baike corpora into dictionary
	:param input_file_path: baidumerge.txt
	:param output_file_path: baike output path for dict
	:return: baike_dict
	"""
	frequency = defaultdict(int)
	word_list = []
	with open(raw_input_path, "r", encoding="utf-8") as input_file:
		for line in input_file:
			line_list = line.split()
			word_list.append(line_list)
		word_list = [[text for text in texts if len(text)>1] for texts in word_list]
		baike_dict = corpora.Dictionary(word_list)
		#baike_dict.filter_extremes(no_below=1, no_above=1, keep_n=None)
		baike_dict.save(baikedict_path)
		return word_list


def lda_of_baike(dict_file, wd_list, mm_path, topicNo, baike_topic_path):
	dict_len = len(dict_file)
	baike_corpus = [dict_file.doc2bow(word) for word in wd_list]
	corpora.MmCorpus.serialize(mm_path, baike_corpus)
	tfidModel = models.TfidfModel(baike_corpus)
	tfidf_vectors = tfidModel[baike_corpus]
	#lda = models.LdaModel(tfidf_vectors, id2word=dict_file, num_topics=topicNo)
	
	lda = models.LdaModel(tfidf_vectors, id2word=dict_file, num_topics=topicNo)
	lda.save("baike_lda.model")
	baike_topics = open(baike_topic_path, "a", encoding="utf-8")
	shown = lda.show_topics(num_topics = topicNo, num_words=dict_len)
	for text in shown:
		baike_topics.write(str(text)+"\n")
		print(len(str(text).split('+')))
	print(len(shown))

	print("LDA in baike is ok.")


"""
	#num_show_term = 15
	for topic_id in range(topicNo):
		term_distribute_all = lda.get_topic_terms(topicid=topic_id)
		#term_distribute = term_distribute_all[:num_show_term]
		term_distribute = term_distribute_all
		num_show_term = len(term_distribute_all)
		term_distribute = np.array(term_distribute)
		term_id = term_distribute[:, 0].astype(np.int)

		topic = []
		j = 0
		baike_topics = open(baike_topic_path, "a", encoding="utf-8")
		for t in term_id:
			topic.append(dict_file.id2token[t]+":"+str(term_distribute[j,1]))
			j += 1
		print(topic)
		topicstr = ",".join(topic)
		baike_topics.writelines(topicstr)
		baike_topics.writelines("\n")

	print("LDA in baike ok.")

			print(dict_file.id2token[t])
			print(type(dict_file.id2token[t]))
			print(dict_file.id2token[t]+":"+str(term_distribute[i,1]), end=" ")
			i += 1

		for t in term_id:
			print(dict_file.id2token[t], end=" ")
		print("\tprobability:\t", term_distribute[:, 1])
"""


if __name__ == "__main__":
	"""
	rawinput_path0 = "D:\\wiki\\merge\\baike\\temp\\baike0.txt"
	baike_dict_path0 = "D:\\wiki\\merge\\baike\\temp\\baike0.dict"
	templist0 = process2dict(rawinput_path0, baike_dict_path0)
	dict0 = corpora.Dictionary.load(baike_dict_path0)
	mm_path0 = "D:\\wiki\\merge\\baike\\temp\\baike0.mm"
	lda_of_baike(dict0, templist0, mm_path0, 4)
	print("baike0 is ok.")

	rawinput_path1 = "D:\\wiki\\merge\\baike\\temp\\baike1.txt"
	baike_dict_path1 = "D:\\wiki\\merge\\baike\\temp\\baike1.dict"
	templist1 = process2dict(rawinput_path1, baike_dict_path1)
	dict1 = corpora.Dictionary.load(baike_dict_path1)
	mm_path1 = "D:\\wiki\\merge\\baike\\temp\\baike1.mm"
	lda_of_baike(dict1, templist1, mm_path1, 4)
	print("baike1 is ok.")


	print("merge begins:")
	templist = templist0 + templist1
	dict2 = dict0
	dict2.merge_with(dict1)
	dict2.save("D:\\wiki\\merge\\baike\\temp\\dict2.dict")
	corpus0 = corpora.MmCorpus(mm_path0)
	corpus1 = corpora.MmCorpus(mm_path1)
	merge_corpus = itertools.chain(corpus0, corpus1)
	mm_path2 = "D:\\wiki\\merge\\baike\\temp\\baike2.mm"
	lda_of_baike(dict0, templist, mm_path2, 8)
	print("Game Over: baike done!")	
	"""

	start = time.time()
	rawinput_path = "/data/xymcorpora/xym/baike/baidumerge.txt"
	baike_dict_path = "/data/xymcorpora/xym/baike/baike_dict.dict"
	baikelist = process2dict(rawinput_path, baike_dict_path)
	baike_dict = corpora.Dictionary.load(baike_dict_path)
	baike_mm_path = "/data/xymcorpora/xym/baike/baike.mm"
	topicNum = 1000
	baike_topic_path = "/data/xymcorpora/xym/baike/baike_topics.txt"
	lda_of_baike(baike_dict, baikelist, baike_mm_path, topicNum, baike_topic_path)
	print("Game Over: baike done.")
	end = time.time()
	print("time: %s sec."%(end-start))
	"""
	rawinput_path = "D:\\wiki\\merge\\baike\\temp\\baike1.txt"
	baike_dict_path = "D:\\wiki\\merge\\baike\\temp\\baike1.dict"
	baikelist = process2dict(rawinput_path, baike_dict_path)
	baike_dict = corpora.Dictionary.load(baike_dict_path)
	baike_mm_path = "D:\\wiki\\merge\\baike\\temp\\baike1.mm"
	topicNum = 5
	baike_topic_path = "D:\\wiki\\merge\\baike\\temp\\baike1topic.txt"
	lda_of_baike(baike_dict, baikelist, baike_mm_path, topicNum, baike_topic_path)
	"""
