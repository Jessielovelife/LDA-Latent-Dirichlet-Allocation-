#!/usr/bin/python
#_*_coding:utf-8_*_

#@Time:  1st March, 2018
#@Author: xymlovelife
#@Function: merge baike and wiki, preprocess two corpora with LDA
#@Corpora: from baike and chinese wikipedia(146 contents// 150~300 topics)


import os
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim import corpora, models, similarities
from collections import defaultdict
import itertools
import numpy as np


def LDAmerge(corpus0_path, corpus1_path, dict0_path, dict1_path, mergeCorpus_mm_path, topicNo, topics_path):
	"""
	merge two corpus and process whole corpus with LDA
	:param corpus0_path: baike corpus path
	:param corpus1_path: wiki corpus path
	:param mergeCorpus_mm_path: whole corpus path ans save as .mm file
	:param topicNo: topic number from whole corpus
	:return: None
	"""
	# merge two corpus with mm file
	baike_corpus = corpora.MmCorpus(corpus0_path)
	wiki_corpus = corpora.MmCorpus(corpus1_path)
	merge_corpus = itertools.chain(baike_corpus, wiki_corpus)
	corpora.MmCorpus.serialize(mergeCorpus_mm_path, merge_corpus)

	# merge two corpus with dict file
	baike_dict = corpora.Dictionary.load(dict0_path)
	wiki_dict = corpora.Dictionary.load(dict1_path)
	merge_dict = baike_dict

	print(type(baike_dict))
	print(baike_dict)
	print(merge_dict)
	merge_dict.merge_with(wiki_dict)
	#merge_dict = merge_dict.merge_with(wiki_dict)

	# process whole corpus with LDA
	tfidModel = models.TfidfModel(merge_corpus)
	tfidf_vectors = tfidModel[merge_corpus]
	lda = models.LdaModel(tfidf_vectors, id2word=baike_dict, num_topics=topicNo)
	lda.save("merge_lad.model")

	# save and show topicNo topics as .txt file
	num_show_term = 20
	topics_file = open(topics_path, "w", encoding="utf-8")
	for topic_id in range(topicNo):
		term_distribute_all = lda.get_topic_terms(topicid=topic_id)
		term_distribute = term_distribute_all[:num_show_term]
		term_distribute = np.array(term_distribute)
		term_id = term_distribute[:, 0].astype(np.int)

		print("word:\t", end = " ")
		for t in term_id:
			print(baike_dict.id2token[t], end = " ")
		print("\tprobability:\t", term_distribute[:,1])

		#i = 0
		#for t in term_id:
			#print(merge_dict.id2token[t] + ":" + str(term_distribute[count, 1]), end=" ")
			#print(merge_dict.id2token[t] + ":" + str(term_distribute[i, 1]), end=" ")
			#target[topic_word] = term_distribute[count,1]
			#topic_word = merge_dict.id2token[t] + ":" + str(term_distribute[count, 1]).encode("utf-8")
			#print(topic_word)
			#i += 1
		#topic_line = topic_word + "\n"
		#topics_file.write(str(target).encode("utf-8"))

	print("MERGE DONE.")

if __name__ == "__main__":
	mm_path0 = "D:\\wiki\\merge\\baike\\temp\\baike0.mm"
	mm_path1 = "D:\\wiki\\merge\\baike\\temp\\baike1.mm"
	baike_dict_path0 = "D:\\wiki\\merge\\baike\\temp\\baike0.dict"
	baike_dict_path1 = "D:\\wiki\\merge\\baike\\temp\\baike1.dict"
	mm_path2 = "D:\\wiki\\merge\\baike\\temp\\baike2.mm"
	topicNo = 4
	topic_path = "D:\\wiki\\merge\\baike\\temp\\topics.txt"

	LDAmerge(mm_path0, mm_path1, baike_dict_path0, baike_dict_path1, mm_path2, topicNo, topic_path)
	print("********DONE*******")



"""
def lda_of_baike(dict_file, wd_list, mm_path, topicNo):
	dict_len = len(dict_file)
	baike_corpus = [dict_file.doc2bow(word) for word in wd_list]
	corpora.MmCorpus.serialize(mm_path, baike_corpus)
	tfidModel = models.TfidfModel(baike_corpus)
	tfidf_vectors = tfidModel[baike_corpus]
	lda = models.LdaModel(tfidf_vectors, id2word=dict_file, num_topics=topicNo)
	lda.save("baike_lad.model")
	print("in lda: ")
	for vector in baike_corpus:
		print(vector)
	lda.print_topics(topicNo)




#dict2 = process2dict("D:\\wiki\\baidu\\baidumerge.txt", "D:\\wiki\\baidu")
if __name__ == "__main__":
	
	rawinput_path0 = "D:\\wiki\\merge\\baike\\temp\\baike0.txt"
	baike_dict_path0 = "D:\\wiki\\merge\\baike\\temp\\baike0.dict"
	templist0 = process2dict(rawinput_path0, baike_dict_path0)
	dict0 = corpora.Dictionary.load(baike_dict_path0)
	mm_path0 = "D:\\wiki\\merge\\baike\\temp\\baike0.mm"
	lda_of_baike(dict0, templist0, mm_path0, 4, 8)
	print("baike0 is ok.")

	rawinput_path1 = "D:\\wiki\\merge\\baike\\temp\\baike1.txt"
	baike_dict_path1 = "D:\\wiki\\merge\\baike\\temp\\baike1.dict"
	templist1 = process2dict(rawinput_path1, baike_dict_path1)
	dict1 = corpora.Dictionary.load(baike_dict_path1)
	mm_path1 = "D:\\wiki\\merge\\baike\\temp\\baike1.mm"
	lda_of_baike(dict1, templist1, mm_path1, 4, 8)
	print("baike1 is ok.")


	print("merge begins:")
	templist = templist0 + templist1
	dict0.merge_with(dict1)
	corpus0 = corpora.MmCorpus(mm_path0)
	corpus1 = corpora.MmCorpus(mm_path1)
	merge_corpus = itertools.chain(corpus0, corpus1)
	mm_path2 = "D:\\wiki\\merge\\baike\\temp\\baike2.mm"
	lda_of_baike(dict0, templist, mm_path2, 8, 8)
	print("Game Over: baike done!")

"""