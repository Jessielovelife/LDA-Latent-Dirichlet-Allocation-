#!/usr/bin/python
#_*_coding:utf-8_*_

#@Time:  9th March, 2018
#@Author: xymlovelife
#@Function: extract topic terms from new test document

import os
import sys
import re
import warnings
import logging
import numpy as np
from collections import defaultdict
from gensim import corpora, models, similarities
from pprint import pprint
warnings.filterwarnings(action="ignore", category=UserWarning, module='gensim')
import logging

dellist = ['id', 'url', 'https', 'zh', 'wikipedia', 'curid', 'org', 'wiki', 'title']
def preprocesstxt(rawfile_path):
	"""
	preprocess txt docs to [[], [], .....]
	:param rawfile_path:
	:return:
	"""
	with open(rawfile_path, "rb") as f:
		strwords = f.read().decode('utf-8').split()
		print(len(strwords))
		print(strwords)
		for text in dellist:
			while text in strwords:
				strwords.remove(text)

	splitword = 'doc'
	numwords = [i for i, v in enumerate(strwords) if v == splitword]
	length = len(numwords)
	print(numwords)
	print(length)

	cnt = 0
	outlist = []
	for k in range(length // 2):
		temp = []
		for j in range(int(numwords[k * 2]), int(numwords[k * 2 + 1])):
			if len(strwords[j]) <= 1:
				pass
			else:
				temp.append(str(strwords[j]))
		# print(len(temp))
		cnt += len(temp)
		while 'doc' in temp:
			temp.remove('doc')
		outlist.append(temp)
	print(outlist)
	print(len(outlist))
	print(cnt)
	return outlist


def ldatest(ldamodel_path, dict_path, newdoc_path, topiclist_path):
	lda = models.LdaModel.load(ldamodel_path)
	dictionary = corpora.Dictionary.load(dict_path)
	#print(len(dictionary))
	ff = open(topiclist_path, 'w+', encoding='utf-8')
	newlist = preprocesstxt(newdoc_path)
	for i in range(len(newlist)):
		newbow = dictionary.doc2bow(newlist[i], allow_update=False, return_missing=False)
		#print(newbow)
		#print(len(newbow))
		topics = lda.get_document_topics(bow=newbow, per_word_topics=False)
		topics = sorted(topics, reverse=True, key=lambda x: x[1])
		print(topics)
		for topic in topics:
			ff.write(str(topic) + " ")
		ff.write('\n')
	ff.close()
	dictionary.compactify()
	print("Game Over.")



if __name__ == "__main__":
	logging.basicConfig(level=logging.DEBUG,
						format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
						datefmt='%a, %d %b %Y %H:%M:%S',
						filename='testlda.log',
						filemode='w')
	ldamdl_path = "D:\\wiki\\tools\\wiki_lda.model"
	dict_path = "D:\\wiki\\tools\\out\\rep_dict.dict"
	newdoc_path = "D:\\wiki\\tools\\out\\represents.txt"
	topic_path = "D:\\wiki\\tools\\out\\newtopic.txt"
	ldatest(ldamdl_path, dict_path, newdoc_path, topic_path)
	print("Test Done.")