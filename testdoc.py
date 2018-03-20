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

def ldatest(ldamodel_path, dict_path, newdoc_path, topiclist_path):
	lda = models.LdaModel.load(ldamodel_path)
	with open(newdoc_path, 'r', encoding='utf-8') as f:
		newdoc = f.read().replace('\n', '').replace('\ufeff', '').split(',')

	print(newdoc)
	print(len(newdoc))
	dictionary = corpora.Dictionary.load(dict_path)
	print(type(dictionary))
	print(len(dictionary))
	newbow = dictionary.doc2bow(newdoc, allow_update=False, return_missing=False)
	print(newbow)
	print(dictionary.token2id)

	topics = lda.get_document_topics(bow=newbow, per_word_topics=False)
	dictionary.compactify()
	topics = sorted(topics, reverse=True, key=lambda x: x[1])
	ff = open(topiclist_path, 'w+', encoding='utf-8')
	for topic in topics:
		ff.write(str(topic)+',')
	ff.close()
	print(topics)
	"""
	#update
	corpora.MmCorpus.serialize("D:\\wiki\\tools\\out\\newbow.mm", newbow)
	mmnewbow = corpora.MmCorpus("D:\\wiki\\tools\\out\\newbow.mm")
	lda.update(mmnewbow)
	"""

	# lda update with mm file
	#mmcorpus = corpora.MmCorpus("D:\\wiki\\merge\\baike\\temp\\baike1.mm")
	#lda.update(mmcorpus)
"""
	#update dictionary
	newlist = []
	newlist.append(newdoc)

	#newdict = corpora.Dictionary()
	print("==========")
	print(len(dictionary))
	newdict = dictionary
	print("++++++++++++")
	print(len(newdict))
	newdict.add_documents(documents=newlist)
	print(len(newdict))
	print(newdict.token2id)
"""


if __name__ == "__main__":
	logging.basicConfig(level=logging.DEBUG,
						format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
						datefmt='%a, %d %b %Y %H:%M:%S',
						filename='testlda.log',
						filemode='w')
	ldamdl_path = "D:\\wiki\\tools\\wiki_lda.model"
	dict_path = "D:\\wiki\\tools\\out\\rep_dict.dict"
	newdoc_path = "D:\\wiki\\tools\\out\\newlist.txt"
	topic_path = "D:\\wiki\\tools\\out\\newtopic.txt"
	ldatest(ldamdl_path, dict_path, newdoc_path, topic_path)
	print("Done.")
