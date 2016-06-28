'''
spark-submit --conf spark.driver.memory=5g --num-executors 3 --executor-cores 1 imdb_random_indexing.py

if you get a heap error, increase executor memory =)
'''
import sys
sys.path.insert(0, "../random_index")
from util import load_corpus, generate_vocabulary, remove_nonvocab_corpus
from RandomIndexing import RandomIndexing
from pyspark import SparkContext
import numpy as np


sc = SparkContext()
# corpusRDD = load_corpus(sc, "/Users/jason/Desktop/spark_random_index/data/*.txt")
corpusRDD = load_corpus(sc, "/Users/jason.xie/Downloads/spark_random_index/data/*.txt")
vocab = generate_vocabulary(sc, corpusRDD, min_count=50)
refinedCorpusRDD = remove_nonvocab_corpus(sc, corpusRDD.sample(True, 0.00003), vocab)

ri = RandomIndexing(vocab)

# ri.load()
# print ri.embedding
# print ri.getMostSimilar("time", topn=50)

ri.generate_random_labels(sc)
ri.train(sc, refinedCorpusRDD.repartition(12))
print ri.embedding
print ri.getVector("time")
print ri.getMostSimilar("time", topn=50)
print ri.getMostSimilar("happy", topn=50)
ri.save()
