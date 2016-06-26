'''
spark-submit --num-executors 6 --executor-cores 1 imdb_random_indexing.py
'''
import sys
sys.path.insert(0, "../random_index")
from util import load_corpus, generate_vocabulary, remove_nonvocab_corpus
from pyspark import SparkContext


sc = SparkContext()
corpusRDD = load_corpus(sc, "/Users/jason/Desktop/spark_random_index/data/*.txt")
vocab = generate_vocabulary(sc, corpusRDD, min_count=50)
refinedCorpusRDD = remove_nonvocab_corpus(sc, corpusRDD, vocab)

