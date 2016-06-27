'''
spark-submit --num-executors 6 --executor-cores 1 imdb_random_indexing.py
'''
import sys
sys.path.insert(0, "../random_index")
from util import load_corpus, generate_vocabulary, remove_nonvocab_corpus
from RandomIndexing import RandomIndexing
from pyspark import SparkContext


sc = SparkContext()
corpusRDD = load_corpus(sc, "/Users/jason/Desktop/spark_random_index/data/*.txt")
vocab = generate_vocabulary(sc, corpusRDD, min_count=50)
refinedCorpusRDD = remove_nonvocab_corpus(sc, corpusRDD.sample(True, 0.002), vocab)

ri = RandomIndexing(vocab)
ri.generate_random_labels(sc)
ri.train(sc, refinedCorpusRDD)
print ri.embedding
print ri.embedding[np.nonzero[ri.embedding]]
