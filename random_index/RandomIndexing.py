import random
import numpy as np
from pyspark import SparkContext
from pyspark import StorageLevel
import random
from scipy.spatial.distance import cosine
import pickle


class RandomIndexing:
	def __init__(self, vocab, dimension=1800, nonsparse_avg=9, nonsparse_sd=2.7):
		self.vocab = vocab
		self.dimension = dimension
		self.nonsparse_avg = nonsparse_avg
		self.nonsparse_sd = nonsparse_sd

	def generate_random_labels(self, sc):
		dim = self.dimension
		nonsparse_avg = self.nonsparse_avg
		nonsparse_sd = self.nonsparse_sd
		def make_a_label(word):
			num_indexes = -1
			while num_indexes < 4:
				num_indexes = int(random.gauss(nonsparse_avg, nonsparse_sd))
			indexes = np.random.choice(dim, num_indexes)
			word_arr = np.zeros((dim,), dtype=np.float32)
			for index in indexes:
				val = 1
				if random.random() < 0.5:
					val = -1
				word_arr[index] = val
			return (word, word_arr)
		self.vocab_sc = sc.broadcast(self.vocab)
		self.random_labels = sc.parallelize(self.vocab).map(make_a_label).collectAsMap()
		self.random_labels_sc = sc.broadcast(self.random_labels)

	def train(self, sc, corpusRDD):
		print "starting training"
		self.embedding = np.zeros((len(self.vocab), self.dimension), dtype=np.float32)
		self.embedding_sc = sc.broadcast(self.embedding)
		embedding_sc = self.embedding_sc
		random_labels_sc = self.random_labels_sc
		vocab_sc = self.vocab_sc
		def train_partition(partition_data):
			local_vocab = vocab_sc.value
			local_embedding = embedding_sc.value.copy()
			local_random_labels = random_labels_sc.value
			count = 0
			for sentence in partition_data:
				count += 1
				if count % 10 == 0:
					print count
				for word_i in sentence:
					word_embedding = local_embedding[local_vocab.index(word_i)]
					for word_j in sentence:
						if word_i == word_j:
							continue
						word_embedding += local_random_labels[word_j]
			return [local_embedding]
		embeddingRDD = corpusRDD.mapPartitions(train_partition)
		self.embedding = embeddingRDD.reduce(lambda a, b: a + b)

	def getVector(self, word):
		return self.embedding[self.vocab.index(word)]

	def getMostSimilar(self, word, topn=10):
		score_dict = {}
		if type(word) is str:
			vector = self.getVector(word)
		else:
			vector = word
		for curr_word in self.vocab:
			dist = cosine(vector, self.getVector(curr_word))
			if np.isnan(dist):
				continue
			simscore = 1 - dist
			score_dict[curr_word] = simscore
		sorted_list = sorted(score_dict.items(), key=lambda kv: kv[1], reverse=True)
		return sorted_list[:topn]

	def save(self, fileName="/tmp/ri.pkl"):
		with open(fileName, 'wb') as output:
			self.vocab_sc = None 
			self.random_labels_sc = None
			self.embedding_sc = None
			pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

	def load(self, fileName="/tmp/ri.pkl"):
		with open(fileName, 'r') as tmp_input:
			obj = pickle.load(tmp_input)
			self.vocab = obj.vocab
			self.dimension = obj.dimension
			self.nonsparse_avg = obj.nonsparse_avg
			self.nonsparse_sd = obj.nonsparse_sd
			self.embedding = obj.embedding
			self.random_labels = obj.random_labels
