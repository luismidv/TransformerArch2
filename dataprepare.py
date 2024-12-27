import torch.utils.data
import torch
import pandas as pd
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import PunktTokenizer
import re


import ssl
import certifi

#ssl._create_default_https_context = ssl.create_default_context(cafile=certifi.where())

class Datapreparer():
    def __init__(self, path):
        self.path = path
        self.all_sentence = []
        self.all_words = []
        self.stemmer = PorterStemmer()
        self.tokenizer = PunktTokenizer()
        self.csv_loader()

    def csv_loader(self):
        dataset_file = pd.read_csv(self.path)
        self.features = dataset_file['review']
        self.labels = dataset_file['rating']
        print(self.features)


    def feature_iterator(self):
        for feature in self.features:
            self.all_sentence.append(feature)
            for word in feature.split(" "):
                if word not in self.all_words:
                    self.all_words.append(word)


    def pre_process_text(self):
        words_to_ignore = ['?', '!', '.', ',']
        new_all_words = [self.stemmer.stem(word) for word in self.all_words if word not in words_to_ignore]
        self.all_words = new_all_words

    def word_tokenize(self):
        #tokenized_words = [nltk.word_tokenize(word) for word in self.all_words]


    def feature_embedding(self,feature, vocab_size, model_size):
        embed = torch.nn.Embedding(vocab_size, model_size)
        feature_embed = embed(feature)
        return feature_embed

model_size = 512
path = './data/data.csv'
dataprep = Datapreparer(path)
vocab_size = dataprep.feature_iterator()
processed_feature = dataprep.pre_process_text()
#TODO TOKENIZE EACH WORD AND MAKE THE EMEBEDDING
dataprep.word_tokenize()
#feature_embedding = feature_embedding(feature, vocab_size, model_size)
#print(f"Feature embedded \n {feature_embedding}")
