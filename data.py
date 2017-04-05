import os
import collections

class DataReader(object):
    def __init__(self, filename, char_level):
        self.filename = filename
        self.char_level = char_level
        assert os.path.exists(self.filename), "Input File / Training Data File %s does not exist" % self.filename

        self.read_data()
    
    def read_data(self):
        with open(self.filename) as f:
            data = f.read()
        if self.char_level:
            self.vocab = set(data)
            self.vocab_size = len(self.vocab)
            self.idx2vocab = dict(enumerate(self.vocab))
            self.vocab2idx = dict(zip(self.idx2vocab.values(), self.idx2vocab.keys()))
            self.data = [self.vocab2idx[char] for char in data]

        #word level
        else:
            data = data.split()
            self.vocab = set(data)
            self.vocab_size = len(self.vocab)
            self.idx2vocab = dict(enumerate(self.vocab))
            self.vocab2idx = dict(zip(self.idx2vocab.values(), self.idx2vocab.keys()))

            self.data = [self.vocab2idx[word] for word in data]


