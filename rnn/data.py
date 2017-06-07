import os
import collections
import nltk
import codecs

class DataReader(object):
    def __init__(self, filename, char_level=False):
        self.filename = filename
        self.char_level = char_level
        assert os.path.exists(self.filename), "Input File / Training Data File %s does not exist" % self.filename

        self.read_data()
    
    def read_data(self):
        #with open(self.filename) as f:
        with codecs.open(self.filename, "r", encoding='utf-8') as f:
            data = f.read()

        if self.char_level is False:
            # word level
#            data = data.split()
            data = [i for i in nltk.word_tokenize(data)]

            #data = [i.decode('utf-8') for i in data.split()]
        data.append(' ')

        self.vocab = set(data)
        self.vocab_size = len(self.vocab)
        self.idx2vocab = dict(enumerate(self.vocab))
        self.vocab2idx = dict(zip(self.idx2vocab.values(), self.idx2vocab.keys()))
        self.data = [self.vocab2idx[char] for char in data]


if __name__=="__main__":
    data = DataReader('/home/andrew/poet/poet/data/training_data')
    data.read_data()
    test = data.vocab2idx.keys()
    if isinstance(test[0], str):
        print 'string'
    elif isinstance(test[0], unicode):
        print 'unicode'


