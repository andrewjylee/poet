import os

class DataReader(object):
    def __init__(self, filename, char_level = True):
        self.filename = filename
        assert os.path.exists(self.filename), "Input File / Training Data File %s does not exist" % self.filename

        with open(self.filename) as f:
            data = f.read()

        if char_level:
            self.vocab = set(data)
            self.vocab_size = len(self.vocab)
            self.idx2vocab = dict(enumerate(self.vocab))
            self.vocab2idx = dict(zip(self.idx2vocab.values(), self.idx2vocab.keys()))

            self.data = [self.vocab2idx[char] for char in data]

        #TODO
        #else:
            #word level
    


