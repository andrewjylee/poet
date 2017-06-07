import tensorflow as tf

import argparse
import os
from six.moves import cPickle

from model import Model



def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--init_from', type=str, default='checkpoints',
                        help='model directory to load/store checkpointed models')
    parser.add_argument('-n', type=int, default=1000,
                        help='number of characters to sample')
    parser.add_argument('--prime', type=str, default=u'The',
                        help='prime text')

    args = parser.parse_args()
    write(args)

def write(args):
    with open(os.path.join(args.init_from, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(args.init_from, 'data.pkl'), 'rb') as f:
        data = cPickle.load(f)

    model = Model(saved_args, training=False)
    poem = model.write(data, pick_top_chars=5)
    print poem
    return poem




if __name__ == '__main__':
    main()

