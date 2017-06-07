import numpy as np
import tensorflow as tf

import argparse
import os
from data import DataReader
from model import Model
from train import train




def setup_parser_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--train_filename', type=str, default='/home/andrew/poet/poet/data/training_data', help='Filename of training data')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='directory to save checkpoint models')
    parser.add_argument('--init_from', type=str, default=None, help='')

    parser.add_argument('--char_level', type=bool, default=False, help='1 - char-rnn, 0 - word-rnn')

    #parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--num_epochs', type=int, default=3, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.002, help='learning rate')
    parser.add_argument('--cell_type', type=str, default='lstm', help='type of cell to use for rnn. options: rnn, lstm, gru, ln_lstm')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--num_steps', type=int, default=20, help='Num steps for rnn. Input size.')
    parser.add_argument('--dropout_prob_input', type=float, default=0.9, help='Dropout keep probability for input')
    parser.add_argument('--dropout_prob_output', type=float, default=0.9, help='Dropout keep probability for output')
    #parser.add_argument('--state_size', type=int, default=256, help='Internal state size for each cell of rnn')
    parser.add_argument('--state_size', type=int, default=32, help='Internal state size for each cell of rnn')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of lyaers')

    parser.add_argument('--gpu_id', type=int, default=None, help='select gpu to use')
    parser.add_argument('--glove_file', type=str, default='/home/andrew/poet/poet/glove/glove.6B.100d.txt', help='glove pretrained embedding to use')
    parser.add_argument('--print_tensors', type=int, default=0, help='np.set_printoptions(). 1 = print tensors verbosely')


    args = parser.parse_args()

    args.train_filename = os.path.abspath(args.train_filename)
    if args.init_from is not None:
        args.init_from = os.path.abspath(args.init_from)

    if args.print_tensors:
        np.set_printoptions(threshold=np.nan)
    return args



if __name__=="__main__":
    # Set up Parser Args
    args = setup_parser_args()

    # Read in Training Data
    data = DataReader(args.train_filename, args.char_level)
    args.num_classes = data.vocab_size

    g = Model(args)

    losses = train(args, g, data)

    poem = g.write(data, pick_top_chars=5)
    print poem
