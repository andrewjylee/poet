import numpy as np
import tensorflow as tf

import argparse
from data import DataReader
from model import Model
from train import train




def setup_parser_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--train_filename', type=str, default='data/tinyshakespeare', help='Filename of training data')
    parser.add_argument('--num_epochs', type=int, default=3, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.002, help='learning rate')

    parser.add_argument('--cell_type', type=str, default='rnn', help='type of cell to use for rnn. options: rnn, lstm, gru, ln_lstm')

    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')

    parser.add_argument('--num_steps', type=int, default=200, help='Num steps for rnn. Input size.')

    parser.add_argument('--dropout_prob_input', type=float, default=1.0, help='Dropout keep probability for input')
    parser.add_argument('--dropout_prob_output', type=float, default=1.0, help='Dropout keep probability for output')

    parser.add_argument('--state_size', type=int, default=100, help='Internal state size for each cell of rnn')

    parser.add_argument('--num_layers', type=int, default=1, help='Number of lyaers')

    parser.add_argument('--print_tensors', type=int, default=0, help='np.set_printoptions(). 1 = print tensors verbosely')


    args = parser.parse_args()

    if args.print_tensors:
        np.set_printoptions(threshold=np.nan)
    return args





if __name__=="__main__":
    # Set up Parser Args
    args = setup_parser_args()

    # Read in Training Data
    data = DataReader(args.train_filename)
    args.num_classes = data.vocab_size

    g = Model(args)
    losses = train(args, g, data.data)

    
