import numpy as np
import tensorflow as tf
import os

class Model(object):
    def __init__(self, args):
        self.args = args
        self.build_graph(args)



    def reset_graph(self):
        if 'sess' in globals() and sess:
            sess.close()
        tf.reset_default_graph()


    def build_graph(self, args):
        self.reset_graph()
        #self.saver = tf.train.Saver()

        self.x = tf.placeholder(tf.int32, [args.batch_size, args.num_steps], name='input_placeholder')
        self.y = tf.placeholder(tf.int32, [args.batch_size, args.num_steps], name='labels_placeholder')

        dropout_input = tf.constant(args.dropout_prob_input)
        dropout_output = tf.constant(args.dropout_prob_output)

        embedding = tf.get_variable('embedding_matrix', [args.num_classes, args.state_size])
        rnn_inputs = tf.nn.embedding_lookup(embedding, self.x)

        if args.cell_type == 'lstm' or args.cell_type == 'LSTM':
            cell = tf.contrib.rnn.BasicLSTMCell(args.state_size, state_is_tuple=True)
        elif args.cell_type == 'LN_LSTM' or args.cell_type == 'ln_lstm':
            cell = tf.contrib.rnn.LayerNormBasicLSTMCell(args.state_size)
        elif args.cell_type == 'GRU' or args.cell_type == 'gru':
            cell = tf.contrib.rnn.GRUCell(args.state_size)
        elif args.cell_type == 'rnn' or args.cell_type == 'RNN':
            cell = tf.contrib.rnn.core_rnn_cell.BasicRNNCell(args.state_size)
        else:
            assert False, "Could not find %s cell type" % args.cell_type

        if args.dropout_prob_input < 1.0:
            cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=dropout_input)

        cell = tf.contrib.rnn.MultiRNNCell([cell] * args.num_layers)

        if args.dropout_prob_output < 1.0:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout_output)
            
        self.init_state = cell.zero_state(args.batch_size, tf.float32)
        #TODO: see how others did this part.
        rnn_outputs, self.final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=self.init_state)
        #decoder_fn_train = tf.contrib.seq2seq.simple_decoder_fn_train(self.init_state)
        #rnn_outputs, self.final_state, final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(cell, decoder_fn_train, inputs=rnn_inputs, sequence_length=args.num_steps)

        with tf.variable_scope('softmax'):
            W = tf.get_variable('W', [args.state_size, args.num_classes])
            b = tf.get_variable('b', [args.num_classes])


        rnn_outputs = tf.reshape(rnn_outputs, [-1, args.state_size])
        y_reshaped = tf.reshape(self.y, [-1])

        logits = tf.matmul(rnn_outputs, W) + b
        #self.logits = tf.convert_to_tensor([tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]) 
        self.predictions = tf.nn.softmax(logits)
        loss_weights = tf.convert_to_tensor([tf.ones([args.batch_size]) for i in range(args.num_steps)])
        #self.losses = tf.contrib.seq2seq.sequence_loss(logits, self.y, loss_weights)
        #self.total_loss = tf.reduce_mean(self.losses)
        self.total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_reshaped, logits=logits))
        self.train_step = tf.train.AdamOptimizer(args.learning_rate).minimize(self.total_loss)

        #tf.summary.histogram('logits', logits)
        #tf.summary.histogram('loss', loss)
        #tf.summary.scalar('train_loss', self.
        
        #return dict(x = x, y = y, init_state = init_state, final_state = final_state, total_loss = total_loss, train_step = train_step, pred = predictions, saver = tf.train.Saver())
