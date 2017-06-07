import numpy as np
import tensorflow as tf
import os

class Model(object):
    def __init__(self, args, training=True):
        self.args = args
        self.build_graph(training)

    def reset_graph(self):
        if 'sess' in globals() and sess:
            sess.close()
        tf.reset_default_graph()


    def build_graph(self, training):
        self.reset_graph()
        args = self.args

        if not training:
            args.batch_size = 1
            args.num_steps = 1

        self.x = tf.placeholder(tf.int32, [args.batch_size, args.num_steps], name='input_placeholder')
        self.y = tf.placeholder(tf.int32, [args.batch_size, args.num_steps], name='labels_placeholder')

        dropout_input = tf.constant(args.dropout_prob_input)
        dropout_output = tf.constant(args.dropout_prob_output)

        # TODO:
        # fix data.py accordingly
        #with tf.variable_scope('embedding'):
        #    self.embedding = tf.Variable(tf.constant(0.0, shape=[args.vocab_size+1, embedding_dim]),
        #            trainable=True,
        #            name='embedding')
        #    self.embedding_placeholder = tf.placeholder(tf.float32, [args.vocab_size+1, embedding_dim])
        #    self.embedding_init = self.embedding.assign(self.embedding_placeholder)
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

        if training and args.dropout_prob_input < 1.0:
            cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=dropout_input)

        cell = tf.contrib.rnn.MultiRNNCell([cell] * args.num_layers)

        if training and args.dropout_prob_output < 1.0:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout_output)
            
        self.cell = cell
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
        
        #return dict(x = x, y = y, init_state = init_state, final_state = final_state, total_loss = total_loss, train_step = train_step, pred = predictions) 


    def write(self, data, prompt='The ', poem_length = 1000, pick_top_chars=5):
        self.build_graph(training=False)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(self.args.save_dir)
            self.saver.restore(sess, ckpt.model_checkpoint_path)

            state = None
            print [i for i in data.vocab2idx.keys()]
            chars = [data.vocab2idx[i] for i in prompt]
            current_char = chars[-1]

            for i in range(poem_length):
                if state is not None:
                    feed_dict = {self.x: [[current_char]], self.init_state: state}
                else:
                    feed_dict = {self.x: [[current_char]]}

                preds, state = sess.run([self.predictions, self.final_state], feed_dict)

                p = np.squeeze(preds)
                if pick_top_chars is not None:
                    p[np.argsort(p)[:-pick_top_chars]] = 0
                    p = p / np.sum(p)
                current_char = np.random.choice(data.vocab_size, 1, p=p)[0]

                chars.append(current_char)

            chars = map(lambda x: data.idx2vocab[x], chars)
            return("".join(chars))

