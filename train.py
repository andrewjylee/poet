import numpy as np
import tensorflow as tf
import os
from six.moves import cPickle

from model import Model

#TODO: write main() here as well to be able to only train

def iterator(raw_data, batch_size, num_steps):
    """Iterate on the raw PTB data.
    This generates batch_size pointers into the raw PTB data, and allows
    minibatch iteration along these pointers.
    Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.
    Yields:
    Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
    The second element of the tuple is the same data time-shifted to the
    right by one.
    Raises:
    ValueError: if batch_size or num_steps are too high.
    """
    raw_data = np.array(raw_data, dtype=np.int32)
    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i*num_steps:(i+1)*num_steps]
        y = data[:, i*num_steps+1:(i+1)*num_steps+1]
        yield (x, y)

def gen_epochs(data, n, num_steps, batch_size):
    for i in range(n):
        yield iterator(data, batch_size, num_steps)



def train(args, model, data, ckpt=None, verbose=True):

    if args.init_from is not None:
        assert os.path.isdir(args.init_from), " %s directory not found." % args.init_from
        assert os.path.isfile(os.path.join(args.init_from, "config.pkl")), "config.pkl file not found in %s" % args.init_from

        print args.init_from
        ckpt = tf.train.get_checkpoint_state(args.init_from)
        assert ckpt, "No checkpoint found"
        assert ckpt.model_checkpoint_path, "No model path found in checkpoint"

        with open(os.path.join(args.init_from, 'config.pkl'), 'rb') as f:
            saved_model_args = cPickle.load(f)
        need_be_same = ["cell_type", 'state_size', 'num_layers', 'num_steps', 'train_filename']
        for check in need_be_same:
            assert vars(saved_model_args)[check] == vars(args)[check], "Command line argument and saved model disagree on '%s'" % check

        with open(os.path.join(args.init_from, 'data.pkl'), 'rb') as f:
            loaded_data = cPickle.load(f)
        assert loaded_data == data, "Data read in and data loaded does not match"
    
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        cPickle.dump(args, f)
    with open(os.path.join(args.save_dir, 'data.pkl'), 'wb') as f:
        cPickle.dump(data, f)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
            
        training_losses = []

        if ckpt is not None:
            saver.restore(sess, ckpt.model_checkpoint_path)

        for idx, epoch in enumerate(gen_epochs(data.data, args.num_epochs, args.num_steps, args.batch_size)):
            training_loss = 0
            steps = 0
            training_state = None
            for X, Y in epoch:
                steps += 1

                feed_dict = {model.x: X, model.y: Y}
                if training_state is not None:
                    feed_dict[model.init_state] = training_state
                training_loss_, training_state, _ = sess.run([model.total_loss, model.final_state, model.train_step], feed_dict)
                training_loss += training_loss_
            if verbose:
                print 'Avg training loss for Epoch', idx, ':', training_loss/steps
            training_losses.append(training_loss/steps)

            checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=idx)

    return training_losses
