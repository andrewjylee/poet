import numpy as np
import tensorflow as tf

from model import Model

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



def train(args, model, data, verbose=True, save=False):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        training_losses = []

        for idx, epoch in enumerate(gen_epochs(data, args.num_epochs, args.num_steps, args.batch_size)):
            training_loss = 0
            steps = 0
            training_state = None
            for X, Y in epoch:
                steps += 1

                #feed_dict = {g['x']: X, g['y']: Y}
                feed_dict = {model.x: X, model.y: Y}
                if training_state is not None:
                    #feed_dict[g['init_state']] = training_state
                    feed_dict[model.init_state] = training_state
                #training_loss_, training_state, _ = sess.run([g['total_loss'], g['final_state'], g['train_step']], feed_dict)
                training_loss_, training_state, _ = sess.run([model.total_loss, model.final_state, model.train_step], feed_dict)
                training_loss += training_loss_
            if verbose:
                print 'Avg training loss for Epoch', idx, ':', training_loss/steps
            training_losses.append(training_loss/steps)

        if isinstance(save, str):
            #g['saver'].save(sess, save)
            model.saver.save(sess, save)

    return training_losses
