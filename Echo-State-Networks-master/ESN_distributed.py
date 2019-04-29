#!/usr/bin/env python
# coding: utf-8


import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops
from ESN import EchoStateRNNCell
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from time import time
import horovod.tensorflow as hvd

# Initialize Horovod
hvd.init()

mnist = tf.keras.datasets.mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

X_train, y_train = X_train[hvd.rank()::hvd.size()], y_train[hvd.rank()::hvd.size()]


print("MNIST shape", X_train.shape, X_test.shape)

if False:
    # debug only
    X_train = X_train[:10000]
    y_train = y_train[:10000]


# Pin GPU to be used to process local rank (one GPU per process)
# takes only current needed GPU memory
config = tf.ConfigProto(
      intra_op_parallelism_threads=hvd.size(),
      inter_op_parallelism_threads=hvd.size())
config.gpu_options.allow_growth = False
config.gpu_options.visible_device_list = str(hvd.local_rank())



# random numbers
random_seed = 1
# rng = np.random.RandomState(random_seed)



# flatten training shape
(N, Ni, Nj) = X_train.shape
(N_test, _, _) = X_test.shape

# x_train = x_train.reshape([N, Ni*Nj])
# x_test = x_test.reshape([x_test.shape[0], Ni*Nj])
if hvd.rank() == 0:
    print("MNIST shape", X_train.shape, X_test.shape)
    print("Size: {}".format(hvd.size()))


#plt.imshow(X_train[0].reshape([Ni,Nj]))
print(y_train[0])


# We define a simple simulation with a single batch. A timeseries of 'n_steps' timesteps is run. The input is a inpulse 
# given in the first timestep consisting of a gaussian noise given to each unit.


# Global variables

# hyperparameters
n_neurons = 256
learning_rate_ = 1e-3
batch_size = 128
n_epochs = 1 # use 4.0 for paper epochs <set epochs here>
n_epochs = int(n_epochs)
# n_epochs = int(np.ceil(n_epochs / hvd.size()))

# parameters
n_steps = 28 # 28 rows aka Ni
n_inputs = 28 # 28 cols aka Nj
n_outputs = 10 # 10 classes


rnn_inputs = np.zeros((batch_size, n_steps, n_inputs), dtype="float32")
rnn_inputs[:,:,:] = X_train[(0),:,:]
activation = lambda x: math_ops.tanh(x)


# Implementing a static graph without tensorflow API:

# Build model...
tf.reset_default_graph()
static_graph = tf.Graph()
with static_graph.as_default() as g:
    
    
    inputs = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    init_state = tf.placeholder(tf.float32, [None, n_neurons])
    
    y = tf.placeholder(tf.int32, [None])
    
    rng = np.random.RandomState(random_seed)

    # Init the ESN cell
    # rand_input = np.random.rand(1, n_neurons)
    zero_input = np.zeros([1, n_neurons])
    rnn_batch_init_state = np.broadcast_to(zero_input, [batch_size, n_neurons])
    rnn_test_init_state = np.broadcast_to(zero_input, [N_test, n_neurons])
    rnn_train_init_state = np.broadcast_to(zero_input, [N, n_neurons])
    cell = EchoStateRNNCell(num_units=n_neurons, 
                            num_inputs=n_inputs,
                            activation=activation, 
                            decay=0.01, 
                            epsilon=1e-10,
                            alpha=0.0100,
                            rng=rng)
    
    
    
    
    
    # Build the graph
    states = []
    state = init_state
    for t in range(n_steps):
        prev_state = state
        
        # LEARNING ERROR IS HERE
        # out, state = cell(inputs=inputs[0,t:(t+1),:], state=prev_state)
        #                               ^ error in batching
        out, state = cell(inputs= tf.reshape(inputs[:,t:(t+1),:], [-1, n_inputs]), state=prev_state)
#         print("out.shape", out.shape)
        # print("state.shape", state.shape)
        states.append(out)
    
    outputs = tf.convert_to_tensor(states)
    # print("outputs1.shape", outputs.shape)
    outputs = tf.transpose(outputs, [1,0,2])
    # print("outputs2.shape", outputs.shape)
    outputs = tf.reshape(outputs, [-1, n_steps * n_neurons]) #tf.reshape(tf.convert_to_tensor(states), [-1, n_steps * n_neurons])
    # print("outputs3.shape", outputs.shape)
    
    logits = tf.layers.dense(outputs, n_outputs)
    
    # RNN example
#     cell = tf.nn.rnn_cell.BasicRNNCell(num_units= n_neurons)
#     output, state = tf.nn.dynamic_rnn(cell, inputs, dtype= tf.float32)
    
#     print(state.shape)
    
#     logits = tf.layers.dense(state, n_outputs)

    
    
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(cross_entropy)
    # optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate_).minimize(loss)

    # horovard inclusion
    opt = tf.train.AdamOptimizer(learning_rate= learning_rate_* hvd.size())
    
    # Add Horovod Distributed Optimizer
    opt = hvd.DistributedOptimizer(opt)
    

    prediction = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
    

    global_step = tf.train.get_or_create_global_step()

    # Add hook to broadcast variables from rank 0 to all other processes during
    # initialization.
    hooks = [hvd.BroadcastGlobalVariablesHook(0)]

    # Make training operation
    train_op = opt.minimize(loss, global_step=tf.train.get_or_create_global_step())

    
    # Save checkpoints only on worker 0 to prevent other workers from corrupting them.
    checkpoint_dir = '/tmp/train_logs' if hvd.rank() == 0 else None
    
    # # initialize the variables
    init = tf.global_variables_initializer()
    

# Implementing a dynamic graph using tensorflow API


stime = time()
runtime = 0
losses = []
# train the model
with tf.Session(graph=static_graph) as sess:


# The MonitoredTrainingSession takes care of session initialization,
# restoring from a checkpoint, saving to a checkpoint, and closing when done
# or an error occurs.
#with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
#                                       config=config,
#                                       hooks=hooks) as sess:#mon_sess:
    
#    while not sess.should_stop():
        # Perform synchronous training.

    sess.run(init)
    n_batches = N // batch_size
#     print(n_batches)
    for epoch in range(n_epochs):
        batch_start = time()
        permutation = np.random.permutation(N)
#         print(permutation[0:1])
        for i, batch in enumerate(range(n_batches)):
            X_batch, y_batch = X_train[permutation[i:i+batch_size], :,:], y_train[permutation[i:i+batch_size]]
#             print(X_batch.shape, y_batch.shape)
#             print(X_batch.shape, y_batch.shape)
#             X_batch = X_batch.reshape([-1, n_steps, n_inputs])
            _, loss_train, acc_train = sess.run([train_op, loss, accuracy], 
                                                feed_dict={inputs: X_batch, y: y_batch, 
                                                           init_state: rnn_batch_init_state})
            losses.append(loss_train)
        
        loss_train, acc_train = sess.run([loss, accuracy], feed_dict={inputs: X_train[:1000], y: y_train[:1000], 
                                                                      init_state: rnn_train_init_state[:1000]})
        
        if hvd.rank() == 0:
            runtime += time() - batch_start
            print('Epoch {}/{}:\n\tTrain Loss: {:.10f}, Train Acc: {:.3f}'.format(
                epoch + 1, n_epochs, loss_train, acc_train))
            print("\tRuntime: {:.2f}s, ({:.2f} per epoch".format(runtime, (runtime)/(float(epoch)+1)))
    if hvd.rank() == 0:
        loss_test, acc_test = sess.run(
                [loss, accuracy], feed_dict={inputs: X_test, y: y_test, init_state: rnn_test_init_state})
if hvd.rank() == 0:
    print('\tTest Loss: {:.10f}, Test Acc: {:.3f}'.format(loss_test, acc_test))
    
    # print(losses)
    plt.plot(range(len(losses)), losses)
    plt.title("Loss over iterations (np= {}, rt= {:2f})".format(hvd.size(), runtime))
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.savefig("./plots/losses_itr.(np= {}, rt= {:2f}).png".format(hvd.size(), runtime))
    #plt.show()
