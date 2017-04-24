
# coding: utf-8

# In[4]:

import numpy as np
from math import ceil
import random
import tensorflow as tf
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as st
import pandas as pd
import pexpect
import time
slim = tf.contrib.slim
patches = mpl.patches


# In[ ]:

class NN(object):
    
    def __init__(self, sess, batch_iterators, num_layers = 1, num_hidden_nodes = 30, activation_fn = tf.nn.relu,
                 learning_rate = 0.00001, model_name = 'NN', target_scaling = True, feature_scaling = True,
                 checkpoint_dir = 'checkpoint', quantile = False):
        
        self.sess = sess
        
        self.train_iter = batch_iterators['train']
        self.val_iter = batch_iterators['val']
        self.test_iter = batch_iterators['test']

        self.num_layers = num_layers
        self.num_hidden_nodes = num_hidden_nodes
        self.activation_fn = activation_fn
        self.learning_rate = learning_rate
                
        self.targets_dim = self.train_iter.targets_dim
        self.features_dim = self.train_iter.features_dim
        
        self.target_scaling = target_scaling
        self.feature_scaling = feature_scaling
        
        self.quantile = quantile
        
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir
        
         # Scalers
        self.t_scaler, self.rev_t_scaler, self.rev_var_scaler = add_scaler(self.train_iter.target_mean,
                                    self.train_iter.target_std, scaling = self.target_scaling, with_var = True, name = 'targets')
        self.f_scaler, _ = add_scaler(self.train_iter.feature_mean, self.train_iter.feature_std, scaling = self.feature_scaling, name = 'features')
        
        self.build_model()
        
        self.saver = tf.train.Saver()
        
       
    def build_model(self):
        self.targets_pl = tf.placeholder(tf.float32, [None, self.targets_dim], 'targets_pl')
        self.features_pl = tf.placeholder(tf.float32, [None, self.features_dim], 'features_pl')
        
        # Scaling step
        self.targets = self.t_scaler(self.targets_pl)
        self.features = self.f_scaler(self.features_pl)
        
        # Build the mean prediction network
        hidden = slim.stack(self.features, slim.fully_connected, [self.num_hidden_nodes] * self.num_layers, 
                       activation_fn=self.activation_fn)
        
        self.output = slim.fully_connected(hidden, self.targets_dim, activation_fn=None, scope = 'final_layer')
        self.sc_output = self.rev_t_scaler(self.output)
        
        # Loss        
        if self.quantile:
            self.loss = quantile_loss(self.targets, self.output, self.quantile)
        else:
            self.loss = tf.reduce_mean(tf.pow(self.targets - self.output, 2))
                  
        # Optimizer
        self.opt = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        
    def train(self, viz_every = 500, num_steps = 5000):
        
        best_val_loss = float('inf')
        
        self.sess.run(tf.global_variables_initializer())
                
        for step in xrange(num_steps):
            
            t_batch, f_batch = self.train_iter.next_batch()
            
            # Initially just train the mean prediction network 
            _ = self.sess.run(self.opt, feed_dict = {self.features_pl: f_batch, self.targets_pl: t_batch})

            if step % viz_every == 0:

                _, TRAIN_LOSS = self.run_data_set(self.train_iter)
                _, VAL_LOSS = self.run_data_set(self.val_iter)
                _, TEST_LOSS = self.run_data_set(self.test_iter)

                print "Step: {0}, Train Loss: {1:.2f}, Val Loss: {2:.2f}, Test Loss: {3:.2f}".format(step,
                                                                    TRAIN_LOSS, VAL_LOSS, TEST_LOSS)            

    
                if VAL_LOSS < best_val_loss:
                    self.save()
                    best_val_loss = VAL_LOSS

        self.saver.restore(self.sess, self.checkpoint_dir + '/' + self.model_name)
        
        self.TRAIN_PREDS, TRAIN_LOSS = self.run_data_set(self.train_iter)
        self.VAL_PREDS, VAL_LOSS = self.run_data_set(self.val_iter)
        self.TEST_PREDS, TEST_LOSS = self.run_data_set(self.test_iter)
                
        print "Final Losses, Train: {1:.2f}, Val: {2:.2f}, Test: {3:.2f}".format(step,
                                                                            TRAIN_LOSS, VAL_LOSS, TEST_LOSS) 
                
    def run_data_set(self, iterator):
        
        # Store starting value of iterator to return to
        counter_start = iterator.counter
        # Make sure we start from the first batch
        iterator.counter = 0

        preds_list = []
        loss_list = []
        
        for step in xrange(iterator.num_batches):
            
            t_batch, f_batch = iterator.next_batch()
            PREDS, LOSS = self.sess.run([self.sc_output, self.loss], feed_dict = {self.features_pl: f_batch, self.targets_pl: t_batch})
            preds_list.append(PREDS)
            loss_list.append(LOSS)
            
        loss = np.average(loss_list)
        preds = np.concatenate(preds_list, axis = 0)

        # Return iterator counter to starting value
        iterator.counter = counter_start
        
        return preds, loss        
    
    def predict(self, features):
        
        self.saver.restore(self.sess, self.checkpoint_dir + '/' + self.model_name)
        
        PREDS = self.sess.run(self.sc_output, feed_dict = {self.features_pl: features})
        
        return PREDS

    def residuals(self):
        train_res = np.concatenate(self.train_iter.targets_data, axis = 0) - self.TRAIN_PREDS
        val_res = np.concatenate(self.val_iter.targets_data, axis = 0) - self.VAL_PREDS
        test_res = np.concatenate(self.test_iter.targets_data, axis = 0) - self.TEST_PREDS
        return train_res, val_res, test_res
   
    def save(self):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.saver.save(self.sess, self.checkpoint_dir + '/' + self.model_name)


# In[ ]:

def add_scaler(mean, std, scaling = False, with_var = False, name = 'Scaler'):
    
    mean = mean.astype(np.float32)
    std = std.astype(np.float32)
    
    if scaling:
    
        def scaler(data):
            scaled_data = tf.divide((data - mean), std)
            return scaled_data
        def reverse_scaler(data):
            reversed_data = tf.add(tf.multiply(data,std), mean, name = name + '_reverse')
            return reversed_data
        def reverse_var_scaler(data):
            reversed_data = tf.multiply(data, tf.pow(std,2))
            return reversed_data
        
    else:  
        def scaler(data):
            return data   
        def reverse_scaler(data):
            return data  
        def reverse_var_scaler(data):
            return data

    if with_var:
        return scaler, reverse_scaler, reverse_var_scaler
    else:
        return scaler, reverse_scaler


# In[ ]:

def read_UCI_data(loc, shuffle = True):
    data = np.array(pd.read_csv(loc, header = None))
    if shuffle:
        np.random.shuffle(data)
    targets = data[:,-1]
    features = data[:,0:-1]
    return targets, features


# In[ ]:

class monitor_gpu(object):
    
    def __init__(self):
        
        self.command = 'nvidia-smi --query-gpu=utilization.gpu --format=csv -l 1 -f ./temp_gpu_log.csv'
    
    def start_monitoring(self):
        self.p = pexpect.spawn(self.command)
        
    def stop_monitoring(self):
        self.p.sendcontrol('c')
        time.sleep(1)
        df = pd.read_csv('temp_gpu_log.csv', sep = ' ')
        self.usage = df['utilization.gpu'].iloc[2:-1]
        self.average_use = np.average(self.usage)
        
        os.remove('temp_gpu_log.csv')


# In[1]:

def single_blog_graph():
    fig = plt.figure(figsize = [6,4])
    ax = plt.axes()    
    mpl.rc('axes', labelsize = 12)
    mpl.rc('figure', titlesize = 14)
    return fig, ax

colours = {'orange': '#f78500', 'yellow': '#fed16c', 'green': '#139277', 'blue': '#0072df',
               'dark_blue': '#001e78', 'pink': '#fd6d77'}

def show_blog_colours():    
    num_colors = len(colours.keys())

    fig = plt.figure(figsize = [num_colors * 2, 2])
    ax = fig.add_subplot(111)

    for c in range(num_colors):
        color = colours[colours.keys()[c]]
        ax.add_patch(patches.Rectangle((0.03 + c*0.12 + c*0.03, 0.1), 0.12, 0.6, color = color))
        ax.annotate(colours.keys()[c], xy=(0.07 + 0.145 * c, 0.85))
    _ = plt.axis('off')


# In[ ]:

def batch_sorter(targets_data, features_data = None, batch_size = 100, shuffle_train = False, random_draw = False,
                train_ratio = 0.7, val_ratio = 0.15, test_ratio = 0.15, set_batch_size = False):

            
    assert train_ratio + val_ratio + test_ratio == 1, 'Percentages don\'t add up for the data sets'
    if features_data:
        assert len(targets_data) == len(features_data), 'Targets and features data different sizes'
        
    # Get rid of excess data if need a constant batch size
    if set_batch_size:
        if (len(targets_data)%batch_size) != 0:
            targets_data = targets_data[:-(len(targets_data)%batch_size)]
            if features_data:
                features_data = features_data[:len(targets_data)]

    # Split the data into train, val and test sets
    train_size = int(ceil(train_ratio * len(targets_data)))
    val_size = int(ceil(val_ratio * len(targets_data)))
    train_size = int(batch_size * round(float(train_size)/batch_size))
    val_size = int(batch_size * round(float(val_size)/batch_size))
    test_size = int(len(targets_data) - train_size - val_size)
    
    if random_draw:
        train_indices = random.sample(range(train_size + val_size), train_size)
        val_indices = [i for i in range(train_size + val_size) if i not in train_indices]
        train_t_data = [targets_data[i] for i in train_indices]
        val_t_data = [targets_data[i] for i in val_indices]
        if features_data:
            train_f_data = [features_data[i] for i in train_indices]
            val_f_data = [features_data[i] for i in val_indices]
    else:
        train_t_data = targets_data[0: train_size]
        val_t_data = targets_data[train_size: train_size + val_size]
        if features_data:
            train_f_data = features_data[0: train_size]
            val_f_data = features_data[train_size: train_size + val_size]
    
    test_t_data = targets_data[train_size + val_size :]
    if features_data:    
        test_f_data = features_data[train_size + val_size :]
    
    print "Train data: {} observations".format(len(train_t_data))
    print "Val data: {} observations".format(len(val_t_data))
    print "Test data: {} observations\n".format(len(test_t_data))


    # Shuffle train data if required
    if shuffle_train == True:

        indices = np.random.permutation(len(train_t_data))
        train_t_data = [train_t_data[i] for i in indices]
        
        if features_data:
            train_f_data = [train_f_data[i] for i in indices]    
    
    # Create the iterators
    if features_data:
        train_iter = batch_iterator(train_t_data, train_f_data, batch_size)
        val_iter = batch_iterator(val_t_data, val_f_data, batch_size)
        test_iter = batch_iterator(test_t_data, test_f_data, batch_size)
    else:
        train_iter = batch_iterator(train_t_data, batch_size = batch_size)
        val_iter = batch_iterator(val_t_data, batch_size = batch_size)
        test_iter = batch_iterator(test_t_data, batch_size = batch_size)
        

    iter_dict = {'train': train_iter, 'val': val_iter, 'test': test_iter}
    
    return iter_dict


# In[2]:

def bootstrap_batch_sorter(targets_data, features_data = None, batch_size = 100, shuffle = False, random_draw = False,
                train_ratio = 0.7, val_ratio = 0.15, test_ratio = 0.15, set_batch_size = False, num_bootstraps = 100):

    
    # features_data: should be shape [data_size * features_dim]
    # targets_data: should be shape [data_size * targets_dim]
    
    assert train_ratio + val_ratio + test_ratio == 1, 'Percentages don\'t add up for the data sets'
    assert targets_data.shape[0] == features_data.shape[0], 'Targets and features data different sizes'

    data_size = targets_data.shape[0]
       
    # Expand out to at least 2 dims if data is of shape (data_size,)
    if len(targets_data.shape) == 1:
        targets_data = np.expand_dims(targets_data, axis = 1)
    if len(features_data.shape) == 1:
        features_data = np.expand_dims(features_data, axis = 1)
    
    # Get rid of excess data if need a constant batch size
    if set_batch_size:
        if (data_size%batch_size) != 0:
            targets_data = targets_data[:-(data_size%batch_size), :]
            features_data = features_data[:len(targets_data), :]
                
    # Shuffle data if required - this will give a random test set
    if shuffle == True:
        indices = np.random.permutation(len(targets_data))
        targets_data = targets_data[indices, :]
        features_data = features_data[indices, :]

    # Split the data into train, val and test sets
    train_size = int(ceil(train_ratio * data_size))
    val_size = int(ceil(val_ratio * data_size))
    train_size = int(batch_size * round(float(train_size)/batch_size))
    val_size = int(batch_size * round(float(val_size)/batch_size))
    test_size = int(data_size - train_size - val_size)
        
    # Draw the bootstrap val and train sets
    train_t_list = []
    val_t_list = []
    
    train_f_list = []
    val_f_list = []
    
    for b in range(num_bootstraps):
    
        train_indices = random.sample(range(train_size + val_size), train_size)
        val_indices = [i for i in range(train_size + val_size) if i not in train_indices]
        train_t_list.append(targets_data[train_indices, :])
        val_t_list.append(targets_data[val_indices, :])
        train_f_list.append(features_data[train_indices, :])
        val_f_list.append(features_data[val_indices, :])  
    
    # Concatenate the boostraps together to give one matrix of size [num_bootstraps * train/val_size * num_features/targets]
    
    train_t_data = np.concatenate([np.expand_dims(b, axis = 0) for b in train_t_list], axis = 0)
    train_f_data = np.concatenate([np.expand_dims(b, axis = 0) for b in train_f_list], axis = 0)
    
    val_t_data = np.concatenate([np.expand_dims(b, axis = 0) for b in val_t_list], axis = 0)
    val_f_data = np.concatenate([np.expand_dims(b, axis = 0) for b in val_f_list], axis = 0)
    
    # Test data is the same for each bootstrap
    test_t_data = targets_data[train_size + val_size :, :]
    test_f_data = features_data[train_size + val_size :, :]
    
    test_t_data = np.tile(np.expand_dims(test_t_data, axis = 0), [num_bootstraps, 1, 1])
    test_f_data = np.tile(np.expand_dims(test_f_data, axis = 0), [num_bootstraps, 1, 1])

    print "Train data: {} observations".format(train_size)
    print "Val data: {} observations".format(val_size)
    print "Test data: {} observations\n".format(test_size)
    
    # Create the iterators
    train_iter = bootstrap_batch_iterator(train_t_data, train_f_data, batch_size)
    val_iter = bootstrap_batch_iterator(val_t_data, val_f_data, batch_size)
    test_iter = bootstrap_batch_iterator(test_t_data, test_f_data, batch_size)

    iter_dict = {'train': train_iter, 'val': val_iter, 'test': test_iter,} #'target_rescale': rev_target_scale}
                            
    return iter_dict


# In[3]:

class batch_iterator(object):
    
    def __init__(self, targets_data, features_data = None, batch_size = 50, shuffle = False):
               
        targets_data = self.sort_format(targets_data)
        self.num_data_points = len(targets_data)
        if features_data:
            features_data = self.sort_format(features_data)
        else:
            self.features_data = None
        
        if shuffle:
            indices = np.random.permutation(len(targets_data))
            targets_data = [targets_data[i] for i in indices]
            if features_data:
                features_data = [features_data[i] for i in indices]    

        self.targets_data = targets_data
        if features_data:
            self.features_data = features_data
        self.batch_size = batch_size
                
        self.counter = 0
        self.num_batches = int(ceil(float(len(targets_data))/float(batch_size)))
        
        self.targets_dim = targets_data[0].shape[1]
        if features_data:
            self.features_dim = features_data[0].shape[1]
        
        self.target_mean, self.target_std = self.target_scale_factors()
        if features_data:
            self.feature_mean, self.feature_std = self.feature_scale_factors()

    def next_batch(self):
        
        targets_batch = self.new_batch(self.counter, self.targets_data)
        if self.features_data:
            features_batch = self.new_batch(self.counter, self.features_data)
        
        self.counter += 1
        
        if self.counter == self.num_batches:
            self.counter = 0
        
        if self.features_data:
            return targets_batch, features_batch
        else:
            return targets_batch
        
    def new_batch(self, counter, data):
        try:
            new_batch = data[counter*self.batch_size: (counter*self.batch_size) + self.batch_size]
        except:
            new_batch = data[counter*self.batch_size: ]
            
        new_batch = np.concatenate(new_batch, axis = 0)
        
        return new_batch
    
    def sort_format(self, data):
        if type(data[0]) != np.ndarray:
            data = [np.array(t) for t in data]
        if len(data[0].shape) == 0:
            data = [np.expand_dims(t,0) for t in data]
        if len(data[0].shape) == 1:
            data = [np.expand_dims(t,0) for t in data]
        return data
    
    def target_scale_factors(self):
        joined = np.concatenate(self.targets_data, axis = 0)
        mean = np.average(self.targets_data, axis = 0)
        std = np.std(self.targets_data, axis = 0)
        return mean, std
    
    def feature_scale_factors(self):
        joined = np.concatenate(self.features_data, axis = 0)
        mean = np.average(self.features_data, axis = 0)
        std = np.std(self.features_data, axis = 0)
        return mean, std
    
    def sample_targets(self, sample_size):
        
        sample = random.sample(self.targets_data, sample_size)
        sample = np.concatenate(sample, axis = 0)
        return sample
    
    def sample_features(self, sample_size, index = 'random'):
        # Return the same set of features tiled, so can draw from distribution for the next prediction
        # Can define which features to return using the index input if required.
        if index == 'random':
            index = random.randint(0, len(self.features_data)-1)
            
        sample = self.features_data[index]
        sample = np.tile(sample, [sample_size, 1])
        
      
        return sample


# In[ ]:

class bootstrap_batch_iterator(object):
    
    def __init__(self, targets_data, features_data = None, batch_size = 50, shuffle = False):
        
        # Data will be of shape [num_bootstrap * data_size * targets/features_dim]
        
        self.num_data_points = targets_data.shape[1]
        self.targets_data = targets_data
        self.features_data = features_data
        self.batch_size = batch_size
                
        self.counter = 0
        self.num_batches = int(ceil(float(self.num_data_points/float(batch_size))))
        
        self.targets_dim = targets_data.shape[2]
        self.features_dim = features_data.shape[2]
        
        self.target_mean, self.target_std = self.scale_factors(targets_data)
        self.feature_mean, self.feature_std = self.scale_factors(features_data)

    
    def next_batch(self):
        
        targets_batch = self.new_batch(self.counter, self.targets_data)
        features_batch = self.new_batch(self.counter, self.features_data)
        
        self.counter += 1
        
        if self.counter == self.num_batches:
            self.counter = 0
            
        return targets_batch, features_batch
        
    def new_batch(self, counter, data):
        try:
            new_batch = data[:, counter*self.batch_size: (counter*self.batch_size) + self.batch_size, :]
        except:
            new_batch = data[:, counter*self.batch_size:, :]
                    
        return new_batch
    
    def scale_factors(self, data):
        mean = np.average(data[0,:,:], axis = 0)
        std = np.std(data[0,:,:], axis = 0)
        return mean, std

