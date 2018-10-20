import pandas as pd
from datetime import datetime
import numpy as np	
import tensorflow as tf

from matplotlib import pyplot


#first load and parse the data
data = pd.read_csv("spread_features.csv", index_col=0, parse_dates=True)

s = data["spread"]
X = data[["1", "2", "3", "4", "5"]]

# spread with a two day lag if you wish to use timeseries as input (ie RNN, CNN, ARIMA, etc...)
shifted_s = s.shift(freq="48H")

split = datetime(2017,8,1)

X_train = X.loc[:split]
X_validate = X.loc[split:]

s_train = s.loc[:split]
s_validate = s.loc[split:]

####
def train_and_predict(X_train, s_train, X_test):
    signal_length = 48
    num_components = 5

    num_hidden = 32
    learning_rate = 0.005
    lambda_loss = 0.1
    total_steps = 20000
    display_step = 200
    batch_size = 20
    nnode_l = 60
    
    graph = tf.Graph()
    
    #model definition
    with graph.as_default():

        X_dataset = tf.placeholder(tf.float32, shape=(None, signal_length, num_components))
        s_dataset = tf.placeholder(tf.float32, shape = (None, 1))

        #use a double layer RNN
        splitted_data = tf.unstack(X_dataset, axis=1)
        cell1 = tf.nn.rnn_cell.BasicLSTMCell(num_hidden, forget_bias=1.0, state_is_tuple=True)  
        cell2 = tf.nn.rnn_cell.BasicLSTMCell(num_hidden, forget_bias=1.0, state_is_tuple=True)
        
        #add dropout layer with 0.3 dropout prob
        cell1_dropout = tf.contrib.rnn.DropoutWrapper(cell1,input_keep_prob=0.7, output_keep_prob=0.7)    
        cell2_dropout = tf.contrib.rnn.DropoutWrapper(cell2,input_keep_prob=0.7, output_keep_prob=0.7)
        
        cell_dropout = tf.nn.rnn_cell.MultiRNNCell([cell1_dropout, cell2_dropout], state_is_tuple=True)
        outputs, current_state = tf.nn.static_rnn(cell_dropout, splitted_data, dtype=tf.float32)
    
        w = tf.Variable(tf.random_normal([num_hidden, nnode_l], stddev=0.1))
        b = tf.Variable(tf.random_normal([nnode_l], stddev=0.1))
        
        #w = tf.Variable(tf.zeros([num_hidden, nnode_l]))
        #b = tf.Variable(tf.zeros([nnode_l]))
        
        wl = tf.Variable(tf.random_normal([nnode_l, 1], stddev=0.1))
        bl = tf.Variable(tf.random_normal([1], stddev=0.1))
        
        #wl = tf.Variable(tf.zeros([nnode_l, 1]))
        #bl = tf.Variable(tf.zeros([1]))
        
        wouts = [tf.sigmoid(tf.matmul(output, w) + b) for output in outputs]
    
        f_X = [(tf.matmul(wout, wl)) + bl for wout in wouts]
        f_X = tf.reshape(tf.convert_to_tensor(f_X), [-1, 1])
        
        output = outputs[-1]
        wout = tf.sigmoid(tf.matmul(output, w) + b)
        f_X_less = tf.matmul(wout, wl) + bl
        s_dataset_split = s_dataset[::signal_length,:]
        #l2 regulation term
        l2 = lambda_loss * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
    
        #maximize profit
        #profit = tf.reduce_sum(tf.multiply(f_X, s_dataset))
        profit = tf.reduce_sum(tf.square(f_X_less - s_dataset_split))
        #add a penalty that quickly blows up when 
        penalty = tf.exp(-(tf.reduce_min(tf.multiply(f_X_less, s_dataset_split)) + 1000)*0.1)
    
        lagrangian = -profit + l2 + penalty*10.0
    
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(lagrangian)

    #model training
    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
    

        #print("\nInitialized")
        for step in range(total_steps):

            offset = (step * batch_size * signal_length) % (X_train.shape[0] - batch_size*signal_length)
            batch_X_data = X_train.iloc[offset:(offset + batch_size*signal_length), :]
            batch_s_data = s_train.iloc[offset:(offset + batch_size*signal_length)]

            batch_X_data = batch_X_data.values.reshape(-1, signal_length, num_components)

            batch_s_data = batch_s_data.values.reshape(-1, 1)

            feed_dict = {X_dataset : batch_X_data, s_dataset : batch_s_data}
            _, l, train_predictions, worst = session.run([optimizer, profit, f_X, penalty], feed_dict=feed_dict)

            if step % display_step == 0:
                #feed_dict = {X_dataset : test_X_dataset, s_dataset : test_s_dataset}
                #p_test, test_predictions, worstcase = session.run([profit, f_X, penalty], feed_dict=feed_dict)
                #print("labels: ", batch_s_data[-1])
                #print("f_X: ", train_predictions[-1])
                message = "step {:04d} : profit is {:06.2f}, worst loss on training set {}".format(step, l, worst)
                print(message)
            
    
    
        #make prediction
        
        ntest = X_test.shape[0]//signal_length * signal_length
        
        test_X_dataset = X_test.values[:ntest,:]
        test_X_dataset = test_X_dataset.reshape(-1, signal_length, num_components)
    
        s_dummy = np.zeros((ntest, 1))
        
        feed_dict = {X_dataset : test_X_dataset, s_dataset : s_dummy}
        p1_test, prediction_p1 = session.run([profit, f_X], feed_dict=feed_dict)
    
        if(ntest < X_test.shape[0]):
            test_X_dataset = X_test.values[-signal_length:,:]
            test_X_dataset = test_X_dataset.reshape(-1, signal_length, num_components)
            s_dummy = np.zeros((signal_length, 1))
            feed_dict = {X_dataset : test_X_dataset, s_dataset : s_dummy}
            p2_test, prediction_p2 = session.run([profit, f_X], feed_dict=feed_dict)
            
        prediction_p2 =  prediction_p2[ntest-X_test.shape[0]:,:]  
         
         
        prediction = np.concatenate((prediction_p1, prediction_p2))
        prediction = prediction.reshape(-1)
        
        
        return prediction
        
v = train_and_predict(X_train, s_train, X_validate)

print((s_validate * v).describe())


