'''
this code is made by Ana Cristina Barbosa following
the tutorial from the sentdex youtube channel.
How it will work:

input data > weitght > hidden layer 1 (activation function) > weights > hidden layer 2
(activation function) weights > output layer
this is a feed-forward network

compare output to intended output = cost function (cross entropy)
optimization function (optimizer) > minimize cost (AdanOptimizer...SGD, AdaGrad)

the manipulation of weights to ensure results is the BACKPROPAGATION

feed forward + backprop = epoch (this is one cicle)

'''
import tensorflow as tf
'''good for multiclass classification
for example: one_hot does something like class 1 is actually [1,0,0,0,0] and so on'''
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

#number of nodes at the hidden layer
n_nodes_hl1 = 500
n_nodes_hl2 = 1000
n_nodes_hl3 = 50

n_classes = 10    
#go to a bunch of 100 features and manipulate them
batch_size = 100

#height x width
x = tf.placeholder('float',[None, 784])
y = tf.placeholder('float')

def neural_network_model(data):
    #random weights for 1st layer one gigant tensor
    
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784,n_nodes_hl1])),
                      'biases':tf.Variables(tf.random_normal(n_nodes_hl1))}
    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
                      'biases':tf.Variables(tf.random_normal(n_nodes_hl2))}
    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
                      'biases':tf.Variables(tf.random_normal(n_nodes_hl3))}
    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3,n_classes])),
                      'biases':tf.Variables(tf.random_normal(n_classe))}

    # (input_data * weights) + biases

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights'])+ hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights'])+ hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l3, hidden_3_layer['weights'])+ hidden_3_layer['biases'])
    l1 = tf.nn.relu(l2)

    output = tf.matmul(l3, output_layer['weights'])+ output_layer['biases']

    return output
