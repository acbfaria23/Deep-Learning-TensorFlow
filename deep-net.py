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
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3,n_classes])),
                      'biases':tf.Variable(tf.random_normal([n_classes]))}

    # (input_data * weights) + biases

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights'])+ output_layer['biases']

    return output

def train_neural_network(x):
    #the array expected from the model output
    prediction = neural_network_model(x)
    #this is the type of the output we defined using OneHot, any other cost function is welcome
    #difference between the prediction from the non labeled 
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction,labels=y))

    #learning rate is a parameter for AdamOptimizer default = 0.001
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    # cycles feed forward and backprop
    hm_epochs = 3
    
    #so in fact this function divide a group of data to minimize the loss 
    with tf.Session()as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range (hm_epochs):
                epoch_loss = 0
                #number of examples/bunch we defined before.
                for _ in range(int(mnist.train.num_examples/batch_size)):
                    epoch_x,epoch_y = mnist.train.next_batch(batch_size)
                    _, c = sess.run([optimizer,cost], feed_dict = {x: epoch_x,y:epoch_y})
                    epoch_loss += c
                    print('Epoch',epoch, 'completed out of', hm_epochs, 'and the loss is:', epoch_loss)

    #sees if the array expected is equal (not sure who is Y)
        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))   
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
        
train_neural_network(x)
