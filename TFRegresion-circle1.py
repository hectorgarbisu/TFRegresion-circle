import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
import random

num_epochs = 100
dataset_lenght = 100
num_batches = dataset_lenght/2
window_size = 40
batch_size = 50
alpha = 0.01
nW_hidden = 10

# t = np.arange(0,20.0, 0.1)
# data = np.sin(t)

t = range(dataset_lenght)
xdata = np.cos(t)/2
ydata = np.sin(t)/2
#data = [np.sin(2*np.pi*i/100) for i in range(dataset_lenght)]
data = np.asarray(zip(xdata,ydata)).flatten()
x = tf.placeholder("float", [None,2*window_size]) #"None" as dimension for versatility between batches and non-batches
y_ = tf.placeholder("float", [batch_size,2])

# W = tf.Variable(np.float32(np.random.rand(step, 1))*0.1)
# b = tf.Variable(np.float32(np.random.rand(1))*0.1)

# y = tf.sigmoid(tf.matmul(x, W) + b)

W_hidden = tf.Variable(tf.truncated_normal([2*window_size, nW_hidden]))
b_hidden = tf.Variable(tf.truncated_normal([nW_hidden]))

W_output = tf.Variable(tf.truncated_normal([nW_hidden, 2]))
b_output = tf.Variable(tf.truncated_normal([2]))

y_hidden = tf.tanh(tf.matmul(x, W_hidden) + b_hidden)

# y = tf.sigmoid(tf.matmul(y_hidden, W_output) + b_output)
y = tf.tanh(tf.matmul(y_hidden, W_output) + b_output)


error_measure = tf.reduce_sum(tf.square(y_ - y))
# cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train = tf.train.GradientDescentOptimizer(alpha).minimize(error_measure)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print "----------------------"
print "   Start training...  "
print "----------------------"

#BEFORE: All samples of size 'window_size' are randomized within each serial batch
# That generates a set of batches that are very local to the dataset
#NOW: whole set of windows should is generated, then randomized and then batched

windows_indexes = 2*range(num_batches)
shuffle(windows_indexes)

for epoch in range(num_epochs):
    for current_batch in range(num_batches): #One batch
        xbatch = list()
        ybatch = list()
        for yy in range(batch_size):
            newx = (([(data[(i+yy+windows_indexes[current_batch])%dataset_lenght],
                            data[(i+yy+windows_indexes[current_batch]+1)%dataset_lenght])
                            for i in range(window_size)]))
            xbatch.append(newx)
            ybatch.append(([data[(window_size+yy+windows_indexes[current_batch])%len(data)],
                            data[(window_size+yy+windows_indexes[current_batch]+1)%len(data)]]))

        #index_shuffle = range(batch_size)
        #shuffle(index_shuffle)
        #xbatch = [xbatch[i] for i in index_shuffle]
        #ybatch = [ybatch[i] for i in index_shuffle]
        xs = np.atleast_2d(xbatch)
        ys = np.atleast_2d(ybatch)
        #print(xs)
        xs = xs.reshape([batch_size,window_size*2])
        print xs
        sess.run(train, feed_dict={x: xs, y_: ys})
        #print sess.run(error_measure, feed_dict={x: xs, y_: ys})
    if (epoch % (num_epochs//10)) == 0:
        print "error:",sess.run(error_measure, feed_dict={x: xs, y_: ys})
        #print sess.run(y, feed_dict={x: xs})
        #print "----------------------------------------------------------------------------------"

print "----------------------"
print "   Start testing...  "
print "----------------------"
outs = data[:2*window_size]
for i in range(dataset_lenght-2*window_size,2):
    xs = np.atleast_2d([outs[jj+i] for jj in range(window_size)])
    out = sess.run(y, feed_dict={x: xs})
    #print xs, out
    outs.append(out[0][0])
    outs.append(out[0][1])

plt.plot(xdata,ydata)
plt.plot(outs[window_size-1], outs[window_size], 'ro')
xout = outs[range(0,len(outs),2)]
yout = outs[range(1,len(outs),2)]
plt.plot(xout,yout)
plt.show()
