import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
import random

num_epochs = 20000
dataset_lenght = 100
window_size = 4
num_batches = 1
batch_size = dataset_lenght
alpha = 0.01
nW_hidden = 5
# nW_hidden2 = 2 # in case

# t = np.arange(0,20.0, 0.1)
# data = np.sin(t)

t = np.arange(0, 2*np.pi, 2*np.pi/dataset_lenght)
# xdata = np.cos(t)/2 # circle
# ydata = np.sin(t)/2
# xdata = np.cos(t)/2 + np.cos(10*t)*0.05 # close to circle
# ydata = np.sin(t)/2 + np.cos(10*t)*0.05

xdata = np.cos(t)/2 + np.sin(10*t)*0.05 # close to circle
ydata = np.sin(t)/2 + np.cos(10*t)*0.05
# xdata = t


# data = [np.sin(2*np.pi*i/100) for i in range(dataset_lenght)]
data = np.asarray(zip(xdata, ydata)).flatten()
# plt.plot(data[0::2],data[1::2])
# plt.show()
x = tf.placeholder("float", [None, 2*window_size])  # "None" as dimension for versatility between batches and non-batches
y_ = tf.placeholder("float", [batch_size, 2])

# W = tf.Variable(np.float32(np.random.rand(step, 1))*0.1)
# b = tf.Variable(np.float32(np.random.rand(1))*0.1)

# y = tf.sigmoid(tf.matmul(x, W) + b)

W_hidden = tf.Variable(tf.truncated_normal([2*window_size, nW_hidden]))
b_hidden = tf.Variable(tf.truncated_normal([nW_hidden]))
y_hidden = tf.tanh(tf.matmul(x, W_hidden) + b_hidden)

# W_hidden2 = tf.Variable(tf.truncated_normal([nW_hidden,nW_hidden2]))
# b_hidden2 = tf.Variable(tf.truncated_normal([nW_hidden2]))
# y_hidden2 = tf.tanh(tf.matmul(y_hidden, W_hidden2) + b_hidden2)


# W_output = tf.Variable(tf.truncated_normal([nW_hidden2, 2])) # If 2 layers
W_output = tf.Variable(tf.truncated_normal([nW_hidden, 2]))
b_output = tf.Variable(tf.truncated_normal([2]))
# y = tf.tanh(tf.matmul(y_hidden2, W_output) + b_output)
y = tf.tanh(tf.matmul(y_hidden, W_output) + b_output) # If 2 layers
# y = tf.sigmoid(tf.matmul(y_hidden, W_output) + b_output)


error_measure = tf.reduce_sum(tf.square(y_ - y))
# cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train = tf.train.GradientDescentOptimizer(alpha).minimize(error_measure)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print "----------------------"
print "   Start training...  "
print "----------------------"

#NOW: whole set of windows should is generated, then randomized and then batched

random_window_indexes = [2*i for i in range(dataset_lenght//2)]
shuffle(random_window_indexes)


i = 0
for epoch in range(num_epochs):
    for current_batch in range(num_batches): #One batch
        xbatch = np.zeros([batch_size,window_size*2])
        ybatch = np.zeros([batch_size,2])
        for yy in range(batch_size): #Each element of the batch is a window-sized set of coordinate pairs
            ri = random_window_indexes[i%len(random_window_indexes)]
            i=i+1
            for jj in range(0,window_size*2,2):
                xbatch[yy][jj] = xdata[(ri+jj)%dataset_lenght]
                xbatch[yy][jj+1] = ydata[(ri+jj+1)%dataset_lenght]
            ybatch[yy][0] = xdata[(ri+2*window_size)%dataset_lenght]
            ybatch[yy][1] = ydata[(ri+2*window_size+1)%dataset_lenght]
            # print "xbatch",xbatch[yy]
            # print "ybatch",ybatch[yy]

        sess.run(train, feed_dict={x: xbatch, y_: ybatch})
        #print sess.run(error_measure, feed_dict={x: xs, y_: ys})
    if (epoch % (num_epochs//10)) == 0:
        print "error:",sess.run(error_measure, feed_dict={x: xbatch, y_: ybatch})
        #print sess.run(y, feed_dict={x: xs})
        #print "----------------------------------------------------------------------------------"

print "----------------------"
print "   Start testing...  "
print "----------------------"
outs = data[:2*window_size].tolist()
test_size = dataset_lenght
for yy in range(test_size):
    xs = np.atleast_2d([outs[2*yy+i] for i in range(2*window_size)])
    out = sess.run(y, feed_dict={x: xs})
    outs.append(out[0][0])
    outs.append(out[0][1])
    # print xs
    # print outs

plt.plot(xdata,ydata)
xout = [outs[i] for i in range(0,test_size,2)]
yout = [outs[i] for i in range(1,test_size,2)]
# yout = outs[range(1,len(outs),2)]
plt.plot(xout,yout)
plt.plot(xout[-1], yout[-1], 'ro')
plt.show()
