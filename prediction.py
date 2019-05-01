import tensorflow as tf
import pickle
import random
from io import StringIO
import sys, json
result = StringIO()
X=pickle.load(open('X.pickle','rb'))
Y=pickle.load(open('y.pickle','rb'))
learning_rate = 0.00000005
training_epochs = 30
batch_size = 32
display_step = 1

x = tf.placeholder(tf.float32, [None, 2500]) 
y = tf.placeholder(tf.float32, [None, 2]) 

W = tf.Variable(tf.zeros([2500, 2]))
b = tf.Variable(tf.zeros([2]))
saver = tf.train.Saver([W,b])

pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax1


cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
values = [0, 2, 3, 5, 8, 6969, 420, 6090, 1234, 1587, 13992, 12104]
init = tf.global_variables_initializer()
def run():
    rslt = dict()
    print('hi')
    random.shuffle(values)
    numin = values[0]
    with tf.Session() as sess:
        


        sess.run(init)
        saver = tf.train.import_meta_graph('Trainmalaria.meta')
        saver.restore(sess,tf.train.latest_checkpoint('./'))


#        for epoch in range(training_epochs):
#            avg_cost = 0
#            total_batch = int(len(X)/batch_size)
#            # Loop over all batches
#            for i in range(total_batch):
#               # Run optimization op (backprop) and cost op (to get loss value)
#                _, c = sess.run([optimizer, cost], feed_dict={x: X[0:13780],
#                                                              y: Y[0:13780]})
#                # Compute average loss
#                avg_cost += c / total_batch
#            # Display logs per epoch step
#            if (epoch+1) % display_step == 0:
#                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
#
        print("Optimization Finished!")
        #print("Give us a number based on the coresponding image ex: first image would be 0")
        saver.save(sess, "Trainmalaria")
        
        #numin = int(input())


        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        old_stdout = sys.stdout
        sys.stdout = result
        prediction = pred.eval({x:[X[numin]], y:[Y[numin]]})
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Prediction:", prediction)
        rslt["left"] = prediction[0][1]
        rslt["right"] = prediction[0][0]
        print("Correct Answer:", y.eval({x:[X[numin]], y:[Y[numin]]}))
        print("Accuracy:", accuracy.eval({x:[X[numin]], y:[Y[numin]]}))
        if(prediction[0][1] > prediction[0][0]):
            print("Your cells are healthy")
            rslt["diag"] = "healthy"
        else:
            print("WARNING: Malaria has been detected...")
            rslt["diag"] = "malaria"
        sys.stdout = old_stdout
        rslt = result.getvalue()
        #rslt = json.dumps(rslt)
        return rslt
