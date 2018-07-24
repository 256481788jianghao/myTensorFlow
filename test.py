import tensorflow as tf

#help(tf.Variable)

m = 1
n = 1

b = tf.Variable(tf.zeros([m,1]))

w = tf.Variable(tf.zeros([m,n]))

in_data = tf.placeholder('float',[1,1])
out_data = tf.placeholder('float',[1,1])

level1 = tf.matmul(w,in_data)+b

cost_fun = (level1 - out_data + 1)**2
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost_fun)



init = tf.global_variables_initializer()

sess = tf.InteractiveSession()
print(sess.run(init))
for i in range(0,1000):
    print(sess.run(train_step,feed_dict={in_data:[[1.0]],out_data:[[0.0]]}))
    print('w=%f'%sess.run(w)+' b=%f'%sess.run(b))
    print(sess.run(cost_fun,feed_dict={in_data:[[1.0]],out_data:[[0.0]]}))
