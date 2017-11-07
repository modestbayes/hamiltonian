import numpy as np
import tensorflow as tf

higgs = np.load('higgs.npy')
features = higgs[:, 1:29]
label = higgs[:, 0]

alpha = 0.01 # prior variance
nLeap = 10.0
stepsize = 0.001

D = features.shape[1]

X = tf.constant(features, dtype='float32')
Y = tf.constant(label, dtype='float32')

beta = tf.Variable(tf.zeros([D, 1]))

eta = tf.matmul(X, beta)
p = 1.0 / (1.0 + tf.exp(-eta))

loglik = tf.reduce_sum(Y * tf.log(p) + (1.0 - Y) * tf.log(1 - p))
logprior = -0.5 * tf.reduce_sum(beta * beta) / alpha

energy = -(loglik + logprior)

nabla = tf.gradients(energy, beta)

delta = tf.placeholder('float', [D, 1])
update = beta.assign_add(delta)

with tf.Session() as sess:
    tf.initialize_all_variables().run()
    
    currentU = -sess.run(loglik)
    
    proposed = 0
    accepted = 0
    
    for i in range(0, 500):
        proposedP = np.random.normal(0, 1, D)
        proposedP = proposedP.reshape((D, 1))
        currentP = proposedP
        move = np.zeros((D, 1))
        
        randomSteps = int(np.random.uniform(0, 1) * nLeap)
        for stepNum in range(0, randomSteps):
            grad = sess.run(nabla)[0]
            if np.any(np.isnan(grad)):
                break
            proposedP = proposedP - 0.5 * stepsize * grad
            move += proposedP * stepsize
            sess.run(update, feed_dict={delta: proposedP * stepsize})
            grad = sess.run(nabla)[0]
            proposedP = proposedP - 0.5 * stepsize * grad
            
        proposedP = -proposedP
    
        proposedU = -sess.run(loglik)
        currentH = currentU + 0.5 * np.sum(np.square(currentP))
        proposedH = proposedU + 0.5 * np.sum(np.square(proposedP))
    
        ratio = currentH - proposedH
                
        if not np.isnan(ratio) and ratio > min(0.0, np.log(np.random.uniform(0, 1))):
            currentU = proposedU
            accepted = accepted + 1
        else:
            sess.run(update, feed_dict={delta: -move})

	if (i + 1) % 100 == 0:
	   print('100 draws...')
