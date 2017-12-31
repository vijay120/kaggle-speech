import argparse
from sklearn import preprocessing
from scipy import signal
from scipy.io import wavfile
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import tensorflow as tf
import vggish_input

test_set_ques = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
	# Conv2D wrapper, with bias and relu activation
	x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
	x = tf.nn.bias_add(x, b)
	return tf.nn.relu(x)


def maxpool2d(x, k=2):
	# MaxPool2D wrapper
	return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
						  padding='SAME')

# Create model
def conv_net(x, weights, biases, dropout):
	# MNIST data input is a 1-D vector of 784 features (28*28 pixels)
	# Reshape to match picture format [Height x Width x Channel]
	# Tensor input become 4-D: [Batch Size, Height, Width, Channel]
	x = tf.reshape(x, shape=[-1, 96, 64, 1])

	# Convolution Layer
	conv1 = conv2d(x, weights['wc1'], biases['bc1'])
	# Max Pooling (down-sampling)
	conv1 = maxpool2d(conv1, k=2)

	# Convolution Layer
	conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
	# Max Pooling (down-sampling)
	conv2 = maxpool2d(conv2, k=2)

	print(conv2.get_shape())
	
	# Fully connected layer
	# Reshape conv2 output to fit fully connected layer input
	fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
	fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
	fc1 = tf.nn.relu(fc1)
	# Apply Dropout
	fc1 = tf.nn.dropout(fc1, dropout)

	# Output, class prediction
	out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
	return out

def get_data(dir, ques):
	np.random.seed(0)
	lb = preprocessing.LabelBinarizer()

	folders = [os.path.join(dir, f) for f in listdir(dir) 
				if not isfile(join(dir, f))]

	que_dict = {}
	for folder in folders:
		for que in ques:
			if que in folder:
				if que in test_set_ques:
					que_dict[que] = folder
				else:
					if "unknown" not in que_dict:
						que_dict["unknown"] = []
					que_dict["unknown"].append(folder)

	lb.fit(list(que_dict.keys()))

	X = []
	Y = []

	for que in que_dict.keys():
		if que == "unknown":
			folders = que_dict[que]

			for folder in folders:
				for file in [os.path.join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]:
					spectogram = vggish_input.wavfile_to_examples(file)
					if len(spectogram.shape) > 2:
						spectogram = spectogram[0,:,]
					X.append(spectogram)
					Y.append(lb.transform([que])[0])
		else:
			folder = que_dict[que]
			for file in [os.path.join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]:
				spectogram = vggish_input.wavfile_to_examples(file)
				if len(spectogram.shape) > 2:
					spectogram = spectogram[0,:,]
				X.append(spectogram)
				Y.append(lb.transform([que])[0])

	examples = np.asarray(X)
	labels = np.asarray(Y)
	indices = np.arange(len(examples))
	np.random.shuffle(indices)

	train_cutoff = int(0.7 * len(examples))

	examples_train = examples[indices[:train_cutoff]]
	labels_train = labels[indices[:train_cutoff]]

	examples_val = examples[indices[train_cutoff:]]
	labels_val = labels[indices[train_cutoff:]]

	return examples_train, labels_train, examples_val, labels_val


if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.RawDescriptionHelpFormatter)

	parser.add_argument('q', help='A comma separated list of ques')
	parser.add_argument('dir', help='The directory containing the files')

	args = parser.parse_args()

	ques = set(args.q.split(','))
	dir = args.dir

	examples_train, labels_train, examples_val, labels_val = get_data(dir, ques)

	# Training Parameters
	learning_rate = 0.001
	num_steps = 500
	batch_size = 50
	display_step = 1
	epochs = 10

	# Network Parameters
	num_classes = len(test_set_ques) + 1 # MNIST total classes (0-9 digits)
	dropout = 0.75 # Dropout, probability to keep units

	# tf Graph input
	X = tf.placeholder(tf.float32, [None, examples_train.shape[1], examples_train.shape[2]])
	Y = tf.placeholder(tf.float32, [None, num_classes])
	keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

	# Store layers weight & bias
	weights = {
		# 5x5 conv, 1 input, 32 outputs
		'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
		# 5x5 conv, 32 inputs, 64 outputs
		'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
		# fully connected, 7*7*64 inputs, 1024 outputs
		'wd1': tf.Variable(tf.random_normal([24*16*64, 1024])),
		# 1024 inputs, 10 outputs (class prediction)
		'out': tf.Variable(tf.random_normal([1024, num_classes]))
	}

	biases = {
		'bc1': tf.Variable(tf.random_normal([32])),
		'bc2': tf.Variable(tf.random_normal([64])),
		'bd1': tf.Variable(tf.random_normal([1024])),
		'out': tf.Variable(tf.random_normal([num_classes]))
	}

	# Construct model
	logits = conv_net(X, weights, biases, keep_prob)
	prediction = tf.nn.softmax(logits)

	# Define loss and optimizer
	loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
		logits=logits, labels=Y))

	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
	train_op = optimizer.minimize(loss_op)

	# Evaluate model
	correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	# Initialize the variables (i.e. assign their default value)
	init = tf.global_variables_initializer()

	# Start training
	with tf.Session() as sess:

		# Run the initializer
		sess.run(init)

		for i in range(epochs):
			for step in range(int(len(examples_train)/batch_size)):
				batch_x = examples_train[step*batch_size : (step+1)*batch_size]
				batch_y = labels_train[step*batch_size : (step+1)*batch_size]
				
				if len(batch_x) == 0:
					break
						
				# Run optimization op (backprop)
				sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout})
				if step % display_step == 0 or step == 1:
					# Calculate batch loss and accuracy
					loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
																		 Y: batch_y,
																		 keep_prob: 1.0})
					# print("Step " + str(step) + ", Minibatch Loss= " + \
					# 	  "{:.4f}".format(loss) + ", Training Accuracy= " + \
					# 	  "{:.3f}".format(acc) + " for epoch {}".format(i))

			total_acc = 0
			for step in range(int(len(examples_val)/batch_size)):
				batch_x = examples_val[step*batch_size : (step+1)*batch_size]
				batch_y = labels_val[step*batch_size : (step+1)*batch_size]		
				batch_acc = sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0})
				total_acc += batch_acc/(int(len(examples_val)/batch_size)*1.0)

			print("Validation acc is: {}".format(total_acc))



