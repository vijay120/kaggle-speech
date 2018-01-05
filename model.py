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
import csv
from scipy import signal
from sklearn.metrics import confusion_matrix

test_set_ques = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "silence"]
extracted_classes = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'silence', 'stop',
       'unknown', 'up', 'yes']

confusion_labels = ['bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four', 'happy', 
'house', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 
'six', 'stop', 'three', 'tree', 'two', 'up', 'wow', 'yes', 'zero', 'go', 'silence']

def batch_norm(x,
               phase,
               shift=True,
               scale=True,
               momentum=0.99,
               eps=1e-3,
               internal_update=False,
               scope=None,
               reuse=None):

    C = x._shape_as_list()[-1]
    ndim = len(x.shape)
    var_shape = [1] * (ndim - 1) + [C]

    with tf.variable_scope(scope, 'batch_norm', reuse=reuse):
        def training():
            m, v = tf.nn.moments(x, range(ndim - 1), keep_dims=True)
            update_m = _assign_moving_average(moving_m, m, momentum, 'update_mean')
            update_v = _assign_moving_average(moving_v, v, momentum, 'update_var')
            tf.add_to_collection('update_ops', update_m)
            tf.add_to_collection('update_ops', update_v)

            if internal_update:
                with tf.control_dependencies([update_m, update_v]):
                    output = (x - m) * tf.rsqrt(v + eps)
            else:
                output = (x - m) * tf.rsqrt(v + eps)
            return output

        def testing():
            m, v = moving_m, moving_v
            output = (x - m) * tf.rsqrt(v + eps)
            return output

        # Get mean and variance, normalize input
        moving_m = tf.get_variable('mean', var_shape, initializer=tf.zeros_initializer, trainable=False)
        moving_v = tf.get_variable('var', var_shape, initializer=tf.ones_initializer, trainable=False)

        if isinstance(phase, bool):
            output = training() if phase else testing()
        else:
            output = tf.cond(phase, training, testing)

        if scale:
            output *= tf.get_variable('gamma', var_shape, initializer=tf.ones_initializer)

        if shift:
            output += tf.get_variable('beta', var_shape, initializer=tf.zeros_initializer)

    return output

def log_spectrogram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
	# Conv2D wrapper, with bias and relu activation
	x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
	x = tf.nn.bias_add(x, b)
	return x

def maxpool2d(x, p, q):
	# MaxPool2D wrapper
	return tf.nn.max_pool(x, ksize=[1, p, q, 1], strides=[1, p, q, 1], padding='SAME')

# Create model
def conv_net(x, weights, biases, dropout, trainable):
	# MNIST data input is a 1-D vector of 784 features (28*28 pixels)
	# Reshape to match picture format [Height x Width x Channel]
	# Tensor input become 4-D: [Batch Size, Height, Width, Channel]
	#x = tf.reshape(x, shape=[-1, 98, 161, 1])
	# x = tf.cond(trainable,
	# 		lambda: tf.contrib.layers.batch_norm(x, decay=0.9, center=False, scale=True, updates_collections=None, is_training=True),
	# 		lambda: tf.contrib.layers.batch_norm(x, decay=0.9, center=False, scale=True, updates_collections=None, is_training=False))

	# x = tf.cond(trainable,
	# 		lambda: batch_norm_wrapper(x,True,0.9),
	# 		lambda: batch_norm_wrapper(x,False,0.9))

	#x = batch_norm_wrapper(x, True, 0.9)

	x = tf.reshape(x, shape=[-1, 98, 126, 1])

	# Convolution Layer
	conv1 = conv2d(x, weights['wc1'], biases['bc1'])
	conv1 = tf.nn.relu(conv1)
	conv1 = maxpool2d(conv1, 2, 3)
	conv1 = tf.nn.dropout(conv1, dropout)
	#fc1 = tf.nn.dropout(fc1, dropout)

	# Convolution Layer
	conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
	conv2 = tf.nn.relu(conv2)
	conv2 = maxpool2d(conv2, 2, 3)
	conv2 = tf.nn.dropout(conv2, dropout)

	print(conv2.get_shape())

	# Convolution Layer
	conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
	conv3 = tf.nn.relu(conv3)
	conv3 = maxpool2d(conv3, 2, 3)
	conv3 = tf.nn.dropout(conv3, dropout)

	print(conv3.get_shape())

	# Fully connected layer
	# Reshape conv2 output to fit fully connected layer input
	fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
	fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
	fc1 = tf.nn.relu(fc1)

	# fc1 = tf.cond(trainable,
	# 		lambda: tf.contrib.layers.batch_norm(fc1, decay=0.9, center=False, scale=True, updates_collections=None, is_training=True),
	# 		lambda: tf.contrib.layers.batch_norm(fc1, decay=0.9, center=False, scale=True, updates_collections=None, is_training=False))

	# fc1 = tf.cond(trainable,
	# 		lambda: batch_norm_wrapper(fc1,True,0.9),
	# 		lambda: batch_norm_wrapper(fc1,False,0.9))

	# Apply Dropout
	# fc1 = tf.nn.dropout(fc1, dropout)

	# Output, class prediction
	out = tf.add(tf.matmul(fc1, weights['out']), biases['bout'])
	return out

def label_classes(dir, ques):
	lb = preprocessing.LabelBinarizer()

	folders = [os.path.join(dir, f) for f in listdir(dir) 
				if not isfile(join(dir, f))]

	que_dict = {}
	for folder in folders:
		for que in ques:
			if que in folder:
				#que_dict[que] = folder
				if que in test_set_ques:
					que_dict[que] = folder
				else:
					if "unknown" not in que_dict:
						que_dict["unknown"] = []
					que_dict["unknown"].append(folder)

	lb.fit(list(que_dict.keys()))
	return lb.classes_

def get_data(dir, ques):
	np.random.seed(0)
	lb = preprocessing.LabelBinarizer()

	folders = [os.path.join(dir, f) for f in listdir(dir) 
				if not isfile(join(dir, f))]

	que_dict = {}
	for folder in folders:
		for que in ques:
			if que in folder:
				#que_dict[que] = folder
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
					# y, sr = librosa.load(file, sr=16000)
					# S = librosa.feature.melspectrogram(y, sr=sr, n_mels=256)
					# log_S = librosa.power_to_db(S, ref=np.max)

					spectogram = vggish_input.wavfile_to_examples(file)
					X.append(spectogram)
					Y.append(lb.transform([que])[0])
		else:
			folder = que_dict[que]
			for file in [os.path.join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]:
				#sample_rate, samples = wavfile.read(file)
				#_, _, spectrogram = log_spectrogram(samples, sample_rate)
				# y, sr = librosa.load(file, sr=16000)
				# S = librosa.feature.melspectrogram(y, sr=sr, n_mels=256)
				# log_S = librosa.power_to_db(S, ref=np.max)
				
				spectogram = vggish_input.wavfile_to_examples(file)
				#X.append(spectrogram[:98])
				X.append(spectogram)
				Y.append(lb.transform([que])[0])

			if len(X) > 1000:
				break

	examples = np.asarray(X)
	labels = np.asarray(Y)
	indices = np.arange(len(examples))
	np.random.shuffle(indices)

	train_cutoff = int(0.99 * len(examples))

	examples_train = examples[indices[:train_cutoff]]
	labels_train = labels[indices[:train_cutoff]]

	examples_val = examples[indices[train_cutoff:]]
	labels_val = labels[indices[train_cutoff:]]

	return examples_train, labels_train, examples_val, labels_val


def get_data_predict(folder):
	X = []

	counter = 0
	for file in [os.path.join(folder, f) for f in listdir(folder)]:
		sample_rate, samples = wavfile.read(file)
		#_, _, spectrogram = log_spectrogram(samples, sample_rate)
		#spectogram = np.transpose(vggish_input.wavfile_to_examples(file)[0,:,])

		# y, sr = librosa.load(file, sr=16000)
		# S = librosa.feature.melspectrogram(y, sr=sr, n_mels=256)
		# log_S = librosa.power_to_db(S, ref=np.max)
		#X.append(spectrogram[:98])
		spectogram = vggish_input.wavfile_to_examples(file)
		X.append(spectogram)

		counter += 1
		if counter%1000==0:
			print(counter)
			#break

	return np.asarray(X)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.RawDescriptionHelpFormatter)

	parser.add_argument('q', help='A comma separated list of ques')
	parser.add_argument('dir', help='The directory containing the files')
	parser.add_argument('p', help='Predict time workflow')
	parser.add_argument('s', help='Save model dir')

	args = parser.parse_args()

	ques = set(args.q.split(','))
	dir = args.dir
	predict_file = args.p
	saved_model_dir = args.s

	classes_ = label_classes(dir, ques)
	print("Classes: {}".format(classes_))
	batch_size = 16

	# tf Graph input
	num_classes = len(test_set_ques) + 1
	
	#num_classes = len(ques)
	#X = tf.placeholder(tf.float32, [None, 98, 161])
	X = tf.placeholder(tf.float32, [None, 98, 128])
	Y = tf.placeholder(tf.float32, [None, num_classes])
	train_phase = tf.placeholder(tf.bool)
	keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

	if predict_file != "":
		print("Predict time modelling")
		imported_meta = tf.train.import_meta_graph("{}model_final.meta".format(saved_model_dir)) 
		predict_data = get_data_predict(predict_file)
		#init = tf.global_variables_initializer()

		with tf.Session() as sess:
			#sess.run(init)
			last_checkpoint = tf.train.latest_checkpoint(saved_model_dir)
			print("Loading checkpoint: {}".format(last_checkpoint))
			imported_meta.restore(sess, last_checkpoint)

			graph = tf.get_default_graph()

			# Store layers weight & bias
			weights = {
				# 5x5 conv, 1 input, 32 outputs
				'wc1': graph.get_tensor_by_name("wc1:0"),
				# 5x5 conv, 32 inputs, 64 outputs
				'wc2': graph.get_tensor_by_name("wc2:0"),
				'wc3': graph.get_tensor_by_name("wc3:0"),
				# fully connected, 7*7*64 inputs, 1024 outputs
				'wd1': graph.get_tensor_by_name("wd1:0"),
				# 1024 inputs, 10 outputs (class prediction)
				'out': graph.get_tensor_by_name("out:0")
			}

			biases = {
				'bc1': graph.get_tensor_by_name("bc1:0"),
				'bc2': graph.get_tensor_by_name("bc2:0"),
				'bc3': graph.get_tensor_by_name("bc3:0"),
				'bd1': graph.get_tensor_by_name("bd1:0"),
				'bout': graph.get_tensor_by_name("bout:0"),
			}

			# Construct model
			logits = conv_net(X, weights, biases, keep_prob, train_phase)
			prediction = tf.nn.softmax(logits)
			arg_max_prediction = tf.argmax(prediction, 1)

			results = []
			steps = int(len(predict_data)/batch_size)
			for i in range(steps + 1):
				start_index = i * batch_size
				end_index = (i+1) * batch_size

				labels = sess.run([arg_max_prediction], feed_dict={X: predict_data[start_index: end_index], keep_prob: 1.0, train_phase: False})
				results += labels[0].tolist()

			FIELD_NAMES = ["fname", "label"]

			with open("out.csv", 'w+') as out:
				writer = csv.DictWriter(out, delimiter=',', fieldnames=FIELD_NAMES)
				writer.writeheader()

				counter = 0
				for file in listdir(predict_file):
					label = classes_[results[counter]]
					# if label not in test_set_ques:
					# 	label = "unknown"
					row = {'fname':file, 'label':label}
					writer.writerow(row)
					counter += 1

	#wd1': tf.Variable(tf.truncated_normal([49*54*94, 128], stddev=0.01), name='wd1'),

	else:
		# Store layers weight & bias
		weights = {
			# 5x5 conv, 1 input, 32 outputs
			'wc1': tf.Variable(tf.truncated_normal([21, 8, 1, 94], stddev=0.01), name='wc1'),
			# 5x5 conv, 32 inputs, 64 outputs
			'wc2': tf.Variable(tf.truncated_normal([15, 5, 94, 94], stddev=0.01), name='wc2'),
			'wc3': tf.Variable(tf.truncated_normal([6, 4, 94, 94], stddev=0.01), name='wc3'),
			# fully connected, 7*7*64 inputs, 1024 outputs
			'wd1': tf.Variable(tf.truncated_normal([8*4*94, 128], stddev=0.01), name='wd1'),
			# 1024 inputs, 10 outputs (class prediction)
			'out': tf.Variable(tf.truncated_normal([128, num_classes], stddev=0.01), name='out')
		}

		biases = {
			'bc1': tf.Variable(tf.zeros([94]), name='bc1'),
			'bc2': tf.Variable(tf.zeros([94]), name='bc2'),
			'bc3': tf.Variable(tf.zeros([94]), name='bc3'),
			'bd1': tf.Variable(tf.zeros([128]), name='bd1'),
			'bout': tf.Variable(tf.zeros([num_classes]), name='bout')
		}

		# Construct model
		logits = conv_net(X, weights, biases, keep_prob, train_phase)
		prediction = tf.nn.softmax(logits)
		arg_max_prediction = tf.argmax(prediction, 1)

		print("Training time modelling")
		examples_train, labels_train, examples_val, labels_val = get_data(dir, ques)

		# Training Parameters
		learning_rate = 0.001
		num_steps = 500
		display_step = 100
		epochs = 15

		# Network Parameters
		dropout = 0.5 # Dropout, probability to keep units

		# Define loss and optimizer
		loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
		train_op = optimizer.minimize(loss_op)

		# Evaluate model
		labels_tf = tf.argmax(Y, 1)
		correct_pred = tf.equal(arg_max_prediction, labels_tf)
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

		saver = tf.train.Saver(tf.global_variables())
		# Initialize the variables (i.e. assign their default value)
		init = tf.global_variables_initializer()

		# Start training
		with tf.Session() as sess:
			# Run the initializer
			sess.run(init)

			global_step = 0

			for i in range(epochs):
				indices = np.arange(len(examples_train))
				np.random.shuffle(indices)
				examples_train = examples_train[indices]
				labels_train = labels_train[indices]

				for step in range(int(len(examples_train)/batch_size)):

					batch_x = examples_train[step*batch_size : (step+1)*batch_size]
					batch_y = labels_train[step*batch_size : (step+1)*batch_size]
					
					if len(batch_x) == 0:
						break
							
					# Run optimization op (backprop)
					sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout, train_phase: True})
					if step % display_step == 0 or step == 1:
						# Calculate batch loss and accuracy
						loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
																			 Y: batch_y,
																			 keep_prob: 1.0,
																			 train_phase: True})
						print("Step " + str(step) + ", Minibatch Loss= " + \
							  "{:.4f}".format(loss) + ", Training Accuracy= " + \
							  "{:.3f}".format(acc) + " for epoch {}".format(i))

						global_step += 1
						print("Global step: {}".format(global_step))
						saver.save(sess, '{}model_iter'.format(saved_model_dir), global_step=global_step)

				saver.save(sess, '{}model_final'.format(saved_model_dir))

				total_labels = []
				total_arg_max_prediction = []
				total_acc = 0
				for step in range(int(len(examples_val)/batch_size)):
					batch_x = examples_val[step*batch_size : (step+1)*batch_size]
					batch_y = labels_val[step*batch_size : (step+1)*batch_size]
					batch_acc, labels_np, arg_max_prediction_np = sess.run([accuracy, labels_tf, arg_max_prediction], feed_dict={X: batch_x, 
						Y: batch_y, keep_prob: 1.0, train_phase: False})
					total_labels += list(labels_np)
					total_arg_max_prediction += list(arg_max_prediction_np)
					total_acc += batch_acc/(int(len(examples_val)/batch_size)*1.0)

				print("Confusion matrix is:\n {}".format(confusion_matrix(total_labels, total_arg_max_prediction)))

				print("Validation acc is: {}".format(total_acc))

			### do some predict time stuff
			print("Predict time modelling")
			predict_file = "/data/kaggle2/test/audio"
			predict_data = get_data_predict(predict_file)

			results = []
			steps = int(len(predict_data)/batch_size)
			for i in range(steps + 1):
				start_index = i * batch_size
				end_index = (i+1) * batch_size
				labels = sess.run([arg_max_prediction], feed_dict={X: predict_data[start_index: end_index], keep_prob: 1.0, train_phase: False})
				results += labels[0].tolist()
			FIELD_NAMES = ["fname", "label"]
			with open("out.csv", 'w+') as out:
				writer = csv.DictWriter(out, delimiter=',', fieldnames=FIELD_NAMES)
				writer.writeheader()
				counter = 0
				for file in listdir(predict_file):
					label = classes_[results[counter]]
					# if label not in test_set_ques:
					# 	label = "unknown"
					row = {'fname':file, 'label':label}
					writer.writerow(row)
					counter += 1