import numpy as np
from optimizers import GradientDescentOptimizer
from modules import Sequence, Linear, ReLU

def mnist(val=False):
	from mnist import MNIST

	print "Loading MNIST {} data ..".format("training" if not val else "validation")

	mndata = MNIST('./mnist_data')
	mndata.gz = True
	if not val:
		images, labels = mndata.load_training()
	else:
		images, labels = mndata.load_testing()
	images = np.array(images,dtype=np.float) / 255.0
	labels = np.array(labels,dtype=np.float).reshape(len(labels),1)
	print "Images shape", images.shape
	print "Labels shape", labels.shape
	
	def gen():
		idxs = np.arange(len(images))
		if not val:
			np.random.shuffle(idxs)
		for idx in idxs:
			yield images[idx],labels[idx]
	return gen

def l2_loss(label,output):
	return ((label - output) ** 2).sum() / 2

def l2_loss_grad(label,output):
	return -(label - output)

def log_softmax(logits):
	return logits - np.log(np.sum(np.exp(logits),axis=0))

def cross_entropy_loss_with_logits(label,logits):
	return -log_softmax(logits)[label]

def cross_entropy_loss_with_logits_grad(label,output):
	temp = np.zeros(output.shape[0])
	temp[label] = 1.0
	return (output - temp)

def train(epochs,batch_size,hidden_size,learning_rate):
	"""
	    Train a simple feed-forward network to classify MNIST digits,
	    using vanilla SGD to minimize the categorical cross entropy between
	    network outputs and ground truth labels.
	"""
	
	ff = Sequence(
		Linear(784,hidden_size),
		ReLU(),
		Linear(hidden_size,hidden_size),
		ReLU(),
		Linear(hidden_size,10)
		)

	loss = cross_entropy_loss_with_logits
	loss_grad = cross_entropy_loss_with_logits_grad

	val_set = mnist(val=True)
	def val():
		gen = val_set()
		val_sum = 0.0
		for i,data in enumerate(gen):
			input,label = data
			output = ff.forward(input)
			val_sum += np.argmax(output) == label
		print "Val", val_sum / float(i)


	optim = GradientDescentOptimizer(ff,lr=learning_rate)

	train_set = mnist()

	print "Training .."

	for epoch in xrange(epochs):
		loss_sum = 0.0
		gen = train_set()
		for i, data in enumerate(gen):
			input, label = data
			label = np.array(label,dtype=np.int32)
			output = ff.forward(input)
			ff.backward(loss_grad(label,output))

			if i > 0 and (i % batch_size == 0):
				optim.step()

			loss_sum += loss(label,output)
		print epoch, "Loss", loss_sum / i
		val()

if __name__ == "__main__":
	import argparse
	ap = argparse.ArgumentParser()
	ap.add_argument("--batch_size",type=int,default=128)
	ap.add_argument("--hidden_size",type=int,default=128)
	ap.add_argument("--epochs",type=int,default=1000)
	ap.add_argument("--learning_rate",type=float,default=0.001)


	args = vars(ap.parse_args())
	train(**args)
	