import numpy as np

class GradientDescentOptimizer(object):
	def __init__(self,sequence,lr=0.001):
		self.lr = lr
		self.sequence = sequence

	def step(self):
		for m in self.sequence:
			for name,data in m.params.iteritems():
				m.add_param(name,-1 * m.grads[name] * self.lr)
			m.zero_grads()
