import numpy as np
from helpers import init

class Module(object):
    """
        A super-class for all NN modules,
        that holds parameters and accumulated gradients
    """
	def __init__(self):
		self.params = {}
		self.grads = {}

	def param(self,name,data=None):
		if data is None:
			return self.params[name]
		else:
			self.params[name] = data
			self.grads[name] = np.zeros_like(data)

	def add_grad(self,name,data):
		self.grads[name] += data

	def zero_grads(self):
		for grad in self.grads.values():
			grad.fill(0.0)
			
	def add_param(self,name,data):
		self.params[name] += data

	def forward(self,input):
	    """
	        This method should be overridden by subclasses,
	        to define the forward pass of the subclass.
	    """
		raise RuntimeError("All subclasses of Module must implement a forward method")
	
	def backward(self,input,grads):
	    """
	        This method should be overridden by subclasses,
	        to define the backward pass of the subclass,
	        i.e. calculate the gradients.
	    """
		raise RuntimeError("All subclasses of Module must implement a forward method")


class Linear(Module):
	def __init__(self,input_size,output_size):
		super(Linear,self).__init__()
		self.param('weights', init(input_size,output_size))
		self.param('bias', np.zeros(output_size))

	def forward(self,input):
		return np.dot(input,self.param('weights')) + self.param('bias')

	def backward(self,input,grads):
		self.add_grad('bias',grads.copy())
		self.add_grad('weights',np.outer(input,grads))
		return np.dot(grads,self.param('weights').T)

class ReLU(Module):
	def __init__(self):
		super(ReLU,self).__init__()

	def forward(self,input):
		return (input > 0.0) * input

	def backward(self,input,grads):
		return grads * (input > 0.0) * 1.0

class L2Loss(Module):
	def __init__(self):
		super(L2Loss,self).__init__()

	def forward(self,input):
		return ((input[0] - input[1]) ** 2).sum() / 2.0

	def backward(self,input,grads = None):
		return (input[0] - input[1])

class Sequence(object):
    """
        This class sequentially applies a number of Modules,
        and handles backpropagation through those Modules.
    """
	def __init__(self,*modules):
		self.modules = modules

	def __getitem__(self,idx):
		return self.modules[idx]

	def params(self):
		return dict([(name + "_{}".format(i),params) for i,m in enumerate(self.modules) for name,params in m.params.items()])

	def grads(self):
		return dict([(name + "_{}".format(i),grad) for i,m in enumerate(self.modules) for name,grad in m.grads.items()])

	def forward(self,input,save_inputs = True):
		if save_inputs:
			self.inputs = []
		for module in self.modules:
			if save_inputs:
				self.inputs.append(input)
			input = module.forward(input)
		return input

	def backward(self,grads):
		for input,module in zip(self.inputs,self.modules)[::-1]:
			grads = module.backward(input,grads)

	def __iter__(self):
		return (m for m in self.modules)