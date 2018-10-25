import numpy as np

def init(input_size,output_size):
	std = np.sqrt(2.0 / (input_size + output_size))
	return np.random.uniform(-std,std,size=(input_size,output_size))