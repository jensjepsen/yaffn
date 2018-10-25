# Yet Another Feed Forward Neural Network
A small modular feed forward network in python, using only numpy.
It has no real purpose, since it's neither fast nor efficient - just for fun. 


## Classify MNIST digits
To use this to classify MNIST digits, first download the MNIST data:
```
./get_mnist.sh
```
which will store the data in the folder `mnist_data`.

Then run 
```
python train.py
```
to train the network with default parameters.

The network parameters are optimized by minimizing the categorical cross-entropy between the network outputs and the ground truth, using vanilla SGD.

Usage: 
```
python main.py --help

usage: main.py [-h] [--batch_size BATCH_SIZE] [--hidden_size HIDDEN_SIZE]
               [--epochs EPOCHS] [--learning_rate LEARNING_RATE]

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
  --hidden_size HIDDEN_SIZE
  --epochs EPOCHS
  --learning_rate LEARNING_RATE
```