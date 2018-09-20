# ConvexityAnnealing
Graduated optimization on neural networks via adjustment of activation functions.

The only paper I could find on using graduated optimization in neural networks is [[arXiv:1503.03712]](https://arxiv.org/abs/1503.03712), which uses a quite complicated algorithm to smooth the loss landscape using random sampling, then iteratively relaxes the smoothing process until the network has returned to its fully nonlinear form.

Here, I'll try to achieve the same result, but instead of smoothing via sampling, we'll start from a convex optimization problem (neural network with linear activation functions), and gradually make the problem more nonconvex by changing the activation function so that it looks more like a ReLU.

## TODO:
* Make a script that takes the log data and makes plots
* Perform experiments
    1) Fix θ=0 (normal ReLU) for baseline comparison
    2) Start at θ=π/4 (linear activation), and slowly reduce to θ=0 over fixed period
    3) Make θ a trainable parameter (shared between all activation functions), with sufficiently small learning rate that convexity of the loss landscape does not change drastically
    4) Make θ a trainable parameter, but with each neuron having its own independent activation.

## Done:
* Make configuration files for each experiment beforehand, then write a bash script to iterate through training procedure for each of them
* Add max pooling between conv layers
* Add regularization! (L1 and L2)
* Add precision, recall and F1 scores as metrics
* Make pipeline for collection of data from training process
	* Record time, epoch, train/val loss/accuracy, value of theta
	* Use tensorboard to view data during training
* Make neural network
	* Simple CNN with fixed architecture (possibly try different architectures?)
	* Provides input for modifying activation function during training (will train on MNIST for ease of gathering data)
		* Activations are PReLU with potentially trainiable slope α=tan(θ)
