# ConvexityAnnealing
Graduated optimization on neural networks via adjustment of activation functions.

The only paper I could find on using graduated optimization in neural networks is [[arXiv:1503.03712]](https://arxiv.org/abs/1503.03712), which uses a quite complicated algorithm to smooth the loss landscape using random sampling, then iteratively relaxes the smoothing process until the network has returned to its fully nonlinear form.

Here, I'll try to achieve the same result, but instead of smoothing via sampling, we'll start from a convex optimization problem (neural network with linear activation functions), and gradually make the problem more nonconvex by changing the activation function so that it looks more like a ReLU.

The basic idea is that when the activation function is the identity, the loss function of any neural network, regardless of depth, becomes convex since the entire network reduces to a chain of matrix multiplications, which itself can be represented as a single matrix, and so the network is simply a linear model. This is easily optimized, as convex functions have been extensively studied and well-known techniques for their optimization have been established. Once this initially linear model has been sufficiently optimized, we tweak the activation functions to produce a small nonlinearity, and repeat the optimization process. Then we make the activation slightly more nonlinear, then optimize again. This series of steps repeats until the activation function has reached its fully nonlinear form.

The specific nonlinearity that we will introduce is a modified leaky ReLU activation, where the slope of the x<0 branch is controlled by a parameter θ. Specifically, the activation is a(x) = max(x tanθ, x). Initially, θ=π/4, which causes the activation to reduce to the identity. But over the course of training, θ is annealed to zero, restoring the canonical ReLU activation.

This approach has the advantage that the modification to the network is almost trivial, but has the potential to be immensely helpful during training. The initially linear activation allows gradients to propagate easily down to the lowest layers of the network during the initial epochs, and smooths the loss landscape so that larger initial learning rates can be used.

## TODO:
* Fix workflow so that empty experiment directories aren't generated each time a config file is run
    * Put directory-generating code in train.py so new directory is only generated at training time if needed
* Perform experiments
    1) Fix θ=0 (normal ReLU) for baseline comparison
    2) Start at θ=π/4 (linear activation), and slowly reduce to θ=0 over fixed period
    3) Make θ a trainable parameter (shared between all activation functions), with sufficiently small learning rate that convexity of the loss landscape does not change drastically
    4) Make θ a trainable parameter, but with each neuron having its own independent activation.

## Done:
* Make a script that takes the log data and makes plots
    * Make a Youden plots for comparing the different experiments?
        * Different theta schedules can be different colors
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
