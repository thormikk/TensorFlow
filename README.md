# TensorFlow
(very) Simple TensorFlow examples to understand the underlying mechanics of the framework.

Many computationally expensive problems becomes cheaper when they are formulated in terms of a graph with elementary operations on
elementary functions. It turns out that calculating the derivatives w.r.t all the independent variables of some complex function
is a problem which benefits greatly from this observation, as noted by several authors in the 70s. This is especially handy in 
non-convex functional approximation because a first-order approximation (gradient descent) based on minimizing a loss-function with
respect to the parameters of the model requires the calculation of derivatives w.r.t several million parameters. The backpropagation
algorithm on neural networks is a simple example of this. Google's machine learning engineers understood that basically every supervised
learning problem can be formulated as an instance of a computation graph, which provides a common framework for a big class of machine 
learning problems. This framework became known as TensorFlow. To understand TensorFlow thoroughly, the author- possessing the self-
proclaimed title of "the dumbest smart guy in the world"- needs to go through several examples, gradually increasing in complexity. 
