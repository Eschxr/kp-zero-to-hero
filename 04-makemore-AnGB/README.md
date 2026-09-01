# Makemore

## Lesson #4: Activations & Gradients, BatchNorm

* So we're going deeper into the activations & how gradients flow backward to build an intuitive understanding so as to understand the history of the development of neural network architectures; sounds reasonable enough to me
* Figured this was interesting so I'd jot it down: recurrent neural networks (RNNs) are formally proven to be a "universal approximator" in that, in principle, they can approximate any continuous function to any arbitrary degree of accuracy
* However they are not easily optimizable (and according to Andrej we will see why later when activations & gradient behavior is covered in more depth

## Fixing Initialization

### Initial loss

* As configured, the neural net currently starts with extremely high loss (~20.0), which is bad because with a uniform distribution the loss is -ln(1/27.0) ~ 3.296
* We currently initialize with a normal distribution where weights can take very high & very low values, whereas ideally we would want everything to be somewhat close to 0
* An easy way to fix this is to set the output layer's biases to 0 and scale down the weights by a tiny number (currently factor of 0.01)
* We don't actually want to set the weights to 0 (will find out why later)
* Keeping a "small amount of entropy" by scaling down to tiny numbers instead of setting weights to 0 breaks symmetry (I will also find out what this means later)
* By performing this simple modification I was able to reach roughly the same loss with 1/5 (200,000) of the original training iterations (2.04/2.12 train/test split original; 2.07/2.13 new)
* And this is because we no longer need to spend initial training iterations on squashing the weights (already done), so that more work can be spent on actual hard training; on the graph this looks like the "hockey stick" is gone

### Saturated Hyperbolic Tangent

* Looking at the values in the hidden layer, there's a disproportionate amount of -1s and 1s (extremes of the tanh function) because the pre-activation values (i.e: input embeddings @ W1 + b1) take on a very wide range of values, so when tanh squashes the values into a range of -1 to 1 most values sit on the extremes
* The backward pass of tanh propagates the gradient using the formula (1-t^2) * out.grad, and because 1^2 = (-1)^2 = 1 we can easily see that having many values at the extremes causes the gradient to effectively be "blocked" (because it gets multiplied by 0)
* Intuitively this makes sense because extreme values sit in the flat "tail" region of the tanh, which has a derivative close to 0 (because changing the value still leaves you sitting on the flat "tail")
* And we really don't want our gradients to vanish from this tanh "obstacle"
* To understand the behavior of tanh:
    * At 0, we see that the gradient effectively becomes 1 * out.grad, meaning the tanh is "inactive" and things just pass through
    * And the further we are from 0 (closer to the tails), the more active tanh becomes, squashing the values inward
* So in conclusion: yes this is a massive problem because for a very large set of values our gradient gets destroyed at the tanh layer, and if we ever had a neuron of which all inputs result in values sitting in the flat tail region of tanh, that neuron effectively never learns (dead neuron)
* This issue can also happen with other activation functions
* Simplest example being ReLU (f(x) = max(0, x)) because if we had a neuron in which all values passing through sit below 0, the gradient is literally set to 0 when we backprop through ReLU, and this neuron will never ever learn

### Activation Functions & "Brain Damage" of a Neural Network
* For all activation functions, there are regions in which the derivative is close to (or sometimes exactly) 0, and by the definition of backpropagation (chain rule being multiplication) this shuts down the gradient from passing through.
* When this happens for all inputs to a neuron, it is effectively dead as it can never learn when no gradients ever pass through
* Some examples of this occuring are:
    * Hyperbolic tangent for values sitting in the flat tail region (> 0.99 | < -0.99)
    * ReLU for values sitting in the deactivated region (<=0)
* This problem can occur during both initialization AND optimization
    * During initialization, if we get unlucky and receive weights that, for a particular set of neurons, produce outputs that always sit in "undesirable" regions of the activation function
    * During optimization, we may knock a neuron off into dead regions if we had a high enough learning rate and/or an extreme gradient happens to pass through, after which the neuron never activates again (because we've adjusted its value too much); I like the name "brain damage" because that's literally what this is: we smacked the neural net on the forehead and now parts of it can't learn anymore

### Fixing the oversaturated tanh
* So we fix this much like how we fixed the previous problem: by squashing the weights and biases closer to 0
* For the biases, setting them to exactly 0 is ok, but Andrej says keeping a bit of entropy can sometimes be helpful in training
* For the weights, once again we don't want them to be exactly 0, so we just squash them
* By adding these simple fixes (squashing init values) we've already surpassed the inital setup (w.r.t. loss) even with only 1/5 the training iterations (2.05/2.11)

### Conclusion
* We were able to see quite substantial gains just from simple fixes to neural net initialization
* While for our current, tiny, single-layer MLP the deficit was rather forgiving, this is not true in the general case (especially with extremely large networks)
* Generally speaking, the deeper a neural network is, the less forgiving it will be towards bad initialization
* This is because the issues we've seen (gradient blocking, softmax being confidently wrong) add up through layers (we had probably >90% of inputs resulting in dead tanh activations, imagine if we had 10 layers, each with this issue, by the end of it our neural network is probably not going to learn at all)
* So, being aware of the internal behavior of activation functions and gradients can help us understand and mitigate these issues (and prevent us from producing braindead neural networks or inflicting permanent brain damage)
* The next natural question is, we've scaled down the randomized initial parameters, but how do we come up with the scaling values themselves? This is especially problematic with larger neural nets as I can't imagine it being fun to manually test hundreds of hyperparameters (the combinations of which grow exponentially)

## Optimal Initialization of Neural Networks

### Motivations
* See above (obviously)
* In all seriousness, generally we want to initialize our neural network such that all layers show a standard gaussian normal distribution, which means that we want not only the mean to be similar, but also the standard deviation (so that our distribution does not scale up/down or fluctuate as we move between layers)
* A mathematical formula has been devised for matrix multiplications (disregarding activation/nonlinearities) in which we divide the weights of the layer by the square root of the number of input elements to the layer and this seems to preserve the stdev of the network
* And so for MLPs we want to make sure that we initialize the weights such that the network behaves reasonably through the layers (i.e: does not expand to infinity or shrink to 0; importantly, we want to make sure this "nice behavior" also persists through backprop)

### Kaiming Initialization
* Arxiv 1502.01852
* Studied initialization of neural nets, and derived (I think) the sqrt(1/nl) formula (in their case, sqrt(2/nl) because factor 2 was needed to compensate for the lost half of the values from ReLU)
* Showed that if the network is initialized such that the forward pass behaves nicely, the backward pass also behaves nicely to *some extent* (there's a mathematical expression but I'm not putting it here since it won't be understandable without the context of the paper)
* Kaiming initialization is actually implemented in PyTorch and is (according to Andrej) "probably the most common way to initialize neural networks nowadays"
* Apparently initialization is not as troublesome as it used to be (~7 years ago when the paper was written) and we have modern innovations that make sure things behave nicely, so we don't need everything initialized *exactly right*

## Batch Normalization

### Description
* One of the *modern innovations* that made it so we could train neural nets effectiely w/o perfect initialization
* When dealing with the tanh problem, we wanted the pre-activation output to be somewhat of a regular gaussian normal distribution; batch normalization says: "Okay, why not just normalize them to be Gaussian?" and it just *works*
* So basically, if we have an intermediate state, we can just normalize it to be unit Gaussian
* This is implemented by taking the intermediate state (in our case, `hpreact`) and subtracting the mean from it, then dividing by the standard deviation
* This will make it so the signals firing from the neurons for the current batch will be normalized to be unit Gaussian, hence the name *Batch Normalization*

### Implementation Details

* However we don't want to always force this distribution upon every batch, only upon normalization; afterwards we want backprop to nudge this in the right direction
* We can make batch normalization itself trainable by adding 2 new sets of parameters: bngain and bnbias, which multiplies to & adds to the normalized intermediate state
* These parameters start with 1s and 0s respectively such that upon initialization we normalize to unit Gaussian, and then backprop allows the network to do whatever it wants with these new params

### Outcomes

* Once again implementing this wouldn't lead to massive improvements for our tiny neural network (we were able to normalize initialization quite easily w/o batch normalization) but with more layers, it'll quickly become intractable to manage the scales of the weight matrices to ensure a unit gaussian distribution; so, another improvement for larger neural nets: sprinkle batch normalization throughout the layers instead of manually scaling the weights
* Basically a shortcut for us to get stable training without needing a bunch of manual mathematics
* Batch normalization has a side effect: because we are normalizing *across the batch*, the batch itself and what it contains becomes mathematically entangled with both the forward and backward passes
* So now we get a *jitter* based on the samples of each batch (since calculating mean and stdev get affected by what the sample is) that affects the logits
* Which is actually a **feature** not a bug
* The entropy introduced from this actually helps prevent the model overfit, a *regularizer*
* Being the first of its kind, there have been many who dislike the entanglement property of batch normalization and have wanted to deprecate its use to move onto better normalization alternatives
* However the regularization side effect has made this process harder because batch normalization just works and happens to work *quite well*
* But now we have another problem: because we need the mean and stdev of a batch as inputs to the neural net, we can't sample a single word because the network *expects* batches as input
* One way to fix this is to calculate the mean and stdev of the training set after training completes, and use them as constant tensors
* Perhaps a better way to do this without having to manually calculate the mean & stdev post-training is to keep a running mean & stdev and adjust them while training (but not with pytorch/gradients)
* Another thing to note: batch normalization (subtracting the mean) removes any effect of the biases of the linear layer, so we can micro-optimize by removing the biases (which won't have any effect, since the responsibility has already been shifted to the batchnorm biases)
* ANOTHER thing to note (oh my days): the way Karpathy does stochastic gradient descent here is unusual -- (human em-dash) randomly sampling batches from the dataset in each training iteration. The standard way afaik is to shuffle & split the dataset into batches at the start of each epoch so as to guarantee that after every epoch the entire dataset passes through our neural net (making the training process more tractable than random sampling at each iter)

