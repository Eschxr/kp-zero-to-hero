# Makemore

## Lesson #4: Activations & Gradients, BatchNorm

* So we're going deeper into the activations & how gradients flow backward to build an intuitive understanding so as to understand the history of the development of neural network architectures; sounds reasonable enough to me
* Figured this was interesting so I'd jot it down: recurrent neural networks (RNNs) are formally proven to be a "universal approximator" in that, in principle, they can approximate any continuous function to any arbitrary degree of accuracy
* However they are not easily optimizable (and according to Andrej we will see why later when activations & gradient behavior is covered in more depth

### Initial loss

* As configured, the neural net currently starts with extremely high loss (~20.0), which is bad because with a uniform distribution the loss is -ln(1/27.0) ~ 3.296
* We currently initialize with a normal distribution where weights can take very high & very low values, whereas ideally we would want everything to be somewhat close to 0
* An easy way to fix this is to set the output layer's biases to 0 and scale down the weights by a tiny number (currently factor of 0.01)
* We don't actually want to set the weights to 0 (will find out why later)
* Keeping a "small amount of entropy" by scaling down to tiny numbers instead of setting weights to 0 breaks symmetry (I will also find out what this means later)
* By performing this simple modification I was able to reach roughly the same loss with 1/5 (200,000) of the original training iterations (2.04/2.12 train/test split original; 2.07/2.13 new)
* And this is because we no longer need to spend initial training iterations on squashing the weights (already done), so that more work can be spent on actual hard training; on the graph this looks like the "hockey stick" is gone

