# Makemore

## Lesson #3: Turning Makemore into an MLP

Let us take what we've built, and push it even further.

### MLP Structure
* We can now introduce context length (since no longer constrained by 1-char context by the definition of a bigram), we set this to 3 by default to align with the video
* So we now preprocess the dataset of words by sliding a window of 3 characters and creating our X (input) and Y(label) tensors by sliding through each word
* The architecture of the MLP is based on Bengio et al. 2003, which I have not read but understand as far as explained in the video (which is not a lot tbf)
* First there's an embedding layer which compressed 17000 words -> 30-d space, we do something similar but with characters, so 27 characters -> 2-d space
* In other words, we have a 2d embedding for each character, and we have 3 characters in each input to the neural net, which makes 3x 2-d tensors that we concatenate to form the 6d input to the first hidden layer (I don't fully understand the reasoning behind concatenation but I'm assuming it's so that the hidden layer has a nice singular 6d input to consider instead of 3 2d tensors that we need to think of ways to combine
* Then we go from 6d input to 100d output, then the second layer goes from 100d input back to 27d output, which is nice because then we have 27 logits each corresponding to one of the characters, which we softmax to get probabilities. 

### Loss Evaluation
* We can evaluate the produced probability distribution using negative log likelihood, (mathematically identical to cross entropy) where we take the log of the probabilities (which takes on values from 0-1, meaning log(1) = 0 and log(0) asymptotically reaches -infinity), we use the negative because then we get a really nice curve that we can try to reduce down instead of increase up to 0
* This provides everything we need to run the training loop, but we wouldn't want to train on hundreds of thousands of sequences each run
* Which means it's the perfect time to introduce minibatches, where we just take random batches from the training set each time, significantly reducing the time it takes to train
* This does mean we don't always move along the gradient consistently (sometimes loss will jump up and down) but we will generally be moving down and moving much more quickly since generally even with a really small batch the gradient is "good enough"
* A really nice quote: It's much better to have an approximate gradient and take more steps, than to take the exact gradient and move less steps. 
* So in practice yes we use minibatches

### Learning Rate Hyperparameter Selection
* We can sort of arbitrarily guess good learning rates (in this case our initial pick of 0.1 is pretty darn good)
* However we can't always rely on being good at guessing so one way is to scan a list of learning rate candidates (here we selected exponentially spaced learning rates between 10^-3 to 10^0)
* High learning rates will explode and low rates will converge too slowly, if we track the stats (learning rate : loss) we can plot this nicely using matplotlib and eyeball a decent learning rate
* In practice people also like to implement learning rate decay, which is, like it sounds, reducing the learning rate as iterations go on, so that when we converge we're making more fine-grained adjustments to avoid plateuing

### Pitfalls
* Obviously we don't want a model that overfits; something that's memorized our training set verbatim isn't useful at predicting unseen data
* Which is why although we can have a neural net with millions of parameters, that achieves a loss of 0 on the training set, we're still not producing a good model because this fella is overfit and will probably perform poorly on unseen data
* So what we do is we split the dataset into train/val/test (80%/10%/10% roughly)
* The training set is used for gradient descent (obvy), the validation set is used to optimize hyperparameters (basically we evaluate performance on the validation set over sets of hyperparameters to find a good fit) and the test set is used to evaluate the model at the very end (use SPARINGLY because we don't want to leak the test set and defeat its purpose)
* If we evaluate a neural net on the test/val sets and find that accuracy remains on unseen data, congrats! We're not overfitting
* However in our case we may be underfitting, since we have a very small neural net (~3000 params) and the accuracy on the training vs test sets are not very different (which means we can expect pretty good performance gains by scaling up the neural net!)

### Improvements
* We know the ~3000 parameter neural net is underfitting, so let's increase the size of the hidden layer from 100 -> 300 parameters and see what happens!
* We now have ~10000 parameters and from a single "get your feet wet" test loss seems to be worse than before
* Increasing number of params in the hidden layer seems to not have changed much, so Karpathy suggests increasing the dimensions of the embedding layer, which is actually really cool when visualized in 2d (as it is right now) and we can see the model put similar characters (i.e: vowels) closer together and outliers (i.e: boundary marker, 'q', etc.) further away
* So we're really gonna multiply the number of embeddings by 5, that's fine by me; going from 2d->10d embeddings should produce pretty significant performance gains I'm expecting (or massive overfitting but hey that's also technically significant performance gain)
* By increasing embedding size and introducing rudimentary learning rate decay I reached ~2.19 loss on the test set with 100k training iterations. Karpathy set a challenge to beat his ~2.17 loss so that'll be the next section

### Challenge: Beat 2.17
* I'm gonna try doubling the batch size, and improving the learning rate decay (at around 10k-20k we get this thick loss fluctuation so that's probably where I'll start scaling learning rate exponentially w.r.t. iter count. I've currently got learning rate at 0.1 when iters <= 10000, and it scales exponentially w.r.t. (i/iters + 1) which should give a smooth curve down to a learning rate of 0.01. Batch sizes also doubled from 32 to 64, results will be added when I test this setup
* So loss is roughly 2.20, which means nothing really changed. I guess next up is changing the model architecture itself, and playing around with the embedding/hiddne layer, potentially also giving the model longer context

