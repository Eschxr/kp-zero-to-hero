# Makemore

### Lesson #2: Intro to language modeling

* Makemore takes lines of text and produces more "things" akin to what it's seen
* Apparently it's an "autoregressive character-level language model" (whatever that means) which I'll have to figure out down the line
* names.txt is a dataset of human names compiled by Andrej Karpathy, which'll be the main target for makemore
* so "character-level" just means the model treats each "item" as sequences of individual characters
* and eventually it learns to predict the next character in a sequence
* now I just gotta figure out what "autoregressive" means

### Bigram Language Model

* Only concerned with relationship between 2 chars at a time (hence, *bi*-gram)
* Given one character, the model tries to predict the next character in the sequence
* Simple & easy to understand, makes a great starting point but only captures local patterns
* i.e: fails to capture patterns between words
* A really simple implementation is to simply count the frequencies of each bigram, then normalize the counts and sample from them as a probability distribution

### Negative Log Likelihood Loss

* A useful way to measure loss by taking the negative log of the predicted probabilities for 'correct' bigrams
* Since probabilities lie between 0 and 1, lower probabilities take on more negative log values, which is why we use the negative log instead (since we want to minimize loss)
* Typically a model's performance is modelled by the average negative log likelihood loss of each correct label. 

### Neural Network Implementation

* Neural networks ideally work with tensors containing floats, so we need a way to encode frequencies (ints) of bigrams as tensors of floats.
* One such way to do this is using one-hot encoding:
    * Let's say we have an integer 5, one-hot encoding turns this into an nd tensor where the 5th dimension has value 1.0 (in PyTorch, the nn.functional.one_hot() method does not support dtype assignments, so we need to cast explicitly)
    * This way we can separate bigrams into input and output chars (xs and ys) where each is encoded using one-hot encoding
* With one-hot encoded inputs, we can make a simple linear single-layer neural network starting with randomized weights.
* To turn inputs * randomized weights (really in this case, since one-hot encoded inputs are just vectors of 0s with a single 1, the dot product turns out to be just the nth row of the weights) into probabilities, we use softmax, which takes logits produced from x @ W, exponentiates them, and normalizes the result
* We can then sample from the resulting distribution all the same. 
* Training/gradient descent turns out to be the exact same as before, since all that's needed is to zerograd and run loss.backward() after the forward pass, then adjust W.data by some learning rate * W.grad
* This method produces a model that, after a few rounds of training, ends up encoding the exact same matrix as the one obtained from simply counting (actually, something close to the log of that matrix). 
* When we counted frequencies, we could normalize the distribution by adding a small epsilon to each count (larger epsilon -> approach uniform distribution). We can do something similar here by adding a small constant times the mean of W^2, which pulls all the weights closer to 0 (if everything is 0 we get uniform distribution). 
* The forward pass of this neural net, particularly exponentiation & normalization, is what's called "SoftMax".
* The model produced by the gradient/neural net method behaves almost identically to the naive counting-based model. However this version is far more flexible and as we complexify it will begin to distinguish itself.

