# Makemore

## Lesson #5: Becoming a Backprop Ninja

* So today we're going to manually backpropagate through the MLP we've built through the last few sessions (which apparently was the norm just over a decade ago)
* For the sake of practice (and history) LET'S DO IT


### Starter code

* Using Andrej's jupyter notebook; initialization is the exact same as before except biases are initialized as small numbers instead of 0 to enforce backprop correctness (0 can mask some errors)
* And so begins the long journey of manual backpropagation, oh boy


### Progress

#### logprobs
* Cross entropy is mathematically identical to negative log likelihood, so the first step is to calculate the derivative of the loss w.r.t. the log probabilities
* The loss is calculated using logprobs, a 32x27 tensor (batch size x output alphabet size) whereby we take the indices of the correct labels to find the probabilities assigned to the correct answers, and find the mean across all 32 examples in the batch
* In other words we find the mean of the probabilities assigned to the correct labels within logprobs
* The mean is simply the sum of every component divided by the count of components, which in this case is 32 because we have a batch size of 32
* So, our gradients for the components indexed by [range(n), Yb] are all -1/batch_size, and 0 for the non-participating components
* Because we can write the mean as (-1/n)a_1 + (-1/n)a_2 + ... + (-1/n)a_32 and the partial derivative of any of these a_i is simply the coefficient in front of them, which is always -1/n

