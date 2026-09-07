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

#### probs

* Since logprobs = probs.log() and the derivative of ln(x) is 1/x, this one's really simple: just 1/probs * dlogprobs (chain rule)
* Realistically most of these operations are easily differentiable and then we just chain rule aka multiply on top of that. Sometimes when tensor dimensions change we do have to be careful about broadcasting/collapsing dimensions

#### counts, counts_sum, and counts_sum_inv

* Recall that probs = counts * counts_sum_inv and counts is a (32, 27) tensor while counts_sum_inv is (32, 1); differentiating the product = taking the other factor and multiplying it by the previous partial derivative, but in this case because the counts_sum_inv column is broadcasted it contributes multiple times to each row in counts_sum_inv, and once again when we have a node contributing multiple times we must sum all the gradients
* And since counts_sum_inv = counts_sum^-1.0, we can simply use the power rule and chain rule to get dcounts_sum = -counts_sum^-2.0 * dcounts_sum_inv
* Finally, we have dcounts, which has 2 contributing components: counts_sum = counts.sum() and probs = counts * counts_sum_inv; for the first component we know summing is equal to passing the gradient through (i.e, partial derivative is 1) and the product is the other factor times the previous gradient, so we have 1 * dcounts_sum + counts_sum_inv * dprobs

#### logits

* I will write this tomorrow
* For now, I want to note that dlogit_maxes is (almost) a zero tensor, meaning the gradients for (effect on the loss of) logit_maxes is nearly zero, which checks out because we use it as a constant to reduce all the logits such that exponentiating is well behaved, and it doesn't change any of the values in probs; this is good to know because even though we do a backward pass through this node, when everything is done correctly we still get the correct result that this thing has pretty much 0 impact
