"""
MLP character-level language model, torchified
"""

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import random
from tqdm import tqdm


words = open('names.txt', 'r').read().splitlines()

# String : Int and Int : String maps
stoi = {}
stoi['.'] = 0
for i in range(ord('a'), ord('z')+1):
    stoi[chr(i)] = i - ord('a') + 1

itos = {i:s for s, i in stoi.items()}
BLOCK_SIZE = 3  # character-level context length
VOCAB_SIZE = 27 # |V|

# Hyperparameters
n_embd = 10     # Dimensionality of character embeddings
n_hidden = 100  # Number of neurons in the hidden layer
n_batchsize = 32# Size of each minibatch for SGD
iters = 200000  # Number of training iterations

# Seed generators
random.seed(42)
g = torch.Generator().manual_seed(42)


class Linear:

    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out), generator=g) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d:

    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.gamma = torch.ones(dim)        # scaling factor
        self.beta = torch.zeros(dim)        # bias factor
        self.running_mean = torch.zeros(dim)# mean buffer
        self.running_var = torch.ones(dim)  # variance buffer

    def __call__(self, x):
        if self.training:
            xmean = x.mean(0, keepdim=True) # Current batch mean
            xvar = x.var(0, keepdim=True)   # Current batch variance
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x-xmean) / torch.sqrt(xvar+self.eps)    # Normalized input
        self.out = self.gamma * xhat + self.beta        # Scaled + bias
        if self.training:
            with torch.no_grad():
                self.running_mean = (1-self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1-self.momentum) * self.running_var + self.momentum * xvar
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    def parameters(self):
        return []


def build_dataset(words):
    # Construct dataset
    X, Y = [], []
    for word in words:
        # print(word)
        context = [0] * BLOCK_SIZE
        for c in word + '.':
            ix = stoi[c]
            X.append(context)
            Y.append(ix)
            # print(''.join(itos[i] for i in context), '-->', itos[ix])
            context = context[1:] + [ix]    # Slide window right & append
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    print(X.shape, Y.shape)
    return X, Y


# Train/dev/test splits
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))
Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])


if __name__ == "__main__":
    # Neural net
    C = torch.randn((VOCAB_SIZE, n_embd), generator=g)   # Bengio et al. 2003 compressed 17000 words -> 30-d space, we shall do 27 chars -> 2-d space

    layers = [
        Linear(n_embd * BLOCK_SIZE, n_hidden), Tanh(),
        Linear(           n_hidden, n_hidden), Tanh(),
        Linear(           n_hidden, n_hidden), Tanh(),
        Linear(           n_hidden, n_hidden), Tanh(),
        Linear(           n_hidden, n_hidden), Tanh(),
        Linear(           n_hidden, VOCAB_SIZE),
    ]

    with torch.no_grad():
        layers[-1].weight *= 0.1    # squash output layer
        for layer in layers[:-1]:   # Apply gain for other layers
            if isinstance(layer, Linear):
                layer.weight *= 5/3 # 5/3 as specified to be tanh optimum

    parameters = [C] + [p for layer in layers for p in layer.parameters()]
    print(f'Total params: {sum(p.nelement() for p in parameters)}')
    for p in parameters:
        p.requires_grad = True


    # Training
    lossi, stepi = [], []
    ud = []
    for i in tqdm(range(iters), ascii=True, desc="Training Progress"):
        # Construct minibatch
        ix = torch.randint(0, Xtr.shape[0], (n_batchsize,))
        Xb, Yb = Xtr[ix], Ytr[ix]

        # Forward pass
        emb = C[Xb]
        x = emb.view(emb.shape[0], -1)
        for layer in layers:
            x = layer(x)
        loss = F.cross_entropy(x, Yb)

        # Backward pass
        for layer in layers:
            layer.out.retain_grad() # Retain gradients of all outputs
        for p in parameters:
            p.grad = None
        loss.backward()

        # Update parameters
        if i <= 10000:  # Learning rate decay (yes ik not great code but it gets the job done)
            lr = 0.1
        else:
            lr = 0.1 ** (i / iters + 1)
        for p in parameters:
            p.data += -lr * p.grad

        # if (i+1) % 1000 == 0:
        #    print(f"Iteration {i+1} of {iters}")
        #    print(f"Learning rate: {lr}")

        # Learning rate stat tracker
        # lri.append(lre[i])
        stepi.append(i)
        lossi.append(loss.log10().item())
        with torch.no_grad():
            # learning rate adjusted gradient stdev : data stdev
            ud.append([(lr*p.grad.std() / p.data.std()).log10().item() for p in parameters])

        if i >= 1000: break   # Temporary single optimization step


    # Visualizations
    plt.figure(figsize=(20, 4))
    legends = []
    for i, layer in enumerate(layers[:-1]):
        if isinstance(layer, Tanh):
            t = layer.out
            print('layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%' % (i, layer.__class__.__name__, t.mean(), t.std(), (t.abs() > 0.97).float().mean()*100))
            hy, hx = torch.histogram(t, density=True)
            plt.plot(hx[:-1].detach(), hy.detach())
            legends.append(f'layer {i} ({layer.__class__.__name__})')
    plt.legend(legends)
    plt.title('activation distribution')

    plt.figure(figsize=(20, 4))
    legends = []
    for i, layer in enumerate(layers[:-1]):
        if isinstance(layer, Tanh):
            t = layer.out.grad
            print('layer %d (%10s): mean %+f, std %e' % (i, layer.__class__.__name__, t.mean(), t.std()))
            hy, hx = torch.histogram(t, density=True)
            plt.plot(hx[:-1].detach(), hy.detach())
            legends.append(f'layer {i} ({layer.__class__.__name__})')
    plt.legend(legends)
    plt.title('gradient distribution')

    # visualize histograms
    plt.figure(figsize=(20, 4)) # width and height of the plot
    legends = []
    for i,p in enumerate(parameters):
        t = p.grad
        if p.ndim == 2:
            print('weight %10s | mean %+f | std %e | grad:data ratio %e' % (tuple(p.shape), t.mean(), t.std(), t.std() / p.std()))
            hy, hx = torch.histogram(t, density=True)
            plt.plot(hx[:-1].detach(), hy.detach())
            legends.append(f'{i} {tuple(p.shape)}')
    plt.legend(legends)
    plt.title('weights gradient distribution')

    plt.figure(figsize=(20, 4))
    legends = []
    for i,p in enumerate(parameters):
        if p.ndim == 2:
            plt.plot([ud[j][i] for j in range(len(ud))])
            legends.append('param %d' % i)
    plt.plot([0, len(ud)], [-3, -3], 'k') # these ratios should be ~1e-3, indicate on plot
    plt.legend(legends);

    plt.show()
