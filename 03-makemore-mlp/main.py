"""
MLP character-level language model
"""

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import random


# String : Int and Int : String maps
stoi = {}
stoi['.'] = 0
for i in range(ord('a'), ord('z')+1):
    stoi[chr(i)] = i - ord('a') + 1

itos = {i:s for s, i in stoi.items()}


def build_dataset(words):
    # Construct dataset
    block_size = 3  # character-level context length
    X, Y = [], []
    for word in words:
        # print(word)
        context = [0] * block_size
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


if __name__ == "__main__":
    words = open('names.txt', 'r').read().splitlines()

    # Random train/val/test dataset splits
    random.seed(42)
    random.shuffle(words)
    n1 = int(0.8*len(words))
    n2 = int(0.9*len(words))

    Xtr, Ytr = build_dataset(words[:n1])
    Xdev, Ydev = build_dataset(words[n1:n2])
    Xte, Yte = build_dataset(words[n2:])

    """
    print(X.shape, X.dtype, Y.shape, Y.dtype)
    print(X)
    print(Y)
    """

    g = torch.Generator().manual_seed(42)

    # Neural net
    C = torch.randn((27, 2), generator=g)   # Bengio et al. 2003 compressed 17000 words -> 30-d space, we shall do 27 chars -> 2-d space

    # Layer 1
    W1 = torch.randn((6, 100), generator=g)
    b1 = torch.randn(100, generator=g)

    # Layer 2
    W2 = torch.randn((100, 27), generator=g)
    b2 = torch.randn(27, generator=g)
    parameters = [C, W1, b1, W2, b2]
    print(f'Total params: {sum(p.nelement() for p in parameters)}')
    for p in parameters:
        p.requires_grad = True

    # Search learning rate hyperparam
    # lre = torch.linspace(-3, 0, 1000)   # Linearly generate exponent candidates
    # lrs = 10**lre   # Exponentially distributed (10^-3 -> 10^0) learning rates
    # lri, lossi = [], []

    # Training
    for i in range(10000):
        # Construct minibatch
        ix = torch.randint(0, Xtr.shape[0], (32,))

        # Forward pass
        emb = C[Xtr[ix]]
        h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
        logits = h @ W2 + b2
        loss = F.cross_entropy(logits, Ytr[ix])   # Previous manual calculation replaced since this is more optimized & well-behaved
        # print(f'Loss: {loss.item()}, Interval: {i}')

        # Backward pass
        for p in parameters:
            p.grad = None
        loss.backward()
        lr = 0.1
        for p in parameters:
            p.data += -lr * p.grad

        # Learning rate stat tracker
        # lri.append(lre[i])
        # lossi.append(loss.item())


    emb = C[Xtr]
    h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr)   # Previous manual calculation replaced since this is more optimized & well-behaved
    print(f'Final Loss (Entire Training Set): {loss.item()}')

    emb = C[Xte]
    h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Yte)   # Previous manual calculation replaced since this is more optimized & well-behaved
    print(f'Final Loss (Entire Test Set): {loss.item()}')

    # Plot learning rates & losses
    # plt.figure(figsize=(16, 16))
    # plt.plot(lri, lossi)
    # plt.show()
