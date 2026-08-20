"""
MLP character-level language model
"""

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


if __name__ == "__main__":
    words = open('names.txt', 'r').read().splitlines()

    # String : Int and Int : String maps
    stoi = {}
    stoi['.'] = 0
    for i in range(ord('a'), ord('z')+1):
        stoi[chr(i)] = i - ord('a') + 1

    itos = {i:s for s, i in stoi.items()}

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
    # print(sum(p.nelement() for p in parameters))
    for p in parameters:
        p.requires_grad = True

    # Training
    for i in range(100):
        # Forward pass
        emb = C[X]
        h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
        logits = h @ W2 + b2
        loss = F.cross_entropy(logits, Y)   # Previous manual calculation replaced since this is more optimized & well-behaved
        print(f'Loss: {loss.item()}, Interval: {i}')

        # Backward pass
        for p in parameters:
            p.grad = None
        loss.backward()
        for p in parameters:
            p.data += -0.1 * p.grad
