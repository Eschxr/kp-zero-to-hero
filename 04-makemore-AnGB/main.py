"""
MLP character-level language model
"""

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import random
from tqdm import tqdm


# String : Int and Int : String maps
stoi = {}
stoi['.'] = 0
for i in range(ord('a'), ord('z')+1):
    stoi[chr(i)] = i - ord('a') + 1

itos = {i:s for s, i in stoi.items()}
BLOCK_SIZE = 3  # character-level context length
VOCAB_SIZE = 27 # |V|

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


if __name__ == "__main__":
    words = open('names.txt', 'r').read().splitlines()

    # Hyperparameters
    n_embd = 10     # Dimensionality of character embeddings
    n_hidden = 200  # Number of neurons in the hidden layer
    n_batchsize = 32# Size of each minibatch for SGD
    iters = 200000 # Number of training iterations

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
    C = torch.randn((VOCAB_SIZE, n_embd), generator=g)   # Bengio et al. 2003 compressed 17000 words -> 30-d space, we shall do 27 chars -> 2-d space

    # Layer 1
    W1 = torch.randn((n_embd*BLOCK_SIZE, n_hidden), generator=g) * (5/3)/((n_embd*BLOCK_SIZE)**0.5)
    # b1 = torch.randn(n_hidden, generator=g) * 0.01

    # Layer 2
    W2 = torch.randn((n_hidden, VOCAB_SIZE), generator=g) * 0.01
    b2 = torch.randn(VOCAB_SIZE, generator=g) * 0

    # Batch normalization parameters
    bngain = torch.ones((1, n_hidden))
    bnbias = torch.zeros((1, n_hidden))
    bnmean_running = torch.zeros((1, n_hidden))
    bnstd_running = torch.ones((1, n_hidden))

    # Grouping parameters
    parameters = [C, W1, W2, b2, bngain, bnbias]
    print(f'Total params: {sum(p.nelement() for p in parameters)}')
    for p in parameters:
        p.requires_grad = True

    # Search learning rate hyperparam
    # lre = torch.linspace(-3, 0, 1000)   # Linearly generate exponent candidates
    # lrs = 10**lre   # Exponentially distributed (10^-3 -> 10^0) learning rates
    # lri, lossi = [], []
    lossi, stepi = [], []

    # Training
    for i in tqdm(range(iters), ascii=True, desc="Training Progress"):
        # Construct minibatch
        ix = torch.randint(0, Xtr.shape[0], (n_batchsize,))

        # Forward pass
        emb = C[Xtr[ix]]
        embcat = emb.view(emb.shape[0], -1)
        hpreact = embcat @ W1
        bnmeani = hpreact.mean(0, keepdim=True)
        bnstdi = hpreact.std(0, keepdim=True)
        hpreact = bngain * (hpreact - bnmeani) / bnstdi + bnbias

        # Batchnorm mean & stdev optimization
        with torch.no_grad():
            bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
            bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi

        h = torch.tanh(hpreact)
        logits = h @ W2 + b2
        loss = F.cross_entropy(logits, Ytr[ix])   # Previous manual calculation replaced since this is more optimized & well-behaved
        # print(f'Loss: {loss.item()}, Interval: {i}')

        # Backward pass
        for p in parameters:
            p.grad = None
        loss.backward()
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

    # Calibrate batchnorm post-train
    with torch.no_grad():
        emb = C[Xtr]
        embcat = emb.view(emb.shape[0], -1)
        hpreact = embcat @ W1
        bnmean = hpreact.mean(0, keepdim=True)
        bnstd = hpreact.std(0, keepdim=True)

    emb = C[Xtr]
    embcat = emb.view(emb.shape[0], -1)
    hpreact = embcat @ W1
    hpreact = bngain * (hpreact - bnmean_running) / bnstd_running + bnbias
    h = torch.tanh(hpreact)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr)   # Previous manual calculation replaced since this is more optimized & well-behaved
    print(f'Final Loss (Entire Training Set): {loss.item()}')

    emb = C[Xte]
    embcat = emb.view(emb.shape[0], -1)
    hpreact = embcat @ W1
    hpreact = bngain * (hpreact - bnmean_running) / bnstd_running + bnbias
    h = torch.tanh(hpreact)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Yte)   # Previous manual calculation replaced since this is more optimized & well-behaved
    print(f'Final Loss (Entire Test Set): {loss.item()}')


    # Sampling
    for _ in range(20):
        out = []
        context = [0 for _ in range(BLOCK_SIZE)]
        while True:
            emb = C[torch.tensor([context])]
            embcat = emb.view(1, -1)
            hpreact = embcat @ W1
            hpreact = bngain * (hpreact - bnmean_running) / bnstd_running + bnbias
            h = torch.tanh(hpreact)
            logits = h @ W2 + b2
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            context = context[1:] + [ix]
            if ix == 0:
                break
            out.append(ix)
        print(''.join(itos[i] for i in out))

    
    # Plot loss w.r.t. steps, learning rate, etc.
    plt.figure(figsize=(8, 8))
    plt.plot(stepi, lossi)
    plt.show()

    # Plot embeddings
    # plt.figure(figsize=(8, 8))
    # plt.scatter(C[:,0].data, C[:,1].data, s=200)
    # for i in range(C.shape[0]):
    #     plt.text(C[i,0].item(), C[i,1].item(), itos[i], ha="center", va="center", color="white")
    # plt.grid('minor')
    # plt.show()
