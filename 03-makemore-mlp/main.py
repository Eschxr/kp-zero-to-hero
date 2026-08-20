"""
WIP neural net version of the bigram language model
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

    # Construct training set
    xs, ys = [], []
    for word in words:
        chs = ['.'] + list(word) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1, ix2 = stoi[ch1], stoi[ch2]
            xs.append(ix1)
            ys.append(ix2)
    xs = torch.tensor(xs)
    ys = torch.tensor(ys)
    num = xs.nelement()
    print(f'Number of examples: {num}')
    xenc = F.one_hot(xs, num_classes=27).float()    # One-hot encoding (int -> Tensor
    yenc = F.one_hot(ys, num_classes=27).float()    # | Tensor[int] = 1, else = 0)

    """
    plt.figure(figsize=(16,16))
    plt.imshow(xenc, cmap='Blues')
    plt.axis('off')
    plt.show()
    """

    # Neural Net
    g = torch.Generator().manual_seed(42)
    W = torch.randn((27, 27), generator=g, requires_grad=True)   # Randomized starting weights

    # Gradient descent
    for k in range(100):
        # Forward pass
        xenc = F.one_hot(xs, num_classes=27).float()
        logits = xenc @ W           # Log-counts produced via matmul
        counts = logits.exp()       # Exponentiate, now positive (equal to N)
        probs = counts / counts.sum(1, keepdim=True) # Normalized counts (THIS IS SOFTMAX BTW)
        loss = -probs[torch.arange(num), ys].log().mean() + 0.01*(W**2).mean() # The second component regularizes W, which is equivalent to when we added 1 to N to make the distribution approach uniform
        print(f'Loss={loss.item()}, iter={k}')

        # Backward pass
        W.grad = None   # Zerograd
        loss.backward()
        W.data += -50 * W.grad  # Update

    # Sampling
    for i in range(20):
        out = ""
        ix = 0
        while True:
            xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
            logits = xenc @ W
            counts = logits.exp()
            p = counts / counts.sum(1, keepdim=True)
            ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
            if ix == 0:
                break
            out += itos[ix]
        print(out)
