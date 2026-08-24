"""
This script constructs all possible bigrams from `names.txt` and stores counts
in a Tensor, which is then normalized and sampled from, forming a simple baseline
character-level bigram model, which is then assessed (only on one word for now)
using the negative log likelihood loss function
"""

import torch
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


if __name__ == "__main__":
    N = torch.zeros((27, 27), dtype=torch.int32)
    words = open('names.txt', 'r').read().splitlines()
    
    stoi = {}
    stoi['.'] = 0
    for i in range(ord('a'), ord('z')+1):
        stoi[chr(i)] = i - ord('a') + 1

    itos = {i:s for s, i in stoi.items()}
    
    for word in words:
        chs = ['.'] + list(word) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1, ix2 = stoi[ch1], stoi[ch2]
            N[ix1, ix2] += 1

    """
    plt.figure(figsize=(16,16))
    plt.imshow(N, cmap='Blues')
    for i in range(27):
        for j in range(27):
            chstr = itos[i] + itos[j]
            plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
            plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
    plt.axis('off')
    plt.show()
    """

    P = (N+1).float()
    P /= P.sum(1, keepdim=True)

    g = torch.Generator().manual_seed(42)
    
    for i in range(20):
        out = ""
        ix = 0
        while True:
            p = P[ix]
            # p = torch.ones(27) / 27.0     # 'untrained' model (uniform distribution)
            ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
            if ix == 0:
                break
            out += itos[ix]
        print(out)


    # Negative log likelihood loss (tested on 'charles')
    log_likelihood = 0.0
    n = 0
    word = ".charles."
    for ch1, ch2 in zip(word, word[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob   # log(a*b) = log(a) + log(b)
        n += 1
        print(f'{ch1}{ch2}: {prob:.4f} {logprob:.4f}')

    print(f'{log_likelihood=}')
    nll = -log_likelihood
    print(f'{nll=}')
    print(f'{nll/n}')
