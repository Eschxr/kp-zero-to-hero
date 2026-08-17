import torch
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


if __name__ == "__main__":
    N = torch.zeros((28, 28), dtype=torch.int32)
    words = open('names.txt', 'r').read().splitlines()
    
    stoi = {}
    for i in range(ord('a'), ord('z')+1):
        stoi[chr(i)] = i - ord('a')
    stoi['<S>'] = 26
    stoi['<E>'] = 27
    
    for word in words:
        chs = ['<S>'] + list(word) + ['<E>']
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1, ix2 = stoi[ch1], stoi[ch2]
            N[ix1, ix2] += 1

    plt.imshow(N, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.show()

