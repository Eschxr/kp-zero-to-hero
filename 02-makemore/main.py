import torch

if __name__ == "__main__":
    N = torch.zeros((28, 28), dtype=torch.int32)
    words = open('names.txt', 'r').read().splitlines()
    
    for word in words:
        chs = ['<S>'] + list(word) + ['<E>']
        for ch1, ch2 in zip(chs, chs[1:]):
            pass
            # use a string to int lookup (ord func) to index tensor and count freqs

