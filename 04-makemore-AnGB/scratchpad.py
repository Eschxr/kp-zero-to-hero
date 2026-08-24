# A scratchpad for throwaway code


"""
This section contains the initial example of exploring the
names.txt dataset, and reading the bigram frequencies into
a python dictionary
"""
if __name__ == "__main__":
    words = open('names.txt', 'r').read().splitlines()
    # print(words[:10])
    bigram_dict = {}                            # dictionary storing bigram frequencies

    # Get bigram freqs of entire dataset, stored in a python dict
    for word in words:
        chars = ['<S>'] + list(word) + ['<E>']  # add special start & end chars
        for ch1, ch2 in zip(chars, chars[1:]):
            bigram = (ch1, ch2)
            bigram_dict[bigram] = bigram_dict.get(bigram, 0) + 1
            # print(ch1, ch2)
    
    bigram_dict = sorted(bigram_dict.items(), key = lambda kv: kv[1], reverse = True)
    print(bigram_dict)

