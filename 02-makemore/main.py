if __name__ == "__main__":
    words = open('names.txt', 'r').read().splitlines()
    print(words[:10])
   
    # view bigrams of first word
    for ch1, ch2 in zip(words[0], words[0][1:]):
        print(ch1, ch2)

