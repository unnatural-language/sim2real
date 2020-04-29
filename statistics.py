# Dataset profiling 
def main(): 
    synthetic_vocab = ['then', 'after', 'you', 'and', 'go', 'to', 'pick', 'up', 
                          'open', 'put', 'next', 'door', 'ball', 'box', 'key', 'on', 
                          'your', 'left', 'right', 'in', 'front', 'of', 'you', 'behind', 
                          'red', 'green', 'blue', 'purple', 'yellow', 'grey', 'the', 'a']
    
    synthetic_vocab = set(synthetic_vocab)

    fine_tune = open('fine_tune/annotations.csv', 'r').readlines()[1:]
    test = open('test/annotations.csv', 'r').readlines()[1:]
    
    all_examples = fine_tune + test 

    vocab = set()
    utterance_lengths = []

    for example in all_examples: 
        comps = example.split(',')
        sentence = comps[-1]
        words = sentence.split()
        for word in words: 
            vocab.add(word)
        utterance_lengths.append(len(sentence))

    avg_len = sum(utterance_lengths) / len(utterance_lengths)
    max_len = max(utterance_lengths)
    min_len = min(utterance_lengths)

    jaccard = len(synthetic_vocab.intersection(vocab)) / len(synthetic_vocab.union(vocab))
    

if __name__ == '__main__': 
    main()