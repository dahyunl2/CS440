"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    res = []
    cnt = {}
    tags = {}

    for sentence in train:
        for pair in sentence:
            word=pair[0]
            tag=pair[1]
            if tag not in tags:
                tags[tag] = 1
            else:
                tags[tag] += 1

            if word not in cnt:
                cnt[word] = {}
            if tag not in cnt[word]:
                cnt[word][tag] = 1
            else:
                cnt[word][tag] += 1

    max_ = max(tags.keys(), key=(lambda key: tags[key]))

    for sentence in test:
        pred = []
        for word in sentence:
            if word in cnt:
                graph = cnt[word]
                best = max(graph.keys(), key=(lambda key: graph[key]))
                pred.append((word, best))
            else:
                pred.append((word, max_))
        res.append(pred)

    return res