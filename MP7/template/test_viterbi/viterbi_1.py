"""
Part 2: This is the simplest ver of viterbi that doesn't do anything spec for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""

import math
from collections import defaultdict, Counter
from math import log


# Note: remember to use these two elements when you find a probability is 0 in the training data.
epsilon_for_pt = 1e-5
emit_epsilon = 1e-5   # exact setting seems to have little or no effect


def training(sentences):
    """
    Computes init tags, emis words and trans tag-to-tag probabilities
    :param sentences:
    :return: intit tag probs, emis words given tag probs, trans of tags to tags probs
    """
    # init_prob = defaultdict(lambda: 0) # {init tag: #}
    # emit_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag: {word: # }}
    # trans_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag0:{tag1: # }}
    
    # TODO: (I)
    # Input the training set, output the formatted probabilities according to data statistics.
    
    init_prob={}

    trans_prob_count = {}
    for sentence in sentences:
        for word in range(len(sentence) - 1):
            if word==0:
                init_prob[sentence[word][1]]=1+init_prob.get(sentence[word][1], 0)
            tags = (sentence[word][1], sentence[word + 1][1])
            if tags not in trans_prob_count:
                trans_prob_count[tags] = 1
            else:
                trans_prob_count[tags] += 1
    trans_total = sum(trans_prob_count.values())
    trans_prob={}
    for i in trans_prob_count:
        # print(i)
        if i[0] not in trans_prob:
            trans_prob[i[0]]={}
        trans_prob[i[0]][i[1]]= (trans_prob_count[i] + epsilon_for_pt) / (trans_total + epsilon_for_pt * (len(trans_prob_count) + 1))
    # trans_prob['t1']={'t2':(epsilon_for_pt) / (trans_total + epsilon_for_pt * (len(trans_prob_count) + 1))}
    for tag in init_prob:
        init_prob[tag]=init_prob[tag]/len(sentences)

    emit_prob_count = {}
    for sentence in sentences:
        for pair in sentence:
            if pair[1] not in emit_prob_count:
                emit_prob_count[pair[1]] = {}
            if pair[0] not in emit_prob_count[pair[1]]:
                emit_prob_count[pair[1]][pair[0]] = 1
            else:
                emit_prob_count[pair[1]][pair[0]] += 1

    emit_prob={}
    words=[]
    for tag in emit_prob_count:
        total = sum(emit_prob_count[tag].values())
        emit_prob[tag]={}
        for word in emit_prob_count[tag]:
            words.append(word)
            emit_prob[tag][word]= (emit_prob_count[tag][word] + epsilon_for_pt) / (total + epsilon_for_pt * (len(emit_prob_count[tag]) + 1))
        emit_prob[tag]['unseen']= (epsilon_for_pt) / (total + epsilon_for_pt * (len(emit_prob_count[tag]) + 1))
            # print(emit_prob[tag])
    # for tag in emit_prob_count:
    #     total = sum(emit_prob_count[tag].values())
    #     for word in words:
    #         if word not in emit_prob_count[tag]:
    #             emit_prob[tag] = {word: (epsilon_for_pt) / (total + epsilon_for_pt * (len(emit_prob_count[tag]) + 1))}
    # print(emit_prob)
    # print(trans_prob)
    return init_prob, emit_prob, trans_prob 

def viterbi_stepforward(i, word, prev_prob, prev_predict_tag_seq, emit_prob, trans_prob):
    """
    Does one step of the viterbi function
    :param i: The i'th column of the lattice/MDP (0-indexing)
    :param word: The i'th observed word
    :param prev_prob: A dictionary of tags to probs representing the max probability of getting to each tag at in the
    previous column of the lattice
    :param prev_predict_tag_seq: A dictionary representing the predicted tag sequences leading up to the previous column
    of the lattice for each tag in the previous column
    :param emit_prob: Emis probabilities
    :param trans_prob: Trans probabilities
    :return: Current best log probs leading to the i'th column for each tag, and the respective predicted tag sequences
    """
    log_prob = {} # This should store the log_prob for all the tags at current column (i)
    predict_tag_seq = {} # This should store the tag sequence to reach each tag at column (i)
    if i==0:
        for tag in prev_prob:
            # print(word)
            # print(emit_prob)
            if word not in emit_prob[tag]:
                log_prob[tag]=log(epsilon_for_pt)+(prev_prob[tag])
            #     sum(emit_prob_count[tag].values())
            #     log_prob[tag]=log((epsilon_for_pt) / (total + epsilon_for_pt * (len(emit_prob_count[tag]) + 1)))+(prev_prob[tag])
            else:
                log_prob[tag]=log(emit_prob[tag][word])+(prev_prob[tag])
            predict_tag_seq[tag]=[tag]

    for tagb in prev_prob:
            max=-100000000000
            max_tag=''
            for taga in prev_prob:
                # print(prev_prob[taga])
                prob= (prev_prob[taga])
                if tagb not in trans_prob[taga]:
                    prob+=log(epsilon_for_pt)
                else:
                    prob+=log(trans_prob[taga][tagb])
                if word not in emit_prob[tagb]:
                    prob+=log(emit_prob[tagb]['unseen'])
                else:
                    prob+=log(emit_prob[tagb][word])
                # print(trans_prob[tagb][taga])
                # print(emit_prob[tagb][word])
                # else:
                #     prob= (prev_prob[taga])+log(trans_prob[tagb][taga])+log(emit_prob[tagb][word])
                if prob>max:
                    max=prob
                    max_tag=taga
            log_prob[tagb]=max
            
            # print('maxtag',max_tag)
            predict_tag_seq[tagb]=prev_predict_tag_seq[max_tag]+[tagb]
    
    # TODO: (II)
    # implement one step of trellis computation at column (i)
    # You should pay attention to the i=0 spec case.
    return log_prob, predict_tag_seq

def viterbi_1(train, test, get_probs=training):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    init_prob, emit_prob, trans_prob = get_probs(train)
    
    predicts = []
    
    for sen in range(len(test)):
        sentence=test[sen]
        length = len(sentence)
        log_prob = {}
        predict_tag_seq = {}
        # init log prob
        for t in emit_prob:
            if t in init_prob:
                log_prob[t] = log(init_prob[t])
            else:
                log_prob[t] = log(epsilon_for_pt)
            predict_tag_seq[t] = []

        # forward steps to calculate log probs for sentence
        for i in range(length):
            log_prob, predict_tag_seq = viterbi_stepforward(i, sentence[i], log_prob, predict_tag_seq, emit_prob,trans_prob)
            
        # TODO:(III) 
        # according to the storage of probabilities and sequences, get the final prediction.
        max=-10000000000000
        max_tag=''
        for tag in log_prob:
            if log_prob[tag]>max:
                max=log_prob[tag]
                max_tag=tag
        max_sequence=predict_tag_seq[max_tag]

        predicts.append([])
        for i in range(len(max_sequence)):
            predicts[sen].append((sentence[i], max_sequence[i]))

    return predicts




