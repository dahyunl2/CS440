# viterbi.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Renxuan Wang (renxuan2@illinois.edu) on 10/18/2018
import math
"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

'''
TODO: implement the baseline algorithm.
input:  training data (list of sentences, with tags on the words)
        test data (list of sentences, no tags on the words)
output: list of sentences, each sentence is a list of (word,tag) pairs.
        E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
'''
def baseline(train, test):
    predicts = []
    return predicts

'''
TODO: implement the Viterbi algorithm.
input:  training data (list of sentences, with tags on the words)
        test data (list of sentences, no tags on the words)
output: list of sentences with tags on the words
        E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
'''
def viterbi(train, test):
    smoothing_parameter = 0.000001
    # Calculating how often does each tag occur at the start of a sentence
    initial_prob = {}
    for i in range(len(train)):
        if(train[i][0][1] not in initial_prob):
            initial_prob[train[i][0][1]] = 0
        else:
            initial_prob[train[i][0][1]] += 1
    initial_total = sum(initial_prob.values())
    for i in initial_prob:
        initial_prob[i] = initial_prob[i] / initial_total

    # Calculating how often does tag tb follow tag ta
    transition_prob = {}
    for sentence in train:
        for word in range(len(sentence) - 1):
            item = sentence[word][1] + sentence[word + 1][1]
            if item not in transition_prob:
                transition_prob[item] = 0
            else:
                transition_prob[item] += 1
    transition_total = sum(transition_prob.values())
    for i in transition_prob:
        transition_prob[i] = (transition_prob[i] + smoothing_parameter) / (transition_total + smoothing_parameter * (len(transition_prob) + 1))

    # Calculating how often does tag t yield word w
    emission_prob = {}
    for sentence in train:
        for word in sentence:
            if word[1] not in emission_prob:
                emission_prob[word[1]] = {}
            if word[0] not in emission_prob[word[1]]:
                emission_prob[word[1]][word[0]] = 1
            else:
                emission_prob[word[1]][word[0]] += 1

    for tag in emission_prob:
        tag_total = sum(emission_prob[tag].values())
        for word in emission_prob[tag]:
            emission_prob[tag][word] = (emission_prob[tag][word] + smoothing_parameter) / (tag_total + smoothing_parameter * (len(emission_prob[tag]) + 1))

    # Start the labelling the test set
    predicts = []
    for i in range(len(test)):
        if(len(test[i]) == 0):
            predicts.append([])
        else:
            sentence = test[i]
            predicts.append(viterbi_helper(sentence, initial_prob, transition_prob, emission_prob, smoothing_parameter))

    return predicts

def viterbi_helper(sentence, initial_prob, transition_prob, emission_prob, smoothing_parameter):
    prob_matrix = {}
    back_pointer = {}
    for state in initial_prob:
        prob_matrix[state] = []
        back_pointer[state] = []
        for i in range(len(sentence)):
            prob_matrix[state].append(0)
            back_pointer[state].append(0)

    for state in prob_matrix:
        back_pointer[state][0] = 0
        if sentence[0] not in emission_prob[state]:
            state_total = sum(emission_prob[state].values())
            prob_matrix[state][0] = math.log(initial_prob[state]) + math.log(smoothing_parameter / (state_total + smoothing_parameter * (len(emission_prob[state]) + 1)))
        else:
            prob_matrix[state][0] = math.log(initial_prob[state]) + math.log(emission_prob[state][sentence[0]])

    for t in range(1, len(sentence)):
        for state in prob_matrix:
            emission_state_total = sum(emission_prob[state].values())
            transition_state_total = sum(transition_prob.values())
            prob_matrix[state][t] = math.inf * -1
            for state_prime in prob_matrix:
                trans_item = state_prime + state
                if trans_item not in transition_prob:
                    curr_transition_prob = math.log(smoothing_parameter / (transition_state_total + smoothing_parameter * (len(transition_prob) + 1)))
                else:
                    curr_transition_prob = math.log(transition_prob[trans_item])
                if sentence[t] not in emission_prob[state]:
                    curr_emission_prob = math.log(smoothing_parameter / (emission_state_total + smoothing_parameter * (len(emission_prob[state]) + 1)))
                else:
                    curr_emission_prob = math.log(emission_prob[state][sentence[t]])
                curr_prob = prob_matrix[state_prime][t - 1] + curr_transition_prob + curr_emission_prob
                if(curr_prob > prob_matrix[state][t]):
                    prob_matrix[state][t] = curr_prob
                    back_pointer[state][t] = state_prime

    bestpathprob = math.inf * -1
    for state in prob_matrix:
        if(prob_matrix[state][len(sentence) - 1] > bestpathprob):
            bestpathprob = prob_matrix[state][len(sentence) - 1]
            bestpathpointer = state

    bestpath = [bestpathpointer]
    for t in range(0, len(sentence) - 1):
        column = len(sentence) - 1 - t
        bestpath.append(back_pointer[bestpathpointer][column])
        bestpathpointer = back_pointer[bestpathpointer][column]
    bestpath = bestpath[::-1]

    for word in range(len(sentence)):
        bestpath[word] = (sentence[word], bestpath[word])

    return bestpath