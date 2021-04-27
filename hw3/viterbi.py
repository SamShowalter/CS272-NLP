import numpy as np


def run_viterbi(emission_scores, trans_scores, start_scores, end_scores):
    """Run the Viterbi algorithm.

    N - number of tokens (length of sentence)
    L - number of labels

    As an input, you are given:
    - Emission scores, as an NxL array
    - Transition scores (Yp -> Yc), as an LxL array
    - Start transition scores (S -> Y), as an Lx1 array
    - End transition scores (Y -> E), as an Lx1 array

    You have to return a tuple (s,y), where:
    - s is the score of the best sequence
    - y is the size N array/seq of integers representing the best sequence.
    """
    L = start_scores.shape[0]
    assert end_scores.shape[0] == L
    assert trans_scores.shape[0] == L
    assert trans_scores.shape[1] == L
    assert emission_scores.shape[1] == L
    N = emission_scores.shape[0]

    # Init the dp matrices
    # - dp sequence codes
    # - dp scores
    dp_seqs = np.zeros((N, L))
    dp_scores = -np.inf*np.ones((N, L))

    # Add beginning scores to dp_scores
    # Seed dynamic program
    # (start_score + emission)
    dp_scores[0,:] = start_scores + emission_scores[0,:]

    # Iterate through tokens
    for i in range(1, N):
        #Iterature through labels
        for j in range(L):
            #Iterate again through labels for preceding one
            for l in range(L):

                # Sum of previous dp_scores, at a label, emissions scores, trans
                total_score = dp_scores[i-1][l] + emission_scores[i,j] + trans_scores[l][j]

                #Update if you have found a more probably subseq
                if (total_score > dp_scores[i,j]):
                    dp_scores[i,j] = total_score
                    dp_seqs[i,j] = l

    # Add the ending scores to dp_scores
    # After the build-up is done
    dp_scores[N-1, :] += end_scores

    # Get sequence index with highest score from dp_scores
    # to plug into dp_seqs
    y = [np.argmax(dp_scores[-1])]


    for i in range(1, N):
        # Dynamically add from dp_seqs using
        # last added element
        y.insert(0,dp_seqs[N - i, int(y[0])])

    # Return largest score from dp_scores
    # (last row, at last index of y)
    # As well as integers representing best seq
    return (dp_scores[-1,y[-1]], y)
