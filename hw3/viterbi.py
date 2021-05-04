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
    # - R - dp sequence codes
    # - bp - dp scores backward pointers
    bp = np.zeros((N, L))
    R = -np.inf*np.ones((N, L))

    # Add beginning scores to R
    # Seed dynamic program
    # (start_score + emission)
    R[0,:] = start_scores + emission_scores[0,:]

    # Iterate through tokens
    for i in range(1, N):
        #Iterature through labels
        for j in range(L):
            #Iterate again through labels for preceding one
            for l in range(L):

                # Sum of previous R, at a label, emissions scores, transition scores
                total_score = R[i-1][l] + emission_scores[i,j] + trans_scores[l][j]

                #Update if you have found a more probably subseq, update backpointers too
                if (total_score > R[i,j]):
                    R[i,j] = total_score
                    bp[i,j] = l

    # Add the ending scores to R
    # After the build-up is done
    R[N-1, :] += end_scores

    # Get sequence index with highest score from R
    # to plug into bp
    y = [np.argmax(R[-1])]


    for i in range(1, N):
        # Dynamically add from backpointer using
        # last added element
        y.insert(0,bp[N - i, int(y[0])])

    # Return largest score from R
    # (last row, at last index of y)
    # As well as integers representing best seq
    return (R[-1,y[-1]], y)
