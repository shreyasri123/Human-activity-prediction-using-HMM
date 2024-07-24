from preprocessing import *
from hmm import *
import pytz # allows accurate and cross platform timezone calculations


def probability_distribution(seq1, seq2):
    n = 1 + max(seq1);
    m = 1 + max(seq2)
    M = np.zeros((n, m))

    # note the occurrences
    for s1, s2 in zip(seq1, seq2):
        M[s1][s2] += 1

    # It sets the probabilities to 'epsilon' which are zero
    M[M == 0] = 10e-2

    # Calculation of probability distributions
    row_sums = M.sum(axis=1)
    M = M / row_sums[:, np.newaxis]

    return M


# Calculate the probability distribution of the states
def prior(transitions):
    g = transitions.groupby(transitions)
    result = g.count() / g.count().sum()
    return result.values


# Compute the transition matrix given the sequence of states at each time t
def transition_matrix(sequence):
    return probability_distribution(sequence, sequence[1:])


# Calculate probabilty distribution of observations for each state
def obs_matrix(seq, obs):
    return probability_distribution(seq, obs)


def date_to_timestamp(m):
    return int(datetime.strptime(m.strip(), "%Y-%m-%d %H:%M:%S").timestamp())


def print_numpy_matrix(m):
    import sys
    np.savetxt(sys.stdout, m, '%6.4f')


# calculate probability matrixes for timeslice
def slice_prob(dt, days):

    # check whether DATASET A or B
    if dt == 1:
        try:
            # takes merged datset from csv
            df = pd.read_csv('dataset_csv/OrdonezA.csv')
        except:
            # if it doesnt find csv it performs preprocessing
            df = generate_dataset()[0]
    else:
        try:
            # takes the merged datset from csv
            df = pd.read_csv('dataset_csv/OrdonezB.csv')
        except:
            # if it doesnt find csv it performs preprocessing
            df = generate_dataset()[1]


    # discritize sensor observations
    df[['sensors']] = df[['sensors']].apply(lambda x: x.astype('category'))
    mapping = dict(enumerate(df['sensors'].cat.categories))
    df[['sensors']] = df[['sensors']].apply(lambda x: x.cat.codes)

    df['date'] = df['timestamp'].apply(lambda x: datetime.fromtimestamp(x, tz=pytz.UTC))

    if dt == 1:
        # DATASET A: TRAIN and TEST SET
        if days == 1:
            trainIndex = range(0, 16401)
            testIndex = range(16402, len(df.index))
            train = df.loc[trainIndex, :]
            test = df.loc[testIndex, :]

        elif days == 2:
            trainIndex = range(2371, len(df.index))
            testIndex = range(0, 2370)
            train = df.loc[trainIndex, :]
            test = df.loc[testIndex, :]

        elif days == 3:
            trainIndex = range(0, 10930)
            train = df.loc[trainIndex, :]
            train = train.append(df.loc[range(13544, len(df.index)), :])
            testIndex = range(10931, 13543)
            test = df.loc[testIndex, :]

        elif days == 4:
            trainIndex = range(0, 13676)
            train = df.loc[trainIndex, :]
            train = train.append(df.loc[range(16402, len(df.index)), :])
            testIndex = range(13677, 16401)
            test = df.loc[testIndex, :]

        elif days == 5:
            trainIndex = range(0, 4063)
            train = df.loc[trainIndex, :]
            train = train.append(df.loc[range(6619, len(df.index)), :])
            testIndex = range(4064, 6618)
            test = df.loc[testIndex, :]

        # test day3
        if days == 6:
            trainIndex = range(0, 15051)
            train = df.loc[trainIndex, :]
            train = train.append(df.loc[range(17899, len(df.index)), :])
            testIndex = range(15052, 17898)
            test = df.loc[testIndex, :]

        elif days == 7:
            trainIndex = range(4065, len(df.index))
            testIndex = range(0, 4064)
            train = df.loc[trainIndex, :]
            test = df.loc[testIndex, :]

        elif days == 8:
            trainIndex = range(0, 9660)
            train = df.loc[trainIndex, :]
            train = train.append(df.loc[range(13678, len(df.index)), :])
            testIndex = range(9661, 13677)
            test = df.loc[testIndex, :]

        elif days == 9:
            trainIndex = range(0, 12297)
            train = df.loc[trainIndex, :]
            train = train.append(df.loc[range(16403, len(df.index)), :])
            testIndex = range(12298, 16402)
            test = df.loc[testIndex, :]

        elif days == 10:
            trainIndex = range(0, 2662)
            train = df.loc[trainIndex, :]
            train = train.append(df.loc[range(6856, len(df.index)), :])
            testIndex = range(2663, 6855)
            test = df.loc[testIndex, :]










    else :
        # DATASET B: TRAIN and TEST SET
        # test day 3
        if days == 1:
            trainIndex = range(0, 6673)
            train = df.loc[trainIndex, :]
            train = train.append(df.loc[range(9036, len(df.index)), :])
            testIndex = range(6674, 9035)
            test = df.loc[testIndex, :]

        elif days == 2:
            trainIndex = range(0, 22329)
            train = df.loc[trainIndex, :]
            train = train.append(df.loc[range(24809, len(df.index)), :])
            testIndex = range(22330, 24808)
            test = df.loc[testIndex, :]

        elif days == 3:
            trainIndex = range(0, 275)
            train = df.loc[trainIndex, :]
            train = train.append(df.loc[range(2672, len(df.index)), :])
            testIndex = range(276, 2671)
            test = df.loc[testIndex, :]

        elif days == 4:
            trainIndex = range(0, 21092)
            train = df.loc[trainIndex, :]
            train = train.append(df.loc[range(23564, len(df.index)), :])
            testIndex = range(21091, 23563)
            test = df.loc[testIndex, :]

        elif days == 5:
            trainIndex = range(0, 13164)
            train = df.loc[trainIndex, :]
            train = train.append(df.loc[range(15765, len(df.index)), :])
            testIndex = range(13165, 15764)
            test = df.loc[testIndex, :]

        # test day7
        if days == 6:
            trainIndex = range(0, 6671)
            train = df.loc[trainIndex, :]
            train = train.append(df.loc[range(15966, len(df.index)), :])
            testIndex = range(6672, 15965)
            test = df.loc[testIndex, :]

        elif days == 7:
            trainIndex = range(0, 15764)
            train = df.loc[trainIndex, :]
            train = train.append(df.loc[range(23565, len(df.index)), :])
            testIndex = range(15765, 23564)
            test = df.loc[testIndex, :]

        elif days == 8:
            # trainIndex = range(0, 274)
            # train = df.loc[trainIndex, :]
            # train = train.append(df.loc[range(9037, len(df.index)), :])
            # testIndex = range(275, 9036)
            # test = df.loc[testIndex, :]

            trainIndex = range(0, 4108)
            train = df.loc[trainIndex, :]
            train = train.append(df.loc[range(16165, len(df.index)), :])
            testIndex = range(4109, 13164)
            test = df.loc[testIndex, :]


        elif days == 9:
            trainIndex = range(0, 11728)
            train = df.loc[trainIndex, :]
            train = train.append(df.loc[range(21092, len(df.index)), :])
            testIndex = range(11729, 21091)
            test = df.loc[testIndex, :]

        elif days == 10:
            trainIndex = range(0, 13164)
            train = df.loc[trainIndex, :]
            train = train.append(df.loc[range(22273, len(df.index)), :])
            testIndex = range(13165, 22272)
            test = df.loc[testIndex, :]



    trainset_s = train['activity']
    trainset_o = train['sensors']
    testset_s = test['activity'].tolist()
    testset_o = test['sensors'].tolist()

    # CalCULATION OF HMM DISTRIBUTIONS
    P = prior(trainset_s)
    T = transition_matrix(trainset_s)
    O = obs_matrix(trainset_s, trainset_o)

    # VITERBI
    viterbi_result, p, x = hmm.viterbi(testset_o, T, O, P)

    # COUNT HOW MANY STATES GUESS
    c = 0
    for i, j in zip(viterbi_result, testset_s):
        if i == j:
            c += 1

    accuracy = c/len(viterbi_result) * 100

    # converts list to array
    testset_s = np.asarray(testset_s)

    return testset_s, viterbi_result, accuracy




# if __name__ == '__main__':
#     slice_prob(1,2)
