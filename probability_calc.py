import numpy
import pandas as pd
import numpy as np


# calculate intial probabilities
def get_start_prob(dataset):

    s = sum(dataset['Activity'].value_counts())
    # print(s)
    norm = [float(i) / s for i in dataset['Activity'].value_counts()]

    # print("calculated intial probabilities")
    ret = pd.DataFrame([norm],columns=dataset['Activity'].unique().tolist())
    return ret


# calculate transition probabilities
def get_trans_prob(dataset, dt):
    nextState = []


    qwer = len(dataset.index)
    activities = dataset['Activity'].unique().tolist()
    matrix = np.zeros(len(activities))
    for state in activities:
        #INDEXES IN WHICH STATE IS PRESENT
        wooo = [i for i, j in enumerate(dataset['Activity'].tolist()) if j == state]
        #STATE INDICES AFTER STATE
        nextIndex = [x + 1 for x in wooo]
        #QWER CONTAINS FIRST INDEXOUTOFBOUND
        if qwer in nextIndex:
            nextIndex.remove(qwer)
        #T CONTAINS THE STATES SUBSEQUENT TO THE STATE I AM ANALYZING (E.G. ALL POST BREAKFAST STATES)
        T = [dataset['Activity'].tolist()[i] for i in nextIndex]
        nextState.append(T)
        counter = []
        for secState in activities:
            num = 0
            num = T.count(secState)
            counter.append(num)
            app = np.array(counter)



        norm = [float(i) / sum(app) for i in app]
        matrix = numpy.vstack([matrix, norm])

    matrix = numpy.delete(matrix, (0), axis=0)
    # dt=1 Dataset A
    if dt==1:
        matrix[matrix == 0] = 10e-3
    else:
        matrix[matrix==0] = 10e-5

    # calculation of probability distribution 
    row_sums = matrix.sum(axis=1)
    matrix = matrix / row_sums[:, np.newaxis]
    ret = pd.DataFrame(matrix, index=activities, columns=activities)
    # print("calculated transition probabilities")
    return ret


# calculate probabilities of observations
def get_obs_prob(dataset, dt):
    evidence = []


    activities = dataset['Activity'].unique().tolist()
    evidenceList = dataset['Evidence'].unique().tolist()
    matrix = np.zeros(len(evidenceList))

    for state in activities:
        # INDEXES IN WHICH STATE IS PRESENT
        wooo = [i for i, j in enumerate(dataset['Activity'].tolist()) if j == state]
        # STATE INDICES AFTER STATE

        # QWER CONTAINS FIRST INDEXOUTOFBOUND

        # T CONTAINS THE STATES SUBSEQUENT TO THE STATE I AM ANALYZING (E.G. ALL POST BREAKFAST STATES)
        T = [dataset['Evidence'].tolist()[i] for i in wooo]
        evidence.append(T)
        counter = []
        for currentEvidence in evidenceList:
            num = T.count(currentEvidence)
            counter.append(num)
            app = np.array(counter)

        norm = [float(i) / sum(app) for i in app]
        matrix = numpy.vstack([matrix, norm])

    matrix = numpy.delete(matrix, (0), axis=0)

    # dt=1 Dataset A
    if dt==1:
        matrix[matrix == 0] = 10e-3
    else:
        matrix[matrix == 0] = 10e-5

    # calculations of probablity distributions
    row_sums = matrix.sum(axis=1)
    matrix = matrix / row_sums[:, np.newaxis]
    # print("calculated transition probablities")
    ret = pd.DataFrame(matrix, index=activities, columns=evidenceList)
    return ret
