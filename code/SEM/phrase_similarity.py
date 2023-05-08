
from nltk.corpus import wordnet as wn

def wordnetSim3(word1, word2):
    totalPoint = 0
    simList = []
    phrase1 = word1
    phrase2 = word2
    word1 = phrase1.split(' ')
    word2 = phrase2.split(' ')
    for w1 in word1:
        for w2 in word2:
            synsets1 = wn.synsets(w1)
            synsets2 = wn.synsets(w2)
            path_sim = 0
            for tmpword1 in synsets1:
                for tmpword2 in synsets2:
                        try:
                            sim = tmpword1.path_similarity(tmpword2)
                            path_sim = max(path_sim, sim)
                        except Exception as e:
                            continue
            simList.append(path_sim)
    for sim in simList:
        totalPoint += sim
    min_len = min(len(word1),len(word2))
    result = totalPoint/min_len
    return result
# print(wordnetSim3("location data", "geographical position information"))

def wordnetSim_modified(word1, word2):
    totalPoint = 0
    simList = []
    phrase1 = word1
    phrase2 = word2
    word1 = phrase1.split(' ')
    word2 = phrase2.split(' ')
    for w1 in word1:
        for w2 in word2:
            synsets1 = wn.synsets(w1)
            synsets2 = wn.synsets(w2)
            path_sim = 0
            for tmpword1 in synsets1:
                for tmpword2 in synsets2:
                        try:
                            sim = tmpword1.path_similarity(tmpword2)
                            path_sim = max(path_sim, sim)
                        except Exception as e:
                            continue
            simList.append(path_sim)
    for sim in simList:
        totalPoint += sim

    result = totalPoint/len(simList)

    # result = max(simList)

    return result
