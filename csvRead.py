import ast
import numpy as np
import csvW


def look():
    import csv
    filename = "100dimBookvectors.csv"
    rows = []
    from itertools import islice
    with open(filename) as fd:
        for row in islice(csv.reader(fd), 1, None):
            rows.append(row)

    wodict = {}
    for row in rows:
        vecs = []
        key1 = []
        for i in range(1, len(row)):
            if i % 2 != 0 and row[i] != '':
                key1.append(row[i])
            elif i % 2 == 0 and row[i] != '':
                vecs.append(row[i])
        for i in range(len(key1)):
            wodict[key1[i]] = vecs[i]

        min_value = []
        Y = []
        X = []
        Y.append(ast.literal_eval(vecs[0]))
        X.append(ast.literal_eval(vecs[0]))
        for i in range(1, len(vecs)):
            Y.append(ast.literal_eval(vecs[i]))
            X.append(ast.literal_eval(vecs[i]))
            min_value = np.min(Y, axis=0)
            max_value = np.max(X, axis=0)
            Y = []
            X = []
            Y.append((min_value).tolist())
            X.append((max_value.tolist()))

        FeaVec = []
        minV = [j for sub in min_value for j in sub]
        maxV=[j for sub in max_value for j in sub]
        FeaVec.append(minV)
        FeaVec.append(maxV)
        MeanVec = np.mean(FeaVec, axis=0)
        FeaVec.append(MeanVec.tolist())

        csvword = []
        FeaVec.insert(0, row[0])
        csvword.append(FeaVec)
        csvW.insert(csvword, "Book Feature Vectors.csv")


def look2():
    import csv
    filename = "WordVectors.csv"
    rows = []
    from itertools import islice
    with open(filename) as fd:
        for row in islice(csv.reader(fd), 1, None):
            rows.append(row)

    wodict = {}
    for row in rows:
        vecs = []
        key1 = []
        for i in range(1, len(row)):
            if i % 2 != 0 and row[i] != '':
                key1.append(row[i])
            elif i % 2 == 0 and row[i] != '':
                vecs.append(row[i])
        for i in range(len(key1)):
            wodict[key1[i]] = vecs[i]

        Y = []
        for i in range(0, len(vecs)):
            Y.append(ast.literal_eval(vecs[i]))

        meanvec = np.mean(Y, axis=0)
        feavec = []
        feavec.append(meanvec.tolist())

        csvword = []
        feavec.insert(0, row[0])
        csvword.append(feavec)
        csvW.insert(csvword, "AllFeature.csv")


def fealook():
    import csv
    vecs = []
    filename = "AllFeature.csv"
    rows = []
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            rows.append(row)
    print(len(rows))
    for row in rows:
        for i in range(1, len(row)):
            if i == 3:
                vecs.append(row[i])
    return vecs


def fealook2():
    import csv
    vecs = []
    filename = "Book Feature Vectors.csv"
    rows = []
    with open(filename, 'r',encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            rows.append(row)
    print(len(rows))
    for row in rows:
        for i in range(1, len(row)):
            if i == 3:
                vecs.append(row[i])
    return vecs
