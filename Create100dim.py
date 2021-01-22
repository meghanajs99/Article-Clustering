import csv
import pandas as pd

words = []
count = 0
vecs = []
x = []
with open('../input/vectors/vectors.csv') as f_obj:
    from itertools import islice

    for line in islice(csv.reader(f_obj), 1, None):
        vec = []
        vecs = []
        words.append(line[0])
        for i in range(1, 101):
            vec.append(float(line[i]))
        vecs.append(vec)
        x.append(vecs)

word_vec_dict = {words[i]: x[i] for i in range(len(words))}
# df = pd.DataFrame.from_dict(word_vec_dict, orient='index')
# df.to_csv('W.csv',index=True)
# print(str(word_vec_dict['leo']))
# print(str(word_vec_dict))
na = []
dict_values = []
count = 0
rows = []
with open('../input/words-csv/words.csv') as f_obj:
    from itertools import islice

    for line in islice(csv.reader(f_obj), 1, None):
        rows.append(line)
    for line in rows:
        word_vec = []
        na.append(line[0])
        row = line[1].replace('[', '')
        row = row.replace(']', '')
        import ast

        row = ast.literal_eval(row)

        for x in row:
            # print(x)
            word_vec.append(x)
            word_vec.append(word_vec_dict[x])

        dict_values.append(word_vec)
print(dict_values)
doc_dict = {na[i]: dict_values[i] for i in range(len(na))}
df = pd.DataFrame.from_dict(doc_dict, orient='index')
df.to_csv('100dimwordvectors.csv', index=True)