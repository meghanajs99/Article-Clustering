import preprocess
import glob
import errno
path = r"C:\Users\RajaniKumari\Documents\projects\ml\DATASET\news\entertainment\*.*"
files = glob.glob(path)
all_data = []
docLabels = []
list_data=[]
for name in files:
    try:
        with open(name) as f:
            data = f.read()
            docLabels.append(name)
            word = preprocess.process(data, name)
            list_data.append(word)
            for x in word:
                all_data.append(x)
    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise
print(all_data,len(all_data))
unique=set(all_data)
print(len(unique))
