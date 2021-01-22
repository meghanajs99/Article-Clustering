def insert(word, filename):
    import csv
    with open(filename, 'a',encoding='utf8') as csvfile:
        csvwriter = csv.writer(csvfile, lineterminator='\n')
        csvwriter.writerows(word)
