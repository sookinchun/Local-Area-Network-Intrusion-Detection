import sys
import csv

label = sys.argv[1]
file_name = sys.argv[2]

file = open(file_name)
content = csv.reader(file)
#row0 = content.next() for python2 and next(content) for python 3
row0 = next(content)
row0.append('label')
all = []
all.append(row0)
for item in content:
    item.append(label)
    all.append(item)

new_file = open("Labeled_" + file_name, 'w')
writer = csv.writer(new_file, lineterminator='\n')
writer.writerows(all)