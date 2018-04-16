import numpy as np

all_filenames=np.array([])
for i in range(1,28):
    filename = ['res/video%s_final.txt'%i]
    all_filenames = np.append(all_filenames,filename)

with open('track1.txt', 'w') as outfile:
    for fname in all_filenames:
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)