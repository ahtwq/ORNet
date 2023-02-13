import os
import glob
import re

paths = glob.glob('cv*.txt')
for path in paths:
	head, tail, _ = re.split(r'_|\.', path)
	newname = tail + str(head[-1]) + '.txt'
	os.rename(path, newname)
#	print(path, newname)


