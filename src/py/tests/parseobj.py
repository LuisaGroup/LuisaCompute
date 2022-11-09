
def parseobj(file):
	v = []
	vt = []
	vn = []
	f = []
	for line in file:
		split = line.split()
		if len(split) == 0:
			continue
		if split[0] == 'v':
			assert len(split[1:]) == 3
			v.append(list(map(float, split[1:])))
		elif split[0] == 'vt':
			assert len(split[1:]) == 2
			vt.append(list(map(float, split[1:])))
		elif split[0] == 'vn':
			assert len(split[1:]) == 3
			vn.append(list(map(float, split[1:])))
		elif split[0] == 'f':
			assert len(split[1:]) == 3
			tri = []
			for x in split[1:]:
				if '/' not in x:
					tri.append(int(x)-1)
				else:
					xs = list(map(int, x.split('/')))
					assert len(xs) == 3
					assert xs[0] == xs[1] == xs[2]
					tri.append(xs[0]-1)
			f.append(tri)
	return v,vt,vn,f
