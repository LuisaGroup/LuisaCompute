#pragma once

static const char *obj_string = R"(
# The original Cornell Box in OBJ format.
# Note that the real box is not a perfect cube, so
# the faces are imperfect in this data set.
#
# Created by Guedis Cardenas and Morgan McGuire at Williams College, 2011
# Released into the Public Domain.
#
# http://graphics.cs.williams.edu/data
# http://www.graphics.cornell.edu/online/box/data.html
#

mtllib CornellBox-Original.mtl

g floor
v  -1.01  0.00   0.99
v   1.00  0.00   0.99
v   1.00  0.00  -1.04
v  -0.99  0.00  -1.04
f -4 -3 -2 -1

g ceiling
v  -1.02  1.99   0.99
v  -1.02  1.99  -1.04
v   1.00  1.99  -1.04
v   1.00  1.99   0.99
f -4 -3 -2 -1

g backWall
v  -0.99  0.00  -1.04
v   1.00  0.00  -1.04
v   1.00  1.99  -1.04
v  -1.02  1.99  -1.04
f -4 -3 -2 -1

g rightWall
v	1.00  0.00  -1.04
v	1.00  0.00   0.99
v	1.00  1.99   0.99
v	1.00  1.99  -1.04
f -4 -3 -2 -1

g leftWall
v  -1.01  0.00   0.99
v  -0.99  0.00  -1.04
v  -1.02  1.99  -1.04
v  -1.02  1.99   0.99
f -4 -3 -2 -1

g shortBox

# Top Face
v	0.53  0.60   0.75
v	0.70  0.60   0.17
v	0.13  0.60   0.00
v  -0.05  0.60   0.57
f -4 -3 -2 -1

# Left Face
v  -0.05  0.00   0.57
v  -0.05  0.60   0.57
v   0.13  0.60   0.00
v   0.13  0.00   0.00
f -4 -3 -2 -1

# Front Face
v	0.53  0.00   0.75
v	0.53  0.60   0.75
v  -0.05  0.60   0.57
v  -0.05  0.00   0.57
f -4 -3 -2 -1

# Right Face
v	0.70  0.00   0.17
v	0.70  0.60   0.17
v	0.53  0.60   0.75
v	0.53  0.00   0.75
f -4 -3 -2 -1

# Back Face
v	0.13  0.00   0.00
v	0.13  0.60   0.00
v	0.70  0.60   0.17
v	0.70  0.00   0.17
f -4 -3 -2 -1

# Bottom Face
v	0.53  0.00   0.75
v	0.70  0.00   0.17
v	0.13  0.00   0.00
v  -0.05  0.00   0.57
f -4 -3 -2 -1

g tallBox

# Top Face
v	-0.53  1.20   0.09
v	 0.04  1.20  -0.09
v	-0.14  1.20  -0.67
v	-0.71  1.20  -0.49
f -4 -3 -2 -1

# Left Face
v	-0.53  0.00   0.09
v	-0.53  1.20   0.09
v	-0.71  1.20  -0.49
v	-0.71  0.00  -0.49
f -4 -3 -2 -1

# Back Face
v	-0.71  0.00  -0.49
v	-0.71  1.20  -0.49
v	-0.14  1.20  -0.67
v	-0.14  0.00  -0.67
f -4 -3 -2 -1

# Right Face
v	-0.14  0.00  -0.67
v	-0.14  1.20  -0.67
v	 0.04  1.20  -0.09
v	 0.04  0.00  -0.09
f -4 -3 -2 -1

# Front Face
v	 0.04  0.00  -0.09
v	 0.04  1.20  -0.09
v	-0.53  1.20   0.09
v	-0.53  0.00   0.09
f -4 -3 -2 -1

# Bottom Face
v	-0.53  0.00   0.09
v	 0.04  0.00  -0.09
v	-0.14  0.00  -0.67
v	-0.71  0.00  -0.49
f -4 -3 -2 -1

g light
v	-0.24  1.98   0.16
v	-0.24  1.98  -0.22
v	 0.23  1.98  -0.22
v	 0.23  1.98   0.16
f -4 -3 -2 -1)";

