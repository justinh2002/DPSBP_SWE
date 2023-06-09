'''upwind SBP order 5'''
'''Author: Justin Kin Jun Hew'''


import numpy as np
m = 9
h=1
np.set_printoptions(precision=16)

H = np.diag(np.ones(m))
H[0:4, 0:4] = np.array([[251./720, 0, 0, 0],
                        [0, 299./240, 0, 0],
                        [0, 0, 41./48, 0],
                        [0, 0, 0, 149./144]])
H[m-4:m, m-4:m] = np.fliplr(np.flipud(H[0:4, 0:4]))
H = H * h
HI = np.linalg.inv(H)

Qp = (1/20 * np.diag(np.ones(m-2), -2) - 1/2 * np.diag(np.ones(m-1), -1) -
      1/3 * np.diag(np.ones(m), 0) + 1 * np.diag(np.ones(m-1), 1) -
      1./4 * np.diag(np.ones(m-2), 2) + 1./30* np.diag(np.ones(m-3), 3))

Q_U = np.array([[-1./120, 941./1440, -47./360, -7./480],
                [-869./1440, -11./120, 25./32, -43./360],
                [29./360, -17./32, -29./120, 1309./1440],
                [1./32, -11./360, -661./1440, -13./40]])

Qp[0:4, 0:4] = Q_U
Qp[m-4:m, m-4:m] = np.flipud(np.fliplr(Q_U[0:4, 0:4])).T

Qm = -Qp.T

e_1 = np.zeros(m)
e_1[0] = 1
e_m = np.zeros(m)
e_m[m-1] = 1

Dp = HI.dot(Qp - 1/2 * np.outer(e_1, e_1) + 1/2 * np.outer(e_m, e_m))
Dm = HI.dot(Qm - 1/2 * np.outer(e_1, e_1) + 1/2 * np.outer(e_m, e_m))
#print(Dm)
print(Dp)
