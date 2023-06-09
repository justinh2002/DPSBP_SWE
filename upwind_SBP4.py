'''author: Justin Kin Jun Hew'''
'''Upwind SBP operators of order 4'''


import numpy as np
np.set_printoptions(precision=13)

def upwind_SBP_order4():    

    m = 9  # problem size
    h = 1
    x = np.linspace(0, 1, m).reshape((-1, 1))

    x0 = x**0 / np.math.factorial(1)
    x1 = x**1 / np.math.factorial(1)
    x2 = x**2 / np.math.factorial(2)
    x3 = x**3 / np.math.factorial(3)
    x4 = x**4 / np.math.factorial(4)
    x5 = x**5 / np.math.factorial(5)
    x6 = x**6 / np.math.factorial(6)
    x7 = x**7 / np.math.factorial(7)
    x8 = x**8 / np.math.factorial(8)
    x9 = x**9 / np.math.factorial(9)
    x10 = x**10 / np.math.factorial(10)
    x11 = x**11 / np.math.factorial(11)
    x12 = x**12 / np.math.factorial(12)
    x13 = x**13 / np.math.factorial(13)
    x14 = x**14 / np.math.factorial(14)
    x15 = x**15 / np.math.factorial(15)
    c = np.ones((m, 1))

    # The coefficients in the norm are given by
    H_diag = np.ones(4)

    H_diag[0] = 49 / 144
    H_diag[1] = 61 / 48
    H_diag[2] = 41 / 48
    H_diag[3] = 149 / 144

    H_diag1 = np.diag(H_diag)
    H_diag2 = np.diag(np.flipud(H_diag))

    H = np.eye(m)
    H[:4, :4] = H_diag1
    H[-4:, -4:] = H_diag2
    H *= h
    HI = np.linalg.inv(H)

    Qp = np.zeros((m, m))
    Qp = -1 / 4 * np.diag(np.ones(m - 1), -1)- 5 / 6 * np.diag(np.ones(m), 0) + 3 / 2 * np.diag(np.ones(m - 1), 1) \
     - 1 / 2 * np.diag(np.ones(m - 2), 2)+ 1 / 12 * np.diag(np.ones(m - 3), 3)
    
     
    

    Qp_c_stencil = [-1 / 4, -5 / 6, +3 / 2, -1 / 2, +1 / 12]
    Qm_c_stencil = -np.flip(Qp_c_stencil)

    Q_U = np.zeros((4, 4))
    Q_U[0, 0] = -1 / 48
    Q_U[0, 1] = 205 / 288
    Q_U[0, 2] = -29 / 144
    Q_U[0, 3] = 1 / 96
    Q_U[1, 0] = -169 / 288
    Q_U[1, 1] = -11 / 48
    Q_U[1, 2] = 33 / 32
    Q_U[1, 3] = -43 / 144
    Q_U[2, 0] = 11 / 144
    Q_U[2, 1] = -13 / 32
    Q_U[2, 2] = -29 / 48
    Q_U[2, 3] = 389 / 288
    Q_U[3, 0] = 1 / 32
    Q_U[3, 1] = -11 / 144
    Q_U[3, 2] = -65 / 288
    Q_U[3, 3] = -13 / 16

    Qp[:4, :4] = Q_U
    Qp[-4:, -4:] = np.flipud(np.fliplr(Q_U)).T

    Qm = -Qp.T

    e_1 = np.zeros(m)
    e_1[0] = 1
    e_m = np.zeros(m)
    e_m[-1] = 1

    Dp = HI @ (Qp - 1 / 2 * np.outer(e_1, e_1) + 1 / 2 * np.outer(e_m, e_m))
    Dm = HI @ (Qm - 1 / 2 * np.outer(e_1, e_1) + 1 / 2 * np.outer(e_m, e_m))
     
    print("Dp is: ", Dp)
   # print("Dm is: ", Dm)
    
    

    
    
    
    
    

if __name__ == "__main__":
    
    upwind_SBP_order4()
    
    #main()

    # now parse arguments
