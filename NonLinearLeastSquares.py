import pandas as pd
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 25})


def fun_f(x):
    df = pd.read_csv('Dados90.csv', names=['Dados'])

    data = df['Dados']

    VEmod=data[0]
    VEfase=data[1]
    IEmod=data[2]
    IEfase=data[3]
    VSmod=data[4]
    VSfase=data[5]
    ISmod=data[6]
    ISfase=data[7]

    n=1000

    sigma = 0.01

    thetamax =30* (10 ** (-3))

    np.random.seed(42)
    w1 = VEmod * np.ones(n) + VEmod * (sigma / 3) * np.random.normal(0, 1, n)
    np.random.seed(41)
    w2 = VEfase + (thetamax / 3) * np.random.normal(0, 1, n)
    np.random.seed(40)
    w3 = IEmod * np.ones(n) + IEmod * (sigma / 3) * np.random.normal(0, 1, n)
    np.random.seed(39)
    w4 = IEfase + (thetamax / 3) * np.random.normal(0, 1, n)
    np.random.seed(38)
    w5 = VSmod * np.ones(n) + VSmod * (sigma / 3) * np.random.normal(0, 1, n)
    np.random.seed(37)
    w6 = VSfase + (thetamax / 3) * np.random.normal(0, 1, n)
    np.random.seed(36)
    w7 = ISmod * np.ones(n) + ISmod * (sigma / 3) * np.random.normal(0, 1, n)
    np.random.seed(35)
    w8 = ISfase + (thetamax / 3) * np.random.normal(0, 1, n)

    realIe = np.ones(n)
    imagVe = np.ones(n)
    imagIe = np.ones(n)
    realVe = np.ones(n)
    realIs = np.ones(n)
    imagVs = np.ones(n)
    imagIs = np.ones(n)
    realVs = np.ones(n)

    for j in range(0, n):
        realIe[j] = w3[j] * np.cos(w4[j])
        imagVe[j] = w1[j] * np.sin(w2[j])
        imagIe[j] = w3[j] * np.sin(w4[j])
        realVe[j] = w1[j] * np.cos(w2[j])
        realIs[j] = w7[j] * np.cos(w8[j])
        imagVs[j] = w5[j] * np.sin(w6[j])
        imagIs[j] = w7[j] * np.sin(w8[j])
        realVs[j] = w5[j] * np.cos(w6[j])

    F1 = np.array(
            [-(2 / x[0]) * realIe[0] - (2 / x[0]) * realIs[0], (2 / x[0]) * imagIe[0] + (2 / x[0]) * imagIs[0],
             (realIe[0] + (x[0] / 2) * imagVe[0]) * x[2] + (-imagIe[0] + (x[0] / 2) * realVe[0]) * x[1],
             (imagIe[0] - (x[0] / 2) * realVe[0]) * x[2] + (realIe[0] + (x[0] / 2) * imagVe[0]) * x[1],
             (-(x[0] / 2) * imagVs[0] - realIs[0]) * x[2] + (-(x[0] / 2) * realVs[0] + imagIs[0]) * x[1],
             ((x[0] / 2) * realVs[0] - imagIs[0]) * x[2] + (-(x[0] / 2) * imagVs[0] - realIs[0]) * x[1]]) - np.array(
            [imagVe[0] + imagVs[0], realVe[0] + realVs[0], realVe[0] - realVs[0], imagVe[0] - imagVs[0],
             realVe[0] - realVs[0], imagVe[0] - imagVs[0]])
    F = F1
    for l in range(1, n):
        F1 = np.array(
            [-(2 / x[0]) * realIe[l] - (2 / x[0]) * realIs[l], (2 / x[0]) * imagIe[l] + (2 / x[0]) * imagIs[l],
            (realIe[l] + (x[0] / 2) * imagVe[l]) * x[2] + (-imagIe[l] + (x[0] / 2) * realVe[l]) * x[1],
            (imagIe[l] - (x[0] / 2) * realVe[l]) * x[2] + (realIe[l] + (x[0] / 2) * imagVe[l]) * x[1],
            (-(x[0] / 2) * imagVs[l] - realIs[l]) * x[2] + (-(x[0] / 2) * realVs[l] + imagIs[l]) * x[1],
            ((x[0] / 2) * realVs[l] - imagIs[l]) * x[2] + (-(x[0] / 2) * imagVs[l] - realIs[l]) * x[1]]) - np.array(
            [imagVe[l] + imagVs[l], realVe[l] + realVs[l], realVe[l] - realVs[l], imagVe[l] - imagVs[l],
            realVe[l] - realVs[l], imagVe[l] - imagVs[l]])
        F = np.concatenate((F, F1))
    return F

Rex = 5

Xex = 48.8

Bex = 3.371 * (10 ** (-4))

alpha=0.7

X0=np.ones(3)

X0[0]=alpha*Bex

X0[1]=alpha*Xex

X0[2]=alpha*Rex

res_1 = least_squares(fun_f,X0,method='lm')

Par_calc=res_1.x

desvB=abs((Par_calc[0]-Bex)/Bex)*100
desvX=abs((Par_calc[1]-Xex)/Xex)*100
desvR=abs((Par_calc[2]-Rex)/Rex)*100

print("O valor do desvio de B é : %.4f %s \n" %(desvB,'%'))
print("O valor do desvio de X é : %.4f %s \n" %(desvX,'%'))
print("O valor do desvio de R é : %.4f %s \n" %(desvR,'%'))










