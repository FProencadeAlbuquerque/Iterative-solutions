import pandas as pd
import numpy as np
from numpy.linalg import inv
import math as m
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 25})

df = pd.read_csv('Dados90.csv')

VEmod=df['Vemod'][399]
VEfase=(df['Vefase'][399]*m.pi/180)
IEmod=df['Iemod'][399]
IEfase=(df['Iefase'][399]*m.pi/180)
VSmod=df['Vsmod'][399]
VSfase=(df['Vsfase'][399]*m.pi/180)
ISmod=df['Ismod'][399]
ISfase=(df['Isfase'][399]*m.pi/180)

Rex=5

Xex=48.8

Bex=3.371*(10**(-4))

n=5000

sigma = 0.04

thetamax = 8 * (10 ** (-3))

itmax = 4

alpha = 0.7

B = np.zeros(itmax + 1)

B[0] = alpha * Bex

R = np.zeros(itmax + 1)

R[0] = alpha * Rex

X = np.zeros(itmax + 1)

X[0] = alpha * Xex

np.random.seed(42)
w1=VEmod*np.ones(n)+VEmod*(sigma/3)*np.random.normal(0,1,n)
np.random.seed(41)
w2=VEfase+(thetamax/3)*np.random.normal(0,1,n)
np.random.seed(40)
w3=IEmod*np.ones(n)+IEmod*(sigma/3)*np.random.normal(0,1,n)
np.random.seed(39)
w4=IEfase+(thetamax/3)*np.random.normal(0,1,n)
np.random.seed(38)
w5=VSmod*np.ones(n)+VSmod*(sigma/3)*np.random.normal(0,1,n)
np.random.seed(37)
w6=VSfase+(thetamax/3)*np.random.normal(0,1,n)
np.random.seed(36)
w7=ISmod*np.ones(n)+ISmod*(sigma/3)*np.random.normal(0,1,n)
np.random.seed(35)
w8=ISfase+(thetamax/3)*np.random.normal(0,1,n)

realIe=np.ones(n)
imagVe=np.ones(n)
imagIe=np.ones(n)
realVe=np.ones(n)
realIs=np.ones(n)
imagVs=np.ones(n)
imagIs=np.ones(n)
realVs=np.ones(n)

for j in range(0,n):
    realIe[j]=w3[j]*np.cos(w4[j])
    imagVe[j]=w1[j]*np.sin(w2[j])
    imagIe[j]=w3[j]*np.sin(w4[j])
    realVe[j]=w1[j]*np.cos(w2[j])
    realIs[j]=w7[j]*np.cos(w8[j])
    imagVs[j]=w5[j]*np.sin(w6[j])
    imagIs[j]=w7[j]*np.sin(w8[j])
    realVs[j]=w5[j]*np.cos(w6[j])

for k in range(0, itmax):
    F1 = np.array([-(2 / B[k]) * realIe[0] - (2 / B[k]) * realIs[0], (2 / B[k]) * imagIe[0] + (2 / B[k]) * imagIs[0],
                   (realIe[0] + (B[k] / 2) * imagVe[0]) * R[k] + (-imagIe[0] + (B[k] / 2) * realVe[0]) * X[k],
                   (imagIe[0] - (B[k] / 2) * realVe[0]) * R[k] + (realIe[0] + (B[k] / 2) * imagVe[0]) * X[k],
                   (-(B[k] / 2) * imagVs[0] - realIs[0]) * R[k] + (-(B[k] / 2) * realVs[0] + imagIs[0]) * X[k],
                   ((B[k] / 2) * realVs[0] - imagIs[0]) * R[k] + (-(B[k] / 2) * imagVs[0] - realIs[0]) * X[
                       k]]) - np.array(
        [imagVe[0] + imagVs[0], realVe[0] + realVs[0], realVe[0] - realVs[0], imagVe[0] - imagVs[0],
         realVe[0] - realVs[0], imagVe[0] - imagVs[0]])
    P1 = [realIe[0] * (2 / (B[k] ** 2)) + realIs[0] * (2 / (B[k] ** 2)), 0, 0]
    P2 = [-(2 / B[k] ** 2) * imagIe[0] - (2 / B[k] ** 2) * imagIs[0], 0, 0]
    P3 = [(R[k] * imagVe[0]) / 2 + (X[k] * realVe[0]) / 2, realIe[0] + (B[k] / 2) * imagVe[0],
          -imagIe[0] + (B[k] / 2) * realVe[0]]
    P4 = [(-R[k] * realVe[0]) / 2 + (imagVe[0] * X[k]) / 2, imagIe[0] - (B[k] / 2) * realVe[0],
          realIe[0] + (B[k] / 2) * imagVe[0]]
    P5 = [-(imagVs[0] * R[k]) / 2 - (realVs[0] * X[k]) / 2, -(B[k] / 2) * imagVs[0] - realIs[0],
          -(B[k] / 2) * realVs[0] + imagIs[0]]
    P6 = [(realVs[0] * R[k]) / 2 - (imagVs[0] * X[k]) / 2, (B[k] / 2) * realVs[0] - imagIs[0],
          -(B[k] / 2) * imagVs[0] - realIs[0]]

    DF1 = np.array([P1, P2, P3, P4, P5, P6])

    F = F1

    DF = DF1

    for l in range(1, n):
        F1 = np.array(
            [-(2 / B[k]) * realIe[l] - (2 / B[k]) * realIs[l], (2 / B[k]) * imagIe[l] + (2 / B[k]) * imagIs[l],
             (realIe[l] + (B[k] / 2) * imagVe[l]) * R[k] + (-imagIe[l] + (B[k] / 2) * realVe[l]) * X[k],
             (imagIe[l] - (B[k] / 2) * realVe[l]) * R[k] + (realIe[l] + (B[k] / 2) * imagVe[l]) * X[k],
             (-(B[k] / 2) * imagVs[l] - realIs[l]) * R[k] + (-(B[k] / 2) * realVs[l] + imagIs[l]) * X[k],
             ((B[k] / 2) * realVs[l] - imagIs[l]) * R[k] + (-(B[k] / 2) * imagVs[l] - realIs[l]) * X[k]]) - np.array(
            [imagVe[l] + imagVs[l], realVe[l] + realVs[l], realVe[l] - realVs[l], imagVe[l] - imagVs[l],
             realVe[l] - realVs[l], imagVe[l] - imagVs[l]])
        P1 = [realIe[l] * (2 / (B[k] ** 2)) + realIs[l] * (2 / (B[k] ** 2)), 0, 0]
        P2 = [-(2 / B[k] ** 2) * imagIe[l] - (2 / B[k] ** 2) * imagIs[l], 0, 0]
        P3 = [(R[k] * imagVe[l]) / 2 + (X[k] * realVe[l]) / 2, realIe[l] + (B[k] / 2) * imagVe[l],
              -imagIe[l] + (B[k] / 2) * realVe[l]]
        P4 = [(-R[k] * realVe[l]) / 2 + (imagVe[l] * X[k]) / 2, imagIe[l] - (B[k] / 2) * realVe[l],
              realIe[l] + (B[k] / 2) * imagVe[l]]
        P5 = [-(imagVs[l] * R[k]) / 2 - (realVs[l] * X[k]) / 2, -(B[k] / 2) * imagVs[l] - realIs[l],
              -(B[k] / 2) * realVs[l] + imagIs[l]]
        P6 = [(realVs[l] * R[k]) / 2 - (imagVs[l] * X[k]) / 2, (B[k] / 2) * realVs[l] - imagIs[l],
              -(B[k] / 2) * imagVs[l] - realIs[l]]

        DF1 = np.array([P1, P2, P3, P4, P5, P6])

        F = np.concatenate((F, F1))

        DF = np.concatenate((DF, DF1))

    DF_T = DF.transpose()

    Aux1 = inv(np.dot(DF_T, DF))

    Aux2 = np.dot(DF_T, F)

    H = np.array([B[k], R[k], X[k]]) - np.dot(Aux1, Aux2)

    B[k + 1] = H[0]
    R[k + 1] = H[1]
    X[k + 1] = H[2]

#### Construçao dos residuos #############

res1=np.ones(n)
res2=np.ones(n)
res3=np.ones(n)
res4=np.ones(n)
res5=np.ones(n)
res6=np.ones(n)

for m in range(0,n):
    res=np.array([-(2/B[itmax])*realIe[m]-(2/B[itmax])*realIs[m],(2/B[itmax])*imagIe[m]+(2/B[itmax])*imagIs[m],(realIe[m]+(B[itmax]/2)*imagVe[m])*R[itmax]+(-imagIe[m]+(B[itmax]/2)*realVe[m])*X[itmax],(imagIe[m]-(B[itmax]/2)*realVe[m])*R[itmax]+(realIe[m]+(B[itmax]/2)*imagVe[m])*X[itmax],(-(B[itmax]/2)*imagVs[m]-realIs[m])*R[itmax]+(-(B[itmax]/2)*realVs[m]+imagIs[m])*X[itmax],((B[itmax]/2)*realVs[m]-imagIs[m])*R[itmax]+(-(B[itmax]/2)*imagVs[m]-realIs[m])*X[itmax]])-np.array([imagVe[m]+imagVs[m],realVe[m]+realVs[m],realVe[m]-realVs[m],imagVe[m]-imagVs[m],realVe[m]-realVs[m],imagVe[m]-imagVs[m]])
    res1[m]=res[0]
    res2[m]=res[1]
    res3[m]=res[2]
    res4[m]=res[3]
    res5[m]=res[4]
    res6[m]=res[5]

desvB=abs((B[itmax]-Bex)/Bex)*100
desvX=abs((X[itmax]-Xex)/Xex)*100
desvR=abs((R[itmax]-Rex)/Rex)*100

print("O valor do desvio de B é : %.4f %s \n" %(desvB,'%'))
print("O valor do desvio de X é : %.4f %s \n" %(desvX,'%'))
print("O valor do desvio de R é : %.4f %s \n" %(desvR,'%'))

axis_font = {'fontname':'Arial', 'size':'25'}

n, bins, patches = plt.hist(x=res1, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value',**axis_font)
plt.ylabel('Absolute Frequency',**axis_font)
plt.title('Residuals Histogram for the first function')
plt.show()



