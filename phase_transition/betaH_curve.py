"""
This code scan over the light scalar masses along the parameter line
where the fine-tuning is fixed to be 10%.
Both 2d and 1d solution are computed.
Requires package of cosmoTransitions.
Author: Isaac Wang
"""

import csv

import numpy as np

from light_scalar import model, model1d
from light_scalar_interpolation import model as model_inter

mS_list = np.logspace(np.log10(0.05),np.log10(12),20).tolist()

mH = 125.13

def sin_theta(mS):
    return np.sqrt(9*mS**2/(mH**2-mS**2))


Interpolation= True

if Interpolation:
    output_file = "./output/betaH_inter.csv"
else:
    output_file = "./output/betaH.csv"



with open(output_file, "w") as f:
    data_writer = csv.writer(f)
    if Interpolation:
        print("Use interpolation to solve the S3/T!")
        mS_list = np.logspace(np.log10(0.01),np.log10(12),20).tolist()
        for mS in mS_list:
            m2d=model_inter(mS,sin_theta(mS))
            m1d=model1d(mS,sin_theta(mS))
            print("mS: " + str(mS) + ", sin theta: " + str(sin_theta(mS)) + "\n")
            betaH_1d = m1d.beta_over_H_at_Tn()
            betaH_2d = m2d.beta_over_H_at_Tn()
            print("mS: " + str(mS) + ", sin theta: " + str(sin_theta(mS)) + " beta\/H of 1d: " + str(betaH_1d) + " beta\/H of 2d: " + str(betaH_2d))
            output = [mS, betaH_1d, betaH_2d]
            data_writer.writerow(output)
    else:
        mS_list = np.logspace(np.log10(0.05),np.log10(12),20).tolist()
        for mS in mS_list:
            m2d=model(mS,sin_theta(mS))
            m1d=model1d(mS,sin_theta(mS))
            print("mS: " + str(mS) + ", sin theta: " + str(sin_theta(mS)) + "\n")
            betaH_1d = m1d.beta_over_H_at_Tn()
            betaH_2d = m2d.beta_over_H_at_Tn()
            print("mS: " + str(mS) + ", sin theta: " + str(sin_theta(mS)) + " beta\/H of 1d: " + str(betaH_1d) + " beta\/H of 2d: " + str(betaH_2d))
            output = [mS, betaH_1d, betaH_2d]
            data_writer.writerow(output)
