"""
This code scan over the light scalar masses along the parameter line
where the fine-tuning is fixed to be 10%.
Both 2d and 1d solution are computed.
Requires package of cosmoTransitions.
Author: Isaac Wang
"""

import csv

import numpy as np

from light_scalar_interpolation import model

input_list = []
with open("./Tcfix_input.csv") as inputfile:
    reader = csv.reader(inputfile,delimiter=",")
    for row in reader:
        mS = float(row[0])
        sin = float(row[1])
        input_list.append([mS,sin])

output_file = "./output/Tcfix_scan.csv"
with open(output_file,"w") as f:
    data_writer = csv.writer(f)
    for row in input_list:
        mS=row[0]
        sin=row[1]
        print("mS: " + str(mS) + ", sin theta: " + str(sin) + "\n")
        m = model(mS,sin)
        betaH_1d = m.beta_over_H_at_Tn_1d()
        betaH_2d = m.beta_over_H_at_Tn()
        Tn1d = m.Tn1d
        Tn2d = m.Tn
        print("mS: " + str(mS) + ", sin theta: " + str(sin) + " beta\/H of 1d: " + str(betaH_1d) + " beta\/H of 2d: " + str(betaH_2d))
        output = [mS, betaH_1d, betaH_2d, Tn1d, Tn2d]
        data_writer.writerow(output)
