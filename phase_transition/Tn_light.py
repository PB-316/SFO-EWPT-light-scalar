"""
Scan over the parameter space around the boundary of SFOPT. The goal is to see
how small mass can affect the nucleation temperature.
"""

import csv

import numpy as np

from light_scalar_interpolation import model

mH = 125.13
mS_list = np.logspace(np.log10(0.05),np.log10(13), 20).tolist()
ft_list = np.linspace(0.1,0.15,5).tolist()

def sintheta(mS, ft):
    """Solve the mixing angle according to find-tuning"""
    return (mS**2*(1/ft-1)/(mH**2-mS**2))**0.5

output_file = "./output/Tn_bound.csv"

with open(output_file, "w") as f:
    data_writer = csv.writer(f)
    for mS in mS_list:
        for ft in ft_list:
            sin = sintheta(mS,ft)
            print("Scanning mS = " + str(mS) + ", sin theta = " + str(sin))
            m = model(mS,sin)
            m.findTn()
            Tn1d = m.Tn1d
            Tn2d = m.Tn
            strength1d = m.strength_Tn1d()
            strength2d = m.strength_Tn()
            data_writer.writerow([mS,sin,Tn1d,strength1d,Tn2d,strength2d])
            print("mS = " + str(mS) + " sin theta = " + str(sin) + " scanning done.")
