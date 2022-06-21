import csv

import numpy as np

from light_scalar_interpolation import model

mH = 125.13
mS_list = np.logspace(np.log10(0.001),np.log10(0.05), 10).tolist()
ft_list = np.linspace(0.09,0.15,3).tolist()

def sintheta(mS, ft):
    """Solve the mixing angle according to find-tuning"""
    return (mS**2*(1/ft-1)/(mH**2-mS**2))**0.5

output_file = "./output/Tn_superlight.csv"

with open(output_file, "w") as f:
    data_writer = csv.writer(f)
    for mS in mS_list:
        for ft in ft_list:
            sin = sintheta(mS,ft)
            print("Scanning mS = " + str(mS) + ", sin theta = " + str(sin))
            m = model(mS,sin)
            m.findTn_1d()
            Tn1d = m.Tn1d
            strength1d = m.strength_Tn1d()
            data_writer.writerow([mS,sin,Tn1d,strength1d])
            print("mS = " + str(mS) + " sin theta = " + str(sin) + " scanning done.")
