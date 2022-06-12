"""
This code output the tunneling profile for different mass points.

The mixing angle is determined so that the fine-tuning is 5%.

For each mass points, output the 2d tunneling profile
at the nucleation temperature of the 1d case.

This basically represent the extra contribution to the action
from the light scalar.
"""
import csv
import os

import numpy as np

from light_scalar import model, model1d

mS_list = np.linspace(0.5,2,10).tolist() + np.linspace(2.1,10,20).tolist()

mH = 125.13

def sin_theta(mS):
    return (19*mS**2/(mH**2-mS**2))**0.5

for mS in mS_list:
    output_file_name = "profile_mS_" + str(round(mS,4)) + ".csv"
    output_file_path = os.path.join("./output/profile", output_file_name)
    m2d = model(mS,sin_theta(mS))
    m1d = model1d(mS, sin_theta(mS))
    print("mS: " + str(mS) + ", sin theta: " + str(sin_theta(mS)) + "\n")
    m1d.findTn()
    Tnuc_1d = m1d.Tn
    print("Tn = " + str(Tnuc_1d))
    tobj = m2d.tunneling_at_T(Tnuc_1d)
    R = tobj.profile1D.R
    hfield = tobj.Phi[:,0]
    Sfield = tobj.Phi[:,1]
    output = np.array([R.tolist(), hfield.tolist(), Sfield.tolist()]).transpose().tolist()
    with open(output_file_path, "w") as f:
        data_writer = csv.writer(f)
        data_writer.writerows(output)
