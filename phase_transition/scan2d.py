"""
Scan over the parameter space.
Author: Isaac Wang
"""
import csv

import numpy as np

from light_scalar import model, model1d

mS_list = np.linspace(1.0,12, 24).tolist()
sin_list = np.linspace(0.01,0.3, 15).tolist()

one_dimension = False

if one_dimension:
    print("One dimensional case...\n")
    csv_file_path = "./output/scan1d.csv"
    with open(csv_file_path, "w") as f:
        header = ["mS", "sin theta", "alpha", "beta over H"]
        data_writer = csv.writer(f)
        data_writer.writerow(header)
        for mS in mS_list:
            for sin in sin_list:
                m = model1d(mS,sin)
                if m.strength <= 1.0:
                    print("Not strong 1st order.\n")
                    continue
                elif m.non_restore_trigger <=0:
                    print("Symmetry not restored.\n")
                    continue
                m.findAllTransitions()
                alpha = m.alpha()
                beta_H = m.beta_over_H_at_Tn()
                data_writer.writerow([mS, sin, alpha, beta_H])
                print("mS = " + str(mS) + ", sin\\theta = " + str(sin) + ", alpha = " + str(alpha) + ", beta\/H = " + str(beta_H) + "\n")



else:
    print("Two dimensional case...\n")
    csv_file_path = "./output/scan.csv"
    with open(csv_file_path, "w") as f:
        header = ["mS", "sin theta", "alpha", "beta over H"]
        data_writer = csv.writer(f)
        data_writer.writerow(header)
        for mS in mS_list:
            for sin in sin_list:
                print("mS = " + str(mS) + ", sin\\theta = " + str(sin) + "\n")
                m = model(mS,sin)
                ##if m.strength <= 1.0:
                ##    print("Not strong 1st order.\n")
                ##    continue
                if m.non_restore_trigger <=0:
                    print("Symmetry not restored.\n")
                    continue
                elif m.strength >= 20:
                    print("Too strong! Encoutered numerical issue, skip.")
                    continue
                m.findTn()
                alpha = m.alpha()
                beta_H = m.beta_over_H_at_Tn()
                Tnuc = m.Tn
                vev = m.truevev(T=Tnuc)
                strength_Tn = m.strength_Tn()
                data_writer.writerow([mS, sin, Tnuc, vev, strength_Tn, alpha, beta_H])
                print("mS = " + str(mS) + ", sin\\theta = " + str(sin) + "Tnuc = " + str(Tnuc) + "Higgs vev = " + str(vev) + "strength at Tn: " + str(strength_Tn) +", alpha = " + str(alpha) + ", beta\/H = " + str(beta_H) + "\n")
