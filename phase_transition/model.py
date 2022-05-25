"""
This file write the effective potential in high-T expansion.
Author: Isaac Wang
Requires package of cosmoTransitions.
"""

import numpy as np
from cosmoTransitions import generic_potential as gp

# -----------------------------------------------------------------
# Physical Constants
# -----------------------------------------------------------------
GF = 1.16637e-05 # Fermi Constant
v = 1/(np.sqrt(2*np.sqrt(2)*GF)) # Higgs vev
mHSM = 125.13 # Higgs mass

# -----------------------------------------------------------------
# Transfer between physical parameters and bare parameters
# -----------------------------------------------------------------
def muH2(mS, sintheta):
    """\mu_H square"""
    return 0.5*(mHSM**2 * (1-sintheta**2) + mS**2 * sintheta**2)

def muS2(mS, sintheta):
    """\mu_S square"""
    return sintheta**2 * mHSM**2 + (1 - sintheta**2) * mS**2

def A(mS, sintheta):
    """A parameter"""
    nominator = (mHSM**2 - mS**2) * sintheta * np.sqrt(1-sintheta**2)
    denominator = np.sqrt(2) * v
    return nominator/denominator

def lm(mS, sintheta):
    """\lambda parameter"""
    nominator = (1 - sintheta**2)*mHSM**2 + sintheta**2 * mS**2
    return nominator/(4*v**2)

# -----------------------------------------------------------------
# Define effective potential
# -----------------------------------------------------------------
class model(gp.generic_potential):
    def init(self, mS, sintheta):
        self.Ndim = 2
        self.Tmax = 100
        self.mS = mS
        self.sintheta = sintheta
        self.lm = lm(self.mS, self.sintheta)
        self.A = A(self.mS,self.sintheta)
        self.muH2 = muH2(self.mS,self.sintheta)
        self.muS2 = muS2(self.mS,self.sintheta)
        self.g = 0.65
        self.gY = 0.36
        self.yt = 0.9945
        self.D = (3*self.g**2 + self.gY**2 + 4*self.yt**2)/16.
        self.E = (2*self.g**3+(self.g**2 + self.gY**2)**(3/2))/(48*np.pi)
        self.cs = 1./3
        self.Deff = self.D - self.cs * self.A**2/(4.*self.muS2)
        self.lmeff = self.lm - self.A**2/(2*self.muS2)
        self.T0 = np.sqrt(0.5*self.muH2 - v**2 * self.A**2 /(2*self.muS2))/np.sqrt(self.D - self.cs*self.A**2/(4*self.muS2))
        self.Tc = self.T0*np.sqrt((self.Deff * self.lmeff)/(-self.E**2 + self.Deff*self.lmeff))
        self.strength = 2*self.E/self.lmeff

    def Vtot1d(self, X, T, include_radiation = True):
        T = np.asanyarray(T, dtype=float)
        X = np.asanyarray(X, dtype=float)
        T2 = (T*T) + 1e-100
        phi1 = X[...,0]
        y = self.Deff * T2 * phi1**2 - (0.5*self.muH2 - 0.5 * v**2 * self.A**2 / (self.muS2))*phi1**2
        y += - self.E * T * phi1 **3
        y += 0.25 * self.lmeff * phi1**4
        return y



    def Vtot(self, X, T, include_radiation=True):
        T = np.asanyarray(T, dtype=float)
        X = np.asanyarray(X, dtype=float)
        T2 = (T*T)+1e-100
        phi1 = X[...,0]
        phi2 = X[...,1]
        y = self.D * T2 * phi1**2 - 0.5 * self.muH2 * phi1**2
        y += - self.E * T * phi1**3
        y += 0.25 * self.lm * phi1**4
        y += 0.5*self.muS2*phi2**2 - 0.5 * self.A * (phi1**2 + self.cs * T2 - 2 * v**2)*phi2
        return y
