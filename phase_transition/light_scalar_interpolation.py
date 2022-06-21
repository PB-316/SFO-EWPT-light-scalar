"""
This file write the effective potential in high-T expansion.
Author: Isaac Wang
Requires package of cosmoTransitions.
"""

import numpy as np
from cosmoTransitions import generic_potential as gp
from cosmoTransitions import helper_functions
from cosmoTransitions import pathDeformation as pd
from scipy import interpolate, optimize

# -----------------------------------------------------------------
# Physical Constants
# -----------------------------------------------------------------
GF = 1.16637e-05  # Fermi Constant
v = 1 / (np.sqrt(2 * np.sqrt(2) * GF))  # Higgs vev
mHSM = 125.13  # Higgs mass

# -----------------------------------------------------------------
# Transfer between physical parameters and bare parameters
# -----------------------------------------------------------------
def muH2(mS, sintheta):
    """\mu_H square"""
    return 0.5 * (mHSM**2 * (1 - sintheta**2) + mS**2 * sintheta**2)


def muS2(mS, sintheta):
    """\mu_S square"""
    return sintheta**2 * mHSM**2 + (1 - sintheta**2) * mS**2


def A(mS, sintheta):
    """A parameter"""
    nominator = (mHSM**2 - mS**2) * sintheta * np.sqrt(1 - sintheta**2)
    denominator = np.sqrt(2) * v
    return nominator / denominator


def lm(mS, sintheta):
    """\lambda parameter"""
    nominator = (1 - sintheta**2) * mHSM**2 + sintheta**2 * mS**2
    return nominator / (4 * v**2)


# -----------------------------------------------------------------
# Define effective potential
# -----------------------------------------------------------------
class model(gp.generic_potential):
    """Effective potential, and some defined functions."""

    def init(self, mS, sintheta):
        self.Ndim = 2
        self.Tmax = 100
        self.mS = mS
        self.sintheta = sintheta
        self.lm = lm(self.mS, self.sintheta)
        self.A = A(self.mS, self.sintheta)
        self.muH2 = muH2(self.mS, self.sintheta)
        self.muS2 = muS2(self.mS, self.sintheta)
        self.g = 0.65
        self.gY = 0.36
        self.yt = 0.9945
        self.D = (3 * self.g**2 + self.gY**2 + 4 * self.yt**2) / 16.0
        self.E = (2 * self.g**3 + (self.g**2 + self.gY**2) ** (3 / 2)) / (
            48 * np.pi
        )
        self.cs = 1.0 / 3
        self.Deff = self.D - self.cs * self.A**2 / (4.0 * self.muS2)
        self.lmeff = self.lm - self.A**2 / (2 * self.muS2)
        self.T0 = np.sqrt(
            0.5 * self.muH2 - v**2 * self.A**2 / (2 * self.muS2)
        ) / np.sqrt(self.D - self.cs * self.A**2 / (4 * self.muS2))
        self.Tc = self.T0 * np.sqrt(
            (self.Deff * self.lmeff) / (-self.E**2 + self.Deff * self.lmeff)
        )
        self.strength = 2 * self.E / self.lmeff
        self.Tn = False
        self.Tn1d = False
        self.non_restore_trigger = self.Deff * self.lmeff - self.E**2

    def Vtot(self, X, T, include_radiation=True):
        T = np.asanyarray(T, dtype=float)
        X = np.asanyarray(X, dtype=float)
        T2 = (T * T) + 1e-100
        phi1 = X[..., 0]
        phi2 = X[..., 1]
        y = self.D * T2 * phi1**2 - 0.5 * self.muH2 * phi1**2
        y += -self.E * T * phi1**3
        y += 0.25 * self.lm * phi1**4
        y += (
            0.5 * self.muS2 * phi2**2
            - 0.5 * self.A * (phi1**2 + self.cs * T2 - 2 * v**2) * phi2
        )
        return y

    def Vtot1d(self, X, T, include_radiation=True):
        T = np.asanyarray(T, dtype=float)
        X = np.asanyarray(X, dtype=float)
        T2 = (T * T) + 1e-100
        phi1 = X[..., 0]
        y = (
            self.Deff * T2 * phi1**2
            - (0.5 * self.muH2 - 0.5 * v**2 * self.A**2 / (self.muS2)) * phi1**2
        )
        y += -self.E * T * phi1**3
        y += 0.25 * self.lmeff * phi1**4
        return y

    def gradV1d(self, X, T):
        f = helper_functions.gradientFunction(
            self.Vtot1d, self.x_eps, 1, self.deriv_order
        )
        T = np.asanyarray(T)[..., np.newaxis, np.newaxis]
        return f(X, T, False)

    def truevev(self, T):
        assert T < self.Tc
        nominator = 3.0 * T * self.E + np.sqrt(
            9.0 * self.E**2 * T**2
            + 8.0 * self.Deff * (self.T0**2 - T**2) * self.lmeff
        )
        denominator = 2.0 * self.lmeff
        return nominator / denominator

    def Spath(self, X, T):
        X = np.asanyarray(X)
        T = np.asanyarray(T)
        phi1 = X[..., 0]
        T2 = (T * T) + 1e-100
        return 0.5 * self.A * (phi1**2 + self.cs * T2 - 2 * v**2) / self.muS2

    def tunneling_at_T_1d(self, T):
        assert T < self.Tc

        def V_(x, T=T, V=self.Vtot1d):
            return V(x, T)

        def dV_(x, T=T, dV=self.gradV1d):
            return dV(x, T)

        tobj = pd.fullTunneling([[self.truevev(T)], [0.0]], V_, dV_)
        return tobj

    def tunneling_at_T(self, T):
        assert T < self.Tc

        def V_(x, T=T, V=self.Vtot):
            return V(x, T)

        def dV_(x, T=T, dV=self.gradV):
            return dV(x, T)

        # tobj = pd.fullTunneling([self.findMinimum(T=T),[0,self.Spath([0],T)]],V_,dV_)
        tobj = pd.fullTunneling(
            [
                [self.truevev(T=T), self.Spath([self.truevev(T=T)], T)],
                [1e-100, self.Spath([1e-100], T)],
            ],
            V_,
            dV_,
        )
        return tobj

    def S_over_T(self, T):
        Tv = T
        ST = self.tunneling_at_T(T=Tv).action / Tv
        return ST

    def findTn_1d(self):
        print("Finding nucleation temperature for 1d case...")
        if self.mS <= 0.1:
            eps = 0.03
        elif self.mS <= 1:
            eps = 0.02
        else:
            eps = 0.01

        def nuclea_trigger(Tv):
            ST = self.tunneling_at_T_1d(T=Tv).action / Tv
            return ST - 140.0

        for i in range(1, 1000):
            if nuclea_trigger(self.Tc - i * eps) <= 0.0:
                break
        Tn1 = self.Tc - (i - 1) * eps
        self.Tn1d = optimize.brentq(nuclea_trigger, Tn1 - 1e-10, Tn1 - eps, disp=False)

    def trace_action(self):
        if self.mS <= 1:
            if self.Tn1d == False:
                self.findTn_1d()
            Tmax = self.Tn1d - 0.01
        elif self.mS <= 0.05:
            if self.Tn1d == False:
                self.findTn_1d()
            Tmax = self.Tn1d - 0.05
        else:
            Tmax = self.Tc - 0.01
        eps = 0.002
        list = []
        for i in range(0, 1000):
            Ttest = Tmax - i * eps
            print("Tunneling at T=" + str(Ttest))
            trigger = self.S_over_T(Ttest)
            print("S3/T=" + str(trigger))
            list.append([Ttest, trigger])
            if trigger < 140.0:
                break
        Tmin = Ttest
        print("Tnuc should be within " + str(Tmin) + " and " + str(Tmin + eps))
        self.action_trace_data = np.array(list).transpose().tolist()

    def findTn(self):
        self.trace_action()
        Tlist = self.action_trace_data[0]
        log_action = [np.log10(i) - np.log10(140) for i in self.action_trace_data[1]]
        # trigger_list=[i-140 for i in self.action_trace_data[1]]
        # Action_drop = interpolate.interp1d(Tlist,trigger_list, kind='cubic')
        function = interpolate.interp1d(Tlist, log_action, kind="cubic")
        # self.Tn = optimize.brentq(Action_drop, Tlist[-2], Tlist[-1],disp=False,xtol=1e-5,rtol=1e-6)
        self.Tn = optimize.brentq(
            function, Tlist[-2], Tlist[-1], disp=False, xtol=1e-5, rtol=1e-6
        )

    def strength_Tn(self):
        if not self.Tn:
            self.findTn()
        Tnuc = self.Tn
        return self.truevev(T=Tnuc) / Tnuc

    def strength_Tn1d(self):
        if not self.Tn1d:
            self.findTn_1d()
        Tnuc = self.Tn1d
        return self.truevev(T=Tnuc) / Tnuc

    def beta_over_H_at_Tn(self):
        "Ridders algorithm"
        if not self.Tn:
            self.findTn()
        Tnuc = self.Tn
        if self.action_trace_data == []:
            self.trace_action()
        Tlist = self.action_trace_data[0]
        trigger_list = [i - 140 for i in self.action_trace_data[1]]
        Action_drop = interpolate.interp1d(Tlist, trigger_list, kind="cubic")
        eps = 0.5 * (Tnuc - Tlist[-1]) * 0.9
        dev = (
            Action_drop(Tnuc - 2.0 * eps)
            - 8.0 * Action_drop(Tnuc - eps)
            + 8.0 * Action_drop(Tnuc + eps)
            - Action_drop(Tnuc + 2.0 * eps)
        ) / (12.0 * eps)
        return dev * Tnuc

    def beta_over_H_at_Tn_1d(self):
        "Ridders algorithm"
        if not self.Tn1d:
            self.findTn_1d()
        Tnuc = self.Tn1d
        eps = 1e-5

        def SoverT(Tv):
            ST = self.tunneling_at_T_1d(T=Tv).action / Tv
            return ST

        dev = (
            SoverT(Tnuc - 2.0 * eps)
            - 8.0 * SoverT(Tnuc - eps)
            + 8.0 * SoverT(Tnuc + eps)
            - SoverT(Tnuc + 2.0 * eps)
        ) / (12.0 * eps)
        return dev * Tnuc

    def alpha(self):
        if not self.Tn:
            self.findTn()
        Tnuc = self.Tn
        if self.Tc - Tnuc >= 0.002:
            eps = 0.001
        else:
            eps = 0.0001

        def deltaV(T):
            falsev = [0, self.Spath([0], T)]
            truev = self.truevev(T)
            return self.Vtot(falsev, T) - self.Vtot(truev, T)

        dev = (
            deltaV(Tnuc - 2 * eps)
            - 8.0 * deltaV(Tnuc - eps)
            + 8.0 * deltaV(Tnuc + eps)
            - deltaV(Tnuc + 2.0 * eps)
        ) / (
            12.0 * eps
        )  # derivative of deltaV w.r.t T at Tn
        latent = deltaV(Tnuc) - 0.25 * Tnuc * dev
        rho_crit = np.pi**2 * 106.75 * Tnuc**4 / 30.0
        return latent / rho_crit
