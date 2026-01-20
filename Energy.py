
import logging
import os
import warnings


import astropy
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.integrate import quad

import astropy.constants as con
from astropy.constants import alpha, c, e, hbar, m_e, m_p, sigma_sb

m_e = con.m_e.cgs.value
c = con.c.cgs.value
h = con.h.cgs.value
el = con.e.gauss.value
erg_to_eV = 624150912588.3258  # conversion from erg to eV
sigma_T = con.sigma_T.cgs.value
mpc2 = (con.m_p * con.c ** 2.).to('eV')
mpc2_erg = mpc2.to('erg').value
mec2_eV = (con.m_e * con.c ** 2.).to('eV').value

###################################################### Priofiles #####################################################


def E_theta_gaussian(theta, thetaw_deg=12.0,theta_core = 5.0 , Eiso_zero=1.0):
    
    theta_rad = np.deg2rad(theta)
    thetacore_rad = np.deg2rad(theta_core)
    thetaw_rad = np.deg2rad(thetaw_deg)

    # supporta scalar e array: convertiamo in array temporaneo
    theta_rad_arr = np.asarray(theta_rad)
    E = np.zeros_like(theta_rad_arr, dtype=float)
    inside = theta_rad_arr <= thetaw_rad
    E[inside] = Eiso_zero * np.exp(-(theta_rad_arr[inside]**2) / (2 * thetacore_rad**2))
    E = np.maximum(E, 1e-40)  # floor per evitare problemi numerici
    # se input era scalar, ritorniamo scalar
    return E

    
def E_theta_powerlaw(theta, thetaw_deg=20.0,theta_core = 5.0,Eiso_zero=1.0,b=4.5):

        theta_rad = np.deg2rad(theta)
        thetacore_rad = np.deg2rad(theta_core)
        thetaw_rad = np.deg2rad(thetaw_deg)

        # supporta scalar e array: convertiamo in array temporaneo
        theta_rad_arr = np.asarray(theta_rad)
        E = np.zeros_like(theta_rad_arr, dtype=float)
        inside = theta_rad_arr <= thetaw_rad

        E[inside] = Eiso_zero * (1 + (theta_rad[inside]**2) / (b * thetacore_rad**2))**(-b/2)
        E = np.maximum(E, 1e-40)#floor to avoid numerical issues
        return E

    
def E_theta_flat(theta, thetaw_deg=90.0, Eiso_zero=1.0):

    theta_rad = np.deg2rad(theta)
    thetaw_rad = np.deg2rad(thetaw_deg)

    E = np.zeros_like(theta_rad)
    inside = theta_rad <= thetaw_rad
    E[inside] = Eiso_zero  # costante all'interno
    E = np.maximum(E, 1e-40)  # floor numerico
    return E

##############################################################################################

def Eiso0_from_Ek_with_profile(Ek, profile_func, profile_kwargs=None, theta_max=np.pi/2):

    if profile_kwargs is None:
        profile_kwargs = {}

    # integranda: dato theta in radianti (come quad fornisce), convertiamo in gradi
    def integrand(theta_rad):
        theta_deg = np.rad2deg(theta_rad)
        # profilo normalizzato a Eiso_zero = 1
        val = profile_func(theta_deg, **profile_kwargs, Eiso_zero=1.0)
        # quad passa float -> val deve essere float
        return float(val) * np.sin(theta_rad)

    # integriamo da 0 a theta_max (rad)
    I, err = quad(integrand, 0.0, theta_max, epsabs=1e-9, epsrel=1e-8, limit=200)
    if I <= 0:
        raise ValueError("Integrale del profilo non positivo (controlla profile_func / parametri).")

    # E_iso(0)
    Eiso0 = 2.0 * Ek / I
    return Eiso0

def Ek_from_profile(Eiso0, profile_func, profile_kwargs=None, theta_max=np.pi/2):
    """
    Calcola l'energia totale Ek di un getto dato il profilo Eiso(theta) e Eiso(0).
    Only 1 jety
    """
    if profile_kwargs is None:
        profile_kwargs = {}

    # integranda: theta in radianti → convertiamo in gradi
    def integrand(theta_rad):
        theta_deg = np.rad2deg(theta_rad)
        # profilo normalizzato a 1
        val = profile_func(theta_deg, **profile_kwargs, Eiso_zero=1.0)
        return float(val) * np.sin(theta_rad)

    integral, err = quad(integrand, 0.0, theta_max, epsabs=1e-9, epsrel=1e-8, limit=200)
    
    # energia totale Ek
    Ek = 0.5 * Eiso0 * integral
    return Ek


def plot_energy_profile(
    theta_deg_max=30,
    profiles=None,
    profile_kwargs=None,
    Eiso0_dict=None,
    bottom_order=3,
    Save=False,
    Name="Energy_profile",
    path="./"):
    
    if profiles is None:
        raise ValueError("Devi fornire almeno un profilo nel dict 'profiles'.")

    theta = np.linspace(0,theta_deg_max, 500)
    
    plt.figure(figsize=(9,6))

    for name, func in profiles.items():
        # parametri del profilo specifico
        kwargs = profile_kwargs.get(name, {}) if profile_kwargs else {}
        Eiso0 = 1.0 if Eiso0_dict is None else Eiso0_dict.get(name, 1.0)

        E = func(theta, **kwargs, Eiso_zero=Eiso0)
        plt.plot(theta, E, label=f"{name} profile")
        # limiti log
    
    all_E = np.concatenate([func(theta, **(profile_kwargs.get(name, {}) if profile_kwargs else {}), 
                                  Eiso_zero=Eiso0_dict.get(name, 1.0) if Eiso0_dict else 1.0)
                            for name, func in profiles.items()])
     
    
    Emax = np.nanmax(all_E)
    ymax = 10**np.ceil(np.log10(Emax))
    ymin = ymax / (10**bottom_order)
    
    plt.xlabel("θ (grades)")
    plt.ylabel(r"$E_{\mathrm{iso}}(\theta)$ [erg]")
    plt.title("Angular Energy Profile", fontsize=15)
    plt.yscale("log")
    plt.ylim(ymin, ymax)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.title(f"{Name}",fontsize=15)

    if Save:

        plt.savefig(f"{path}SED_{Name}.jpg", format="jpg", dpi=300)    
        print(f"Plot saved as: {path}SED_{Name}.png/pdf")
    
    plt.show()