import astropy
from astropy.table import Table, hstack
import astropy.units as u
from astropy.io import ascii
from astropy.constants import alpha, c, e, hbar, m_e, m_p, sigma_sb
from astropy.utils.data import get_pkg_data_filename
from astropy.cosmology import WMAP9 as cosmo

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

from Validator import (
    validate_array,
    validate_physical_type,
    validate_scalar,
)
#from .model_utils import memoize
from Utils import trapz_loglog

import Models
import RadiativeNEW
import grbloader
from GRBmodel import GRBModel1
from GRBmodelstr import GRBModel_topstruc
from Models import EblAbsorptionModel

from astropy.units import def_physical_type

try:
    #def_physical_type(u.Unit("1 / eV"), "differential energy")
    def_physical_type(u.erg / u.cm**2 / u.s, "flux")
    def_physical_type(u.Unit("1/(s cm2 erg)"), "differential flux")
    def_physical_type(u.Unit("1/(s erg)"), "differential power")
    def_physical_type(u.Unit("1/TeV"), "differential energy")
    def_physical_type(u.Unit("1/cm3"), "number density")
    def_physical_type(u.Unit("1/(eV cm3)"), "differential number density")

except ValueError:
    print("New quantities already defined")


import astropy.constants as con
from astropy.constants import alpha, c, e, hbar, m_e, m_p, sigma_sb
m_e = con.m_e.cgs.value
c = con.c.cgs.value
mec2_eV = (con.m_e * con.c ** 2.).to('eV').value
h = con.h.cgs.value
el = con.e.gauss.value
erg_to_eV = 624150912588.3258  # conversion from erg to eV
sigma_T = con.sigma_T.cgs.value
mpc2 = (con.m_p * con.c ** 2.).to('eV')
mpc2_erg = mpc2.to('erg').value

def Ek_from_Eiso(Eiso, theta_c):
    """
    Calcola l'energia cinetica totale di un jet top-hat
    dato l'E_iso e l'angolo di apertura.

    Parametri
    ----------
    Eiso : float
        Energia isotropica equivalente [erg].
    theta_c : float
        Semi-apertura del getto [rad].

    Ritorna
    -------
    Ek : float
        Energia cinetica totale del getto [erg].
    """
    theta=np.deg2rad(theta_c)
    return 0.5 * (1 - np.cos(theta)) * Eiso

def Eiso_from_Ek(Ek, theta_c, profile='gaussian', k=2):
    """
    Calcola E_iso on-axis dato Ek per un getto strutturato.

    Parametri:
    -----------
    Ek : float
        Energia totale nei due getti [erg]
    theta_c : float
        Parametro di apertura [rad]
    profile : str
        'gaussian' o 'powerlaw'
    k : float
        indice del profilo power-law (solo se profile='powerlaw')

    Ritorna:
    --------
    Eiso : float
        Energia isotropica equivalente on-axis [erg]
    """
    # Definiamo la funzione integranda per il calcolo di Ek
    if profile == 'gaussian':
        def integrand(theta, dE0):
            return dE0 * np.exp(-theta**2 / (2*theta_c**2)) * np.sin(theta)
    elif profile == 'powerlaw':
        def integrand(theta, dE0):
            return dE0 * (1 + (theta/theta_c)**2)**(-k) * np.sin(theta)
    else:
        raise ValueError("profile deve essere 'gaussian' o 'powerlaw'")

    # Troviamo la normalizzazione dE/dOmega_0
    def eq_dE0(dE0):
        result, _ = quad(lambda theta: integrand(theta, dE0), 0, np.pi/2)
        return 2*np.pi*result - Ek

    from scipy.optimize import brentq
    dE0 = brentq(eq_dE0, 1e-10, 1e60)  # ricerca radice per normalizzazione

    # Energia isotropica on-axis
    Eiso = 4*np.pi*dE0
    return Eiso

Eiso = 8e51  # erg
Ek=Ek_from_Eiso(Eiso,90 )


theta_core=5.0
theta_c =np.deg2rad(theta_core)  # rad
Eiso_gauss = Eiso_from_Ek(Ek, theta_c, profile='gaussian')
Eiso_pl = Eiso_from_Ek(Ek, theta_c, profile='powerlaw', k=2)

############################################### PARAMETERS ####################################

Eiso = 8e51  # erg
density = 0.01 #0.5  # cm-3
redshift = 0.1
tstart = 300  # s
tstop = 320  # s

#-----list of parameters of a electron distribution (log10) ------

eta_e=-1.44
Ebreak=-1.62
Index2=3.3
Ec=1.32  # cutoff energy in TeV
B=0.25

#################################################################################################

#multi_Eiso=np.logspace(49,52,10)
multi_Eiso = [8e51]

# Plot
plt.figure(figsize=(12,8))
plt.tick_params(axis='both', which='major', labelsize=15)
plt.tick_params(axis='both', which='minor', labelsize=12) 

plt.rc('font', family='sans')
plt.rc('mathtext', fontset='custom')

#plt.loglog(spectrum_energy,sed_SYN,lw=2,label='Sync',c=cmap2(0.9))

for Eiso in multi_Eiso:   
    
    #---------------------------------- ENERGY GRID ----------------------------------------------
    emin=1e-3
    emax=1e16
    bins = int(np.log10(emax/emin) * 20.)  # use 20 bins per decade
    ener = Table([np.logspace(np.log10(emin), np.log10(emax), bins) * u.eV], names=['energy'])  # energy in eV
    #----------------------------------------------------------------------------------------------
    
    grb = GRBModel1(eiso=Eiso, dens=density, tstart=tstart, tstop=tstop, redshift=redshift,
                        pars=[eta_e, Ebreak,Index2, Ec,  B],
                        labels=['log10(eta_e)', 'log10(Ebreak)', 'Index2', 'log10(Ec)', 'log10(B)'],scenario='ISM',cooling_constrain=False,data=ener)
    
    
    grb_struc= GRBModel_topstruc(eiso_zero=Eiso_pl, dens=density, tstart=tstart, tstop=tstop, redshift=redshift,
                        pars=[eta_e, Ebreak,Index2, Ec,  B],
                        labels=['log10(eta_e)', 'log10(Ebreak)', 'Index2', 'log10(Ec)', 'log10(B)'],
                        energy_profile='gaussian',shells=4, theta_obs=0.0*u.deg,
                        scenario='ISM',cooling_constrain=False,data=ener)               
    

    model= grb._SSCmodel_ind1fixed(pars=[eta_e, Ebreak,Index2, Ec, B],data=ener)
    model_str= grb_struc._SSCmodel_ind1fixed(pars=[eta_e, Ebreak,Index2, Ec, B],data=ener)

    SSC=model[0]
    #SSC_no_abs=model[1]
    SSC2=model_str[1]
    
    SSC2_approx= np.sum(grb_struc.synch_comp_approx, axis=0)+ np.sum(grb_struc.ic_comp_approx, axis=0)
    
    #SSC2=np.sum(grb_struc.synch_comp, axis=0)+ np.sum(grb_struc.ic_comp, axis=0)
    
    SSC2_val = np.clip(SSC2_approx.value, 1e-30, 1e50)  # limiti ragionevoli
    ymax=np.max(SSC2_val)
    ymin=np.min(SSC2_val)
    
    ordine = int(np.ceil(np.log10(ymax)))
    ymax = 10**(ordine+1)
    ymin = 10**(ordine - 5)
    
    #grb.plot_sed(1e-3,1e16,6)
    spectrum_energy=np.logspace(np.log10(emin), np.log10(emax), bins) * u.eV



    #plt.loglog(spectrum_energy,SSC,lw=2,label=f'SSC isotropic - Eiso= {"{:.1e}".format(Eiso)}',c="darkred")
    #plt.loglog(spectrum_energy,SSC_no_abs,lw=2,label=f'SSC isotropic - Eiso= {"{:.1e}".format(Eiso)}',c="blue")

    plt.loglog(spectrum_energy,SSC2,lw=2,label=f'SSC struc - Eiso= {"{:.1e}".format(Eiso)}',c="darkgreen")
    
    plt.loglog(spectrum_energy,SSC2_approx,lw=2,label=f'SSC struc approx - Eiso= {"{:.1e}".format(Eiso)}',c="darkblue")
    #plt.loglog(spectrum_energy,SSC_approx,lw=2,label=f'SSC isotropic approx - Eiso= {"{:.1e}".format(Eiso)}',c="orange")
    
    """n_shells = grb_struc.shells

    for i in range(n_shells):  # o range(0,4)
        shell_flux = (grb_struc.synch_comp_approx[i,:] + grb_struc.ic_comp_approx[i,:])
        plt.loglog(spectrum_energy, shell_flux, lw=2, ls='--', color=cmap2(i / n_shells), label=f'shell{i} total')
        #plt.loglog(spectrum_energy, grb_struc.synch_comp_approx[i,:], lw=2, ls='--', color=cmap2(i / n_shells), label=f'shell{i}')
        #plt.loglog(spectrum_energy, grb_struc.ic_comp_approx[i,:], lw=1.5, ls=':', color=cmap2(i / n_shells),label=f'shell{i}')"""
        


plt.xlabel('Photon energy [{0}]'.format(spectrum_energy.unit.to_string('latex_inline')),fontsize=15)
plt.ylabel('$E^2 dN/dE$ [{0}]'.format(model[0].unit.to_string('latex_inline')),fontsize=15)

plt.ylim(ymin, ymax)
#plt.ylim(1e-14, 1e-10)
plt.xlim(1e-3,5e16)
plt.tight_layout()
plt.legend(loc='lower left')


plt.title(f"SSC struc vs struc approx",fontsize=15)
print("")
plt.grid(True, which="both", linestyle="--", alpha=0.6)

#plt.savefig("/media/tobia-matcovich/PortableSSD/JOB/Projects/GRB-modelling/Plots-September15/struc vs struc approx.jpg", format="jpg", dpi=300)
plt.show()