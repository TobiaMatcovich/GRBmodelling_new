from log_config import setup_logger
log = setup_logger(__name__)  

from multiprocessing import Pool
from astropy.table import Table, hstack
import astropy.units as u
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
from Utils import trapz_loglog

from GRBmodel import GRBModel1
from GRBmodelstr import GRBModel_topstruc
from GRBmodelstr_parallel import GRBModel_topstruc_parallel
from Energy import E_theta_flat, E_theta_gaussian, E_theta_powerlaw,Ek_from_profile,Eiso_from_Ek_with_profile,Ek_from_profile

from astropy.units import def_physical_type

from matplotlib import cm
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.colorbar import ColorbarBase

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

import os
from datetime import datetime

############################ Spinner on terminal #########################
import itertools, sys, threading, time
def spinner(stop_event):
    for c in itertools.cycle('|/-\\'):
        if stop_event.is_set():
            break
        sys.stdout.write('\rProcessing ' + c)
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write('\rDone!       \n')

stop_event = threading.Event()
t = threading.Thread(target=spinner, args=(stop_event,))
t.start()

########################### SAVE ############################################
plot_Save = True 

if plot_Save:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H")
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    save_dir = f"plots_{script_name}_{timestamp}"

    os.makedirs(save_dir, exist_ok=True)

    print(f"The new plots will be stored in this folder:: {save_dir}")
else:
    save_dir = None

##############################################################################
start=time.time()

Ek = 8e51  # erg

Eiso_flat = Eiso_from_Ek_with_profile(Ek, E_theta_flat,profile_kwargs={'thetaw_deg':30.0})
log.info(f"E_iso_flat(0) = {Eiso_flat:.3e} erg")

Eiso_gauss= Eiso_from_Ek_with_profile(Ek, E_theta_gaussian, profile_kwargs={'thetaw_deg':12.0, 'theta_core':5.0})
log.info(f"E_iso_gauss(0) = {Eiso_gauss:.3e} erg")

Eiso_powerlaw = Eiso_from_Ek_with_profile(Ek, E_theta_powerlaw, profile_kwargs={'thetaw_deg':15.0,'theta_core':4.0})
log.info(f"E_iso_pl(0) = {Eiso_powerlaw:.3e} erg")

############################################### PARAMETERS ####################################

#Eiso = 8e51  # erg
density = 0.01 #0.5  # cm-3
redshift = 0.01
tstart = 200  # s
tstop = 240  # s

#-----list of parameters of a electron distribution (log10) ------

eta_e=-1.44
Ebreak=-1.62
Index2=3.3
Ec=1.32  # cutoff energy in TeV
B=0.25

############################################ ENERGY GRID ########################################

emin=1e-3
emax=1e16
bins = int(np.log10(emax/emin) * 20.)  # use 20 bins per decade
ener = Table([np.logspace(np.log10(emin), np.log10(emax), bins) * u.eV], names=['energy'])  # energy in eV

###################################################################################################

grb = GRBModel1(eiso=Eiso_powerlaw, dens=density, tstart=tstart, tstop=tstop, redshift=redshift,
                    pars=[eta_e, Ebreak,Index2, Ec,  B],
                    labels=['log10(eta_e)', 'log10(Ebreak)', 'Index2', 'log10(Ec)', 'log10(B)'],scenario='ISM',cooling_constrain=False,data=ener)

grb.print_GRB_status()
grb._SSCmodel_ind1fixed(pars=[eta_e, Ebreak,Index2, Ec, B],data=ener)

emin=1e-3
emax=1e16
grb.plot_sed(emin,emax,9,plot_true=True,plot_gg_abs=True,Save=plot_Save,Path=save_dir,Name="SED")
grb.plot_gamma_radius_vs_time(tmin=1, tmax=1e7, num=200,Save=True,Path=save_dir,Name="Gamma_vs_R_and_t")

log.info(f"Execution time for TEST_model.py: {time.time() - start:.3f} s")
######################################################
stop_event.set()
t.join()
######################################################