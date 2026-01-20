
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

#from GRBmodel import GRBModel1
from GRBmodelstr import GRBModel_topstruc
#from GRBmodelstr_parallel import GRBModel_topstruc_parallel
from Models import EblAbsorptionModel
from Energy import E_theta_flat, E_theta_gaussian, E_theta_powerlaw,Ek_from_profile,Eiso_from_Ek_with_profile,Ek_from_profile

from astropy.units import def_physical_type

import matplotlib
#print("Backend attuale:", matplotlib.get_backend())
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

m_e = con.m_e.cgs.value
c = con.c.cgs.value
mec2_eV = (con.m_e * con.c ** 2.).to('eV').value
h = con.h.cgs.value
el = con.e.gauss.value
erg_to_eV = 624150912588.3258  # conversion from erg to eV
sigma_T = con.sigma_T.cgs.value
mpc2 = (con.m_p * con.c ** 2.).to('eV')
mpc2_erg = mpc2.to('erg').value

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


grb_gauss= GRBModel_topstruc(eiso_zero=Eiso_gauss, dens=density, tstart=tstart, tstop=tstop, redshift=redshift,
                    pars=[eta_e, Ebreak,Index2, Ec,  B],
                    labels=['log10(eta_e)', 'log10(Ebreak)', 'Index2', 'log10(Ec)', 'log10(B)'],
                    energy_profile='gaussian', theta_obs=0.0*u.deg,thetacore=5.0*u.deg, theta_end=12.0*u.deg,
                    scenario='ISM',cooling_constrain=False,data=ener)       

#grb_pow= GRBModel_topstruc(eiso_zero=Eiso_powerlaw, dens=density, tstart=tstart, tstop=tstop, redshift=redshift,
#                    pars=[eta_e, Ebreak,Index2, Ec,  B],
#                    labels=['log10(eta_e)', 'log10(Ebreak)', 'Index2', 'log10(Ec)', 'log10(B)'],
#                    energy_profile='powerlaw',theta_obs=0.0*u.deg,thetacore=4.0*u.deg, theta_end=15.0*u.deg,
#                    scenario='ISM',cooling_constrain=False,data=ener)   


model_gauss= grb_gauss._SSCmodel_ind1fixed(pars=[eta_e, Ebreak,Index2, Ec, B],data=ener)

log.info(f"Computation time for structured model: {time.time() - start:.3f} s")
grb_gauss.print_GRB_status()

#model_pow= grb_pow._SSCmodel_ind1fixed(pars=[eta_e, Ebreak,Index2, Ec, B],data=ener)

grb_gauss.plot_doppler_segmentation(theta_max_deg=20*u.deg,phi_fixed=0.0*u.deg,plot_Dphi=True,plot_Dmedio=True,
                                   Save=plot_Save,Path=save_dir,Name="Doppler_segmentation")

grb_gauss.plot_jet_profile(Save=plot_Save,Path=save_dir,Name="Jet Profiles")

grb_gauss.plot_sed(emin=emin, emax=emax,plot_approx_spectrum=False,plot_approx_shells=False,plot_true_shells=True,plot_true_spectrum=True,
                 Save=plot_Save,Path=save_dir,Name="SED")

grb_gauss.plot_sed(emin=emin, emax=emax,plot_approx_spectrum=True,plot_approx_shells=True,plot_true_shells=False,plot_true_spectrum=False,
                 Save=plot_Save,Path=save_dir,Name="APPROX_SED")


grb_gauss.plot_sed3D(emin=emin, emax=emax,order_bottom=6, elev=30,azim=-40,
                   plot_true_spectrum=True,
                   plot_shells=True,
                   Save=plot_Save,Path=save_dir,Name="SED_3D")



grb_gauss.plot_Gamma_3D(time=1e5,crop=True)

grb_gauss.plot_gamma_R_3D_range(t_min=10, t_max=1e5, theta_min=0, theta_max=40,n_t=100, n_theta=100,slice=True,elevation=35,azimut=60,
                                Save=plot_Save,Name="Gamma_R_struc_",Path=save_dir)

grb_gauss.plot_doppler_3D(plane=True,circle=True,surface=True,pixels=300,crop=True,
                         Save=plot_Save,Path=save_dir)
      
#grb_gauss.animate_doppler(tmin=100,tmax=10000,crop=True,pixels=300,Save=plot_Save,Path=save_dir)
#grb_gauss.animate_Gamma(tmax=1000000,crop=True,pixels=300,Save=plot_Save,Path=save_dir)'''


log.info(f"Execution time for TEST_struc_model.py: {time.time() - start:.3f} s")
######################################################
stop_event.set()
t.join()
######################################################