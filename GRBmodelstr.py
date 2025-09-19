
import logging
import os
import warnings
from collections import OrderedDict
from multiprocessing import Pool

import astropy
from astropy.table import Table, hstack
import astropy.units as u
from astropy.io import ascii
import astropy.constants as con
from astropy.constants import alpha, c, e, hbar, m_e, m_p, sigma_sb
from astropy.utils.data import get_pkg_data_filename
from astropy.cosmology import WMAP9 as cosmo

import numpy as np
import matplotlib.pyplot as plt

from Validator import (
    validate_array,
    validate_physical_type,
    validate_scalar,
)
#from .model_utils import memoize
from Utils import trapz_loglog

import Models
import Radiative

import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.colorbar import ColorbarBase


from scipy.integrate import quad


#-------------------------------------------------- static variables: ---------------------------------------------

m_e = con.m_e.cgs.value
c = con.c.cgs.value
mec2_eV = (con.m_e * con.c ** 2.).to('eV').value
h = con.h.cgs.value
el = con.e.gauss.value
erg_to_eV = 624150912588.3258  # conversion from erg to eV
sigma_T = con.sigma_T.cgs.value
mpc2 = (con.m_p * con.c ** 2.).to('eV')
mpc2_erg = mpc2.to('erg').value

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap
  
def Lorentz_factor(Etheta,n0,t):
    """
    Ryan et al 2020 (Afterglowpy)
    """
    
    L_factor=np.sqrt(Etheta/n0)*t**(-3/2)
    return L_factor

def R_forward_shock(t,Lfactor):
    
    Radius=c*t*(1-1/(16*Lfactor**2))
    return Radius

def tobs(t,Lfactor,phi_deg):
    """ Photons emitted from a sector of teh blast wave at time t wil be seen by the observer at tobs.
        Here we ignore the effect of cosmological redshuft as it does non affect the closure relations (is it ok for us?)

    """
    phi= np.deg2rad(phi_deg)
    mu=np.cos(phi)
    
    t_obs=(1-mu)*t+(mu*t)/(16*Lfactor**2)
    return t_obs

#------------------------------------------------------------------------------------------------------------------

# Emissività nel sistema della shell (puoi sostituirla con la tua funzione)
def j_nu_prime(nu_prime):
    # esempio di legge di potenza: j' ~ nu'^(-1)
    return nu_prime**(-1)

# Fattore Doppler
def doppler_factor(Gamma, theta):
    thetas_rad = np.radians(theta)
    beta = np.sqrt(1.0 - 1.0 / Gamma**2)
    return 1.0 / (Gamma * (1.0 - beta * np.cos(thetas_rad )))

# Integrando da inserire nell'integrale
def integrand(theta, nu_obs, z, Gamma, R, delta_R, theta_j):
    delta = doppler_factor(Gamma, theta)
    nu_prime = (1 + z) * nu_obs / delta
    j_nu = j_nu_prime(nu_prime)
    return np.sin(theta) * delta**2 * j_nu

# Flusso osservato
def flux_observed(nu_obs, z, d_L, Gamma, R, delta_R, theta_j):
    prefactor = (1 + z) * R**2 * delta_R / (2 * d_L**2)
    integral, _ = quad(integrand, 0, theta_j,
                       args=(nu_obs, z, Gamma, R, delta_R, theta_j))
    return prefactor * integral

#----------------------------------------------------------------------------------------------


# Flusso osservato sommando N sottojet
def flux_observed_multi_jet(nu_obs, z, d_L, Gamma, R, delta_R, theta_j, N):
    theta_edges = np.linspace(0, theta_j, N + 1)
    prefactor = (1 + z) * R**2 * delta_R / (2 * d_L**2)
    total_flux = 0.0

    for n in range(N):
        theta_min = theta_edges[n]
        theta_max = theta_edges[n + 1]

        integral, _ = quad(integrand, theta_min, theta_max,
                           args=(nu_obs, z, Gamma, R, delta_R))
        total_flux += integral

    return prefactor * total_flux


#-----------------------------------------------------------------------------------------------------------

#------------------------------------------------- static methods --------------------------------------------------

def sigma_gammagamma(Eph1, Eph2):
    """
    gamma-gamma cross section averaged over scattering angle
    The value is returned in cm2

    Equation 5) from Eungwanichayapant, A.; Aharonian, F., 2009
    (originally from Aharonian, 2004) Approximation good within 3%

    Parameters
    ----------
       Eph1 : array_like
         numpy array of energy of gamma ray in eV
       Eph2 : array_like
         np.array of energy of target photon in eV
    Returns
    -------
        cross_section : astropy.quantity
          angle average cross section for gamma gamma absorption
    """

    CMene = Eph1 * Eph2 / (mec2_eV * mec2_eV)
    mask = CMene > 1.  # mask condition to account for the threshold effect.
    res = np.full(CMene.shape, 0.)
    
    res[mask] = 3. / (2. * CMene[mask] * CMene[mask]) * sigma_T * \
                ((CMene[mask] + 0.5 * np.log(CMene[mask]) - 1. / 6. + 1. / (2. * CMene[mask]))
                * np.log(np.sqrt(CMene[mask]) + np.sqrt(CMene[mask] - 1.)) -
                (CMene[mask] + 4. / 9. - 1. / (9. * CMene[mask])) * np.sqrt(1. - (1. / CMene[mask])))
    cross_section = res * u.cm * u.cm
    return cross_section


def absorption_coeff(egamma, targetene, target):
    """
    Returns the absorption coefficient K that will then be
    spatially integrated.

    K(E) = \int_e sigma_gg(E,e) * dn/de * de

    where E is the gamma ray energy, e is the energy of the target photon and dn/de is the number distribution
    of the target photon field. (Inner integral of equation 3.24 of Aharonian, 2004)

    Parameters
    ----------
      egamma : array_like
        Energy of the gamma ray photon as astropy unit quantity. Format e.g. [1.]*u.TeV
      targetene : array_like
        energy array of the photon distribution e.g. [1.,2.]*u.eV
      target : array_like
        dnde value of the target photon distribution as 1/(eV cm3)
    Returns
    -------
      abs_coeff : astropy.quantity
        absorption coefficient as astropy quantity (should have units 1/u.cm)
    """

    product = sigma_gammagamma(np.vstack(egamma.to('eV')),targetene.to('eV').value) * target  # make sure the units are correct
    abs_coeff = trapz_loglog(product, targetene, axis=1)
    return abs_coeff


def tau_val(Egamma, targetene, target, size):
    """
    Optical depth assuming homogeneous radiation field.

    From equation 3.24 of Aharonian, 2004 with the assumption of homogeneous photon field.

    Parameters
    ----------
      Egamma    : array_like
        Energy of the gamma ray photon as astropy unit quantity. Format e.g. [1.]*u.TeV
      targetene : array_like
        energy array of the photon distribution e.g. [1.,2.]*u.eV
      target    : array_like
        dnde value of the target photon distribution as 1/(eV cm3)
      size      : astropy.quantity
        size of the integration length as astropy spatial quantity (normally units of cm)
    Returns
    -------
      tau : array_like
        optical depth
    """

    coeff = absorption_coeff(Egamma, targetene, target)
    tau = size.to('cm') * coeff
    return tau

def cutoff_limit(bfield):
    """
     Account for the maximum energy of particles
     for synchrotron emission. Due to the effect of the synchrotron burn-off
     due to the balancing of acceleration and losses of the particle in magnetic field.
     Expression 18 from Aharonian, 2000

    Parameters
    ----------
      bfield : float
        Magnetic field intensity to be given in units of Gauss
    Returns
    -------
      cutoff_ev : float
        log10 of the cutoff energy in units of TeV
    """

    eff = 1.  # acceleration efficiency parameter (eff >= 1)
    cutoff = ((3. / 2.) ** (3. / 4.) *np.sqrt(1. / (el ** 3. * bfield)) * (m_e ** 2. * c ** 4.)) * eff ** (-0.5) * u.erg
    cutoff_TeV = (cutoff.value * erg_to_eV * 1e-12)
    return np.log10(cutoff_TeV)

def synch_cooltime_partene(bfield, partene):
    """
    Computes the cooling time for an electron with energy 'partene' in
    Bfield. Returns in units of seconds
    Equation 1 from Aharonian, 2000

    Parameters
    ----------
       bfield : astropy.quantity
         magnetic field as astropy quantity (u.G)
       partene : astropy.quantity
         particle energy as astropy quantity (u.eV)
    Returns
    -------
       tcool : astropy.quantity
         Synchrotron cooling time as astropy quantity (u.s)
    """

    bf = bfield.to('G').value
    epar = partene.to('erg').value
    tcool = (6. * np.pi * m_e ** 4. * c ** 3.) / (sigma_T * m_e ** 2. * epar * bf ** 2.)
    return tcool * u.s

def synch_charene(bfield, partene):
    """
    Function to return
    characteristic energy of synchrotron spectrum

    Equation 3.30 from Aharonian, 2004 (adapted for electrons)

    Parameters
    ----------
       bfield : astropy.quantity
         magnetic field as astropy quantity (u.G)
       partene : astropy.quantity
         particle energy as astropy quantity (u.eV)
    Returns
    -------
       charene : astropy.quantity
         synchrotron characteristic energy as astropy quantity (u.eV)
    """

    bf = bfield.to('G').value
    epar = partene.to('erg').value
    charene = np.sqrt(3. / 2.) * (h * el * bf) / (2. * np.pi * (m_e ** 3. * c ** 5.)) * epar ** 2.  # in ergs
    return charene * erg_to_eV * u.eV


class GRBModel_topstruc:
    """
      
    Attributes
    ----------
    Eiso : float
        Isotropic energy of the GRB (in units of erg)
    density : float
        density of the circumburst material (in units of cm-3)
    dataset : list of astropy.table.table.Table
        table of observational data. Attribute exists only if a list of tables is passed in the initialization
    tstart : float
        starting time of the observational interval (in units of seconds)
    tstop : float
        stop time of the observational interval (in units of seconds)
    avtime : float
        average time of the observational interval
    redshift : float
        redshift of the GRB
    Dl : astropy.quantity
        luminosity distance of the GRB (as astropy quantity)
    pars : list
        list of parameters of the electron distribution (es. ExponentialCutoffBrokenPowerLaw)
    labels : list
        list of parameter names (as strings)
    scenario : string
        dictates the density of the circumburst material (DEFAULT = 'ISM')
    mass_loss : float
        mass loss rate of the progenitor (in solar masses per year; for `Wind` scenario only)
    wind_speed : float
        wind speed of the progenitor star (in km per second; for `Wind` scenario only)
    cooling_constrain : boolean
        If True adds to the prior a constrain for which cooling time at break ~ age of the system. DEFAULT = True
        If synch_nolimit = True, this option does not do anything.
    synch_nolimit : boolean
        False for standard SSC model, True for synchtrotron dominated model. DEFAULT = False
    gamma : float
        Lorentz factor of the GRB at time avtime
    sizer : float
        radius of the expanding shell at time avtime
    shock_energy : astropy.quantity (u.erg)
        available energy in the shock
    Emin : astropy.quantity
        minimum injection energy of the electron distribution
    Wesyn : astropy.quantity
        total energy in the electron distribution
    eta_e : float
        fraction of available energy ending in the electron distribution
    eta_b : float
        fraction of available energy ending in magnetic field energy density
    synch_comp : numpy.array
        synchrotron component of the emission
    ic_comp : numpy.array
        inverse compton component of the emission
    synch_compGG : numpy.array
        synchrotron component of the emission
        with gamma gamma absorption included METHOD 1
    ic_compGG : numpy.array
        inverse compton component of the emission
        with gammagamma absorption included METHOD 1
    synch_compGG2 : numpy.array
        synchrotron component of the emission
        with gamma gamma absorption included METHOD 2
    ic_compGG2 : numpy.array
        inverse compton component of the emission
        with gammagamma absorption included METHOD 2
    naimamodel : bound method
        bound method to the model function
        associated with function load_model_and_prior()
    lnprior : bound method
        bound method to the prior function
        associated with function load_model_and_prior()
    """

    def __init__(self, eiso_zero, dens, tstart, tstop, redshift, pars, labels,
                 scenario='ISM',
                 energy_profile='gaussian',
                 shells=10, # number of concentric shells
                 #thetaend=10.0*u.deg,
                 thetacore=5.0*u.deg, # core angle of the jet
                 mass_loss=0,
                 wind_speed=0,
                 cooling_constrain=True,
                 synch_nolimit=False,
                 data=None):
        """
        Class initialization

        Parameters
        ----------
          eiso : float
            Isotropic energy of the gamma ray burst (given in erg)
          dens : float
            density of the circumburst material (given in cm-3)
          data : list
            list of astropy table with the obs. data. Optional, theoretical line can be computed anyway
          tstart : float
            start time of the observational interval (given in seconds after trigger)
          tstop : float
            stop time of the observational interval (given in seconds after trigger)
          redshift : float
            redshift of the GRB
          pars : list
            list of parameters passed to the model function
          labels : list
            names of the parameters passed to the model
          scenario : string
            'ISM', 'Wind' or 'average'
          mass_loss : float
            mass loss rate of the progenitor star (in solar masses per year for Wind scenario, no effect otherwise)
          wind_speed : float
            wind speed of the progenitor star (in km/s for Wind scenario, no effect otherwise)
          cooling_constrain : bool
            boolean to add a contrain on cooling time at break ~ age of of the system in the prior function
          synch_nolimit : bool
            boolean to select the synchrotron dominated model
        """

        if isinstance(data, list):
            if all(isinstance(x, astropy.table.table.Table) for x in data):
                self.dataset = data  # dataset astropy table
            else:
                print("WARNING: Not all the elements in your list are formatted as astropy tables!")
                print("Not loading the dataset,")
                print("the code can be used only for computation of theoretical curves")
        else:
            print("WARNING: No dataset given,")
            print("the code can be used only for computation of theoretical curves")
            
        self.Eiso_zero = eiso_zero  # Eiso of the burst
        self.thetacore = thetacore #theta core of the jet in units of grad
        self.thetaend= 0 # truncation angle outside of which the energy is initially 0
        self.energy_profile=energy_profile
        self.shells= shells
        self.theta_limits=0
        self.theta_shells=0
        self.Eavg_array = 0
        self.density = dens  # ambient density around the burst units of cm-3
        self.tstart = tstart  # units of s
        self.tstop = tstop  # units of s
        self.avtime = (tstart + tstop) / 2.  # units of s
        self.redshift = redshift
        self.Dl = cosmo.luminosity_distance(redshift)  # luminosity distance with units
        self.B=0
        self.pars = pars  # parameters for the fit
        self.labels = labels  # labels of the parameters
        self.scenario = scenario  # string valid options: 'average', 'Wind', 'ISM'
        self.mass_loss = mass_loss  # Value of the mass loss rate of progenitor in solar masses per year
        self.wind_speed = wind_speed  # Value of the wind speed of the projenitor in km/s
        self.cooling_constrain = cooling_constrain  # if True add in the prior a constrain on cooling break
        self.synch_nolimit = synch_nolimit  # boolean for SSC (=0) or synchrotron without cut-off limit model (=1)
        self.gamma = 0  # Gamma factor of the GRB at certain time
        self.sizer = 0  # External radius of the shell
        self.depthpar = 0  # private attribute to control the depth of the shock: d = R/(self.depthpar * Gamma)
        self.shell_dept=0
        self.volume=0
        self.shock_energy = 0  # Available energy in the shock
        self.Emin = 0 * u.eV  # Minimum injection energy for the particle distribution
        self.Wesyn = 0  # Total energy in the electrons
        self.eta_e = 0  # Fraction of thermal energy going into electron energy
        self.eta_b = 0  # Fraction of thermal energy going into magnetic field
        self.naimamodel = 0  # Model used for the fit - initialized in later function
        self.lnprior = 0  # Prior used for the fit - initialized in later function
        self.load_model_and_prior()  # Loads the model and the relative prior
        self.esycool = 0  # Characteristic synchrotron energy corresponding to the break energy of the electrons
        self.synchedens = 0  # total energy density of synchrotron photons

        self.synch_comp_approx= 0  # Spectrum of the synchrotron component
        self.ic_comp_approx = 0  # Spectrum of the IC component
        self.synch_comp = 0  # Spectrum of the synchrotron component
        self.ic_comp = 0  # Spectrum of the IC component
        self.synch_compGG = 0  # synchrotron component of the model with gammagamma absorption with METHOD 1
        self.ic_compGG = 0  # inverse compton component of the model with gammagamma absorption with METHOD 1
        self.synch_compGG2 = 0  # synchrotron component of the model with gammagamma absorption with METHOD 2
        self.ic_compGG2 = 0  # inverse compton component of the model with gammagamma absorption with METHOD 2
        
    
    def E_theta_gaussian(self,theta, thetaw_deg=12.0):
        self.thetaend= thetaw_deg
        thetacore=self.thetacore.value
        E = np.zeros_like(theta)
        inside = theta <= thetaw_deg
        Eiso_zero=self.Eiso_zero
        E[inside]= Eiso_zero * np.exp(-(theta[inside]**2) / (2 * thetacore**2))
        E = np.maximum(E, 1e-40) #floor to avoid numerical issues
        return E 
      
    def E_theta_powerlaw(self,theta,thetaw_deg=20.0,b=4.5):
        self.thetaend= thetaw_deg
        thetacore=self.thetacore.value
        Eiso_zero=self.Eiso_zero
        
        E = np.zeros_like(theta)
        inside = theta <= thetaw_deg
        E[inside] =Eiso_zero * (1 + (theta[inside]**2) / (b * thetacore**2))**(-b/2)
        E = np.maximum(E, 1e-40) #floor to avoid numerical issues
        return E 

      
    """def gammaval(self,avtime,theta):
        
        Computes the Lorentz factor and the size of the region
        Expression from Blandford&McKee,1976.

        Gamma^2 = E_iso / Mc^2

        where M is the mass of the material swept by the shock which can be computed in case of homogenous
        density or wind scenario, with the density that decreases as r^-2 (see documentation file for more details).
        The calculation of the radius uses the relation

        R = A * Gamma^2 * (ct)

        where A can be 4 (for Wind scenario), 8 (ISM scenario), 6 (for the average)

        Time is the average between the tstart and tstop.
        The functions takes automatically the initialization parameters
        
        
        #theta_shells = self.theta_shells
        theta = np.atleast_1d(theta)

        
        if (self.scenario == 'ISM'):
            self.depthpar = 9. / 1.
            if self.energy_profile=='gaussian':
                Energy = self.E_theta_gaussian(theta, thetaw_deg=12)
            elif self.energy_profile == 'powerlaw':
                Energy = self.E_theta_powerlaw(theta,thetaw_deg=20,b=4.5)
            else:
                raise ValueError(f"Unknown energy profile: {self.energy_profile}")       
            
            #Gamma = (1. / 8.) ** (3. / 8.) * (3.0 * Energy / (4.0 * np.pi * self.density * mpc2_erg * (c * avtime) ** 3.0)) ** 0.125 OLD
            Gamma =  (3.0 * Energy / (8.0 * np.pi *512.0* self.density * mpc2_erg * (c * avtime) ** 3.0)) ** 0.125
            R = 8. * c * avtime * Gamma ** 2
            
            self.gamma = Gamma
            self.sizer = R
            self.Eavg_array = Energy
        else:
            text = "Chosen scenario: %s\n" \
                   "The scenario indicated not found. Please choose 'ISM' scenario" % self.scenario
            raise ValueError(text)    
        
        if Gamma.size == 1:
          return Gamma[0], R[0]
        else:
          return Gamma, R"""
          
    def calc_photon_density(self, Lsy, sizereg,theta_inf,theta_sup):
        """

        Parameters
        ----------
            Lsy : array_like
              emitted photons per second (units of 1/s)
            sizereg : astropy.quantiy
              size of the region as astropy u.cm quantity
              
        Returns
        -------
          ph_dens : array_like
            Photon density in the considered emission region.
        """
        
        Omega_i = 2 * np.pi * (np.cos(np.radians(theta_inf)) - np.cos(np.radians(theta_sup)))

        return Lsy / (Omega_i * sizereg ** 2. * c * u.cm / u.s)
      
    """def load_model_and_prior(self):
        
        Associates the bound methods
        naimamodel and lnprior to the chosen
        model and prior function.

        Modify here if you want to change the model
        or the priors
        
        theta_limits = np.linspace(0, self.thetaend, self.shells+1)
        self.theta_limits=theta_limits 
        theta_inf = theta_limits[:-1]  # N elementi
        theta_sup = theta_limits[1:]
        theta_shells = 0.5 * (theta_inf + theta_sup)
        self.theta_shells=theta_shells 
        
        self.shock_energy=np.zeros(self.shells)*u.erg
        self.shell_dept=np.zeros(self.shells)
        self.volume=np.zeros(self.shells)
      
      
        self.gammaval(self.avtime, self.theta_shells)  # call the function to compute the basic GRB initialization parameters
        self.naimamodel = self._SSCmodel_ind1fixed
        #-------------------------- change here for the prior functions -------------------------------------
        # For performance it is better to use if statements here to avoid having them in the prior function
        # the prior function is called everytime and it's better if it does not have if statements inside
        
        if self.synch_nolimit:
            self.lnprior = self._lnprior_ind2free_nolim
        else:
            if self.cooling_constrain:
                self.lnprior = self._lnprior_ind2free_wlim_wcooling_constrain
            else:
                self.lnprior = self._lnprior_ind2free_wlim
        """

 

    def compute_dynamics(self, avtime, theta):
        """Calcola Lorentz factor, raggio e energia"""
        theta = np.atleast_1d(theta)
        
        if (self.scenario == 'ISM'):
                    self.depthpar = 9. / 1.
                    if self.energy_profile=='gaussian':
                        Energy = self.E_theta_gaussian(theta, thetaw_deg=12)
                    elif self.energy_profile == 'powerlaw':
                        Energy = self.E_theta_powerlaw(theta,thetaw_deg=20,b=4.5)
                    else:
                        raise ValueError(f"Unknown energy profile: {self.energy_profile}")  
            

        Gamma = (3.0 * Energy / (8.0 * np.pi * 512.0 * self.density * mpc2_erg * (c * avtime) ** 3.0)) ** 0.125
        R = 8. * c * avtime * Gamma ** 2

        return Gamma, R, Energy
    
    def _setup_shells(self):

        if self.energy_profile=='gaussian':
            self.thetaend= 12.0
        elif self.energy_profile == 'powerlaw':
            self.thetaend= 20.0

        theta_limits = np.linspace(0, self.thetaend, self.shells + 1)
        print("Theta limits (deg):", theta_limits)
        self.theta_limits = theta_limits
        self.theta_shells = 0.5 * (theta_limits[:-1] + theta_limits[1:])
        
        self.shock_energy = np.zeros(self.shells) * u.erg
        self.shell_dept = np.zeros(self.shells)
        self.volume = np.zeros(self.shells)

    def load_model_and_prior(self):
        """ Geometry setup model selection"""
        print("")
        print(" ------------------ Starting structured GRB initialization ------------------")
        
        self._setup_shells()
        Gamma, R, Energy = self.compute_dynamics(self.avtime, self.theta_shells)
        print("Gamma:", Gamma)
        self.gamma = Gamma
        self.sizer = R
        self.Eavg_array = Energy

        self.naimamodel = self._SSCmodel_ind1fixed

        #-------------------------- change here for the prior functions -------------------------------------
        # For performance it is better to use if statements here to avoid having them in the prior function
        # the prior function is called everytime and it's better if it does not have if statements inside
        """
        if self.synch_nolimit:
            self.lnprior = self._lnprior_ind2free_nolim
        else:
            if self.cooling_constrain:
                self.lnprior = self._lnprior_ind2free_wlim_wcooling_constrain
            else:
                self.lnprior = self._lnprior_ind2free_wlim
        """
        print(" ------------------ Ending structured GRB initialization ------------------")
        print("")
        
    def _SSCmodel_ind1fixed(self, pars, data):
        """"
        Example set-up of the free parameters for the SSC implementation
        Index1 of the BPL is fixed as Index2 - 1 (cooling break).Index2 of the BPL is free
        The minimum energy and the normalization of the electron distribution are derived from the parameter eta_e

        Parameters
        ----------
           pars : list
             parameters of the model as list
           data : astropy.table.table.Table
             observational dataset (as astropy table) or
             if interested only in theoretical lines, astropy table
             containing only a column of energy values.
        Returns
        -------
           model : array_like
             values of the model in correspondence of the data
           electron_distribution : tuple
             electron distribution as tuple energy, electron_distribution(energy) in units of erg
        """
        print("")
        print("-------------------------- Starting structured GRB model computation --------------------------")
        n_shells = self.shells  # number of shells to consider
        energy_grid = data['energy']  # array di energie, dimensione (n_energy,)
        n_energy = len(energy_grid)

        # Inizializza una matrice vuota (n_shells x n_energy)
        self.synch_comp_approx = np.zeros((n_shells, n_energy))* (u.erg / (u.cm**2 * u.s)) 
        self.ic_comp_approx = np.zeros((n_shells, n_energy))* (u.erg / (u.cm**2 * u.s))
        self.synch_comp = np.zeros((n_shells, n_energy))* (u.erg / (u.cm**2 * u.s))
        self.ic_comp = np.zeros((n_shells, n_energy)) * (u.erg / (u.cm**2 * u.s))
        self.synch_compGG2 = np.zeros((n_shells, n_energy))* (u.erg / (u.cm**2 * u.s))
        self.ic_compGG2= np.zeros((n_shells, n_energy)) * (u.erg / (u.cm**2 * u.s))
        #-------------------------------------------------------------------------------------------------

        eta_e = 10. ** pars[0] # parameter 0: fraction of available energy ending in non-thermal electrons
        self.eta_e = eta_e
        ebreak = 10. ** pars[1] * u.TeV  # parameter 1: linked to break energy of the electron distribution (as log10)
        alpha1 = pars[2] - 1.  # fixed to be a cooling break
        alpha2 = pars[2]  # parameter 2: high energy index of the ExponentialCutoffBrokenPowerLaw
        e_cutoff = (10. ** pars[3]) * u.TeV  # parameter 3: High energy cutoff of the electron distribution (as log10)
        
        bfield = 10. ** (pars[4]) * u.G  # parameter 4: Magnetic field (as log10)
        self.B=bfield
        redf = 1. + self.redshift  # redshift factor
        
        #doppler_approx = self.gamma  # assumption of doppler boosting ~ Gamma
        size_reg = self.sizer * u.cm  # size of the region as astropy quantity
        
        #theta_array = np.linspace(0, self.thetaend, self.shells+1)
        theta_inf =self.theta_limits[:-1]  # N elementi
        theta_sup =self.theta_limits[1:]   # N elementi

        print("Theta inf (deg):", theta_inf)
        print("Theta sup (deg):", theta_sup)
        
        # ------------------- Volume shell where the emission takes place. The factor 9 comes from considering the shock in the ISM ----------------
        #  (Eq. 7 from GRB190829A paper from H.E.S.S. Collaboration)
        #vol = 4. * np.pi * self.sizer ** 2. * (self.sizer / (9. * self.gamma))
        
        target_unit = u.erg / (u.cm**2 * u.s)
        model_wo_abs = np.zeros(len(data['energy'])) * target_unit
        model= np.zeros(len(data['energy'])) * target_unit
        Gamma=self.gamma

        
        for i in range(n_shells):

          print("")
          print("Shell number:", i+1, "over", n_shells)  
          E_shell = self.Eavg_array[i]  
          Gamma_i = Gamma[i]
          print("Gamma:", Gamma_i)
          print("Energy in the shell (erg):", E_shell)

          if (E_shell <= 1e-40) or (Gamma_i <= 1.01):
                print(f"Skipping shell {i}: E={E_shell}, Gamma={Gamma_i}")
                continue
          
          deltaR=self.sizer[i]/(self.depthpar*Gamma_i)
          self.shell_dept[i]=deltaR
          Omega_i = 2 * np.pi * (np.cos(np.radians(theta_inf[i])) - np.cos(np.radians(theta_sup[i])))*2
          volume_element= self.sizer[i]**2.*deltaR*Omega_i
          self.volume[i]=volume_element
          
          shock_energy = 2. * Gamma_i* self.density * mpc2_erg * u.erg  # available energy in the shock
          self.shock_energy[i] = shock_energy

        
          eemax = e_cutoff.value * 1e13 # maximum energy of the electron distribution, based on 10 * cut-off value in eV (1 order more then cutoff)
          self.eta_b = (bfield.value ** 2 / (np.pi * 8.)) / shock_energy.value  # ratio between magnetic field energy and shock energy   
        
          #--------------------------------------------------------------------------------------------------------------------------
          ampl = 1. / u.eV  # temporary amplitude
          ECBPL = Models.ExponentialCutoffBrokenPowerLaw(ampl, 1. * u.TeV, ebreak, alpha1, alpha2, e_cutoff)
          #---------------------------------------------- E min iterative process ----------------------------------------------------
          Emin_0=1e9*u.eV
          Emin_0_exp=9
          energies = np.logspace(Emin_0_exp, np.log10(eemax), 100) * u.eV
          eldis = ECBPL(energies)
          fact1 = eta_e * self.gamma[i] * mpc2
          E_medium = trapz_loglog(energies * eldis, energies) / trapz_loglog(eldis, energies)
          K=E_medium/fact1
        
          emin= Emin_0/K  # calculation of the minimum injection energy. See detailed model explanationn(! iteration)
          self.Emin = emin
          #--------------------------------------------- E min non iterative process -------------------------------------------------
        
          #emin=(p-2)/(p-1)*fact1 # p=? abbiamo una broken powerlaw, quindi sono 2 
        
          #----------- (https://www.cv.nrao.edu/~sransom/web/Ch5.html)------------------------
          SYN = Radiative.Synchrotron(ECBPL, B=bfield, Eemin=emin, Eemax=eemax * u.eV, nEed=20)
      
          #----------------------------------------------------------------------------------------------------------------------------
        
          amplitude = ((eta_e * shock_energy * volume_element) / SYN.compute_Etot(Eemin=emin, Eemax=eemax * u.eV)) / u.eV

        
          ECBPL = Models.ExponentialCutoffBrokenPowerLaw(amplitude, 1. * u.TeV, ebreak, alpha1, alpha2, e_cutoff)
          SYN = Radiative.Synchrotron(ECBPL, B=bfield, Eemin=emin, Eemax=eemax * u.eV, nEed=20)
          self.Wesyn += SYN.compute_Etot(Eemin=emin, Eemax=eemax * u.eV)           # E tot in the electron distribution
        
          #----- energy array to compute the target photon number density to compute IC radiation and gamma-gamma absorption -----------
        
          cutoff_charene = np.log10((synch_charene(bfield, e_cutoff)).value)      # characteristic energy at the electron cutoff
          min_synch_ene = -4                                                      # minimum energy to start sampling the synchrotron spectrum
          bins_per_decade = 20                                                    # 20 bins per decade to sample the synchrotron spectrum
          bins = int((cutoff_charene - min_synch_ene) * bins_per_decade)
        
          #---------------------------------------------------------------------------------------------------------------------
          Esy = np.logspace(min_synch_ene, cutoff_charene + 1, bins) * u.eV
        
          Lsy = SYN.flux(Esy, distance=0 * u.cm) # number of synchrotron photons per energy per time (units of 1/eV/s)
          phn_sy = self.calc_photon_density(Lsy, size_reg[i],theta_inf[i],theta_sup[i])   # number density of synchrotron photons (dn/dE) units of 1/eV/cm3
 
          
        #----------------------------------------------------------------------------------------------------------------------
          self.esycool = (synch_charene(bfield, ebreak))
          self.synchedens = trapz_loglog(Esy * phn_sy, Esy, axis=0).to('erg / cm3')
                
          IC = Radiative.InverseCompton(ECBPL, seed_photon_fields=[['SSC', Esy, phn_sy]], 
                                        Eemin=emin, Eemax=eemax * u.eV, nEed=20)
        
          #--------------------------- SYN and IC in detector frame-------------------------------------
      
          
          synch_comp_approx=(Gamma[i] ** 2.) * SYN.sed(data['energy'] / Gamma_i *redf, distance=self.Dl)*redf
          print("synch_comp_approx max",np.max(synch_comp_approx)) 
          self.synch_comp_approx[i,:] = synch_comp_approx 
          print(type(self.synch_comp_approx))

          ic_comp_approx=(Gamma[i] ** 2.) *  IC.sed(data['energy'] / Gamma_i* redf, distance=self.Dl)*redf
          print("ic_comp_approx max",np.max(ic_comp_approx))
          self.ic_comp_approx [i,:] =   ic_comp_approx 
          print(type(self.ic_comp_approx))

          #SSC_wo_abs = synch_comp_approx + ic_comp_approx
          #model_wo_abs += SSC_wo_abs
          
          #--------------------------------------------------------------------------------------------------------------------

          # --- Parametri ---
          Dl = self.Dl  # distanza con unità (es. Mpc)
          E_obs = data['energy'].quantity  # array di energie con unità (es. eV)

          #unit_syn = SYN.sed(E_obs[0:1], distance=Dl).unit
          n_theta =10 
          #theta_min=self.theta_limits[i]
          #theta_max=self.theta_limits[i+1]     # risoluzione angolare, aumenta se serve

          #thetas_deg= np.linspace(theta_inf[i], theta_sup[i], n_theta)
  
          #thetas_rad = np.deg2rad(thetas_deg) 
          #print("Theta values1 (rad):", thetas_rad)

          mu_max= np.cos(np.deg2rad(theta_inf[i]))
          mu_min = np.cos(np.deg2rad(theta_sup[i]))
          mu_vals = np.linspace(mu_min, mu_max, n_theta)
         
          
          thetas = np.arccos(mu_vals)  # uniform in mu
          print("thetas values2 (rad):", thetas)
          #weights = np.sin(thetas) * 2*np.pi
          deltas = doppler_factor(Gamma[i], thetas)  # shape (n_theta,)  approximation because gamma is varyng little in the shell
          print("Doppler factors:", deltas)
         

          # --- Loop sulle energie, vettoriale sugli angoli ---
          synch_flux = []
          ic_flux = []

          for E in E_obs:
              # energia comovente per ogni angolo
              E_com = (E * redf / deltas).to(u.eV)

              # calcolo delle SED (array 1D)
              sed_syn = SYN.sed(E_com, distance=Dl).value
              sed_ic  = IC.sed(E_com,  distance=Dl).value

              integrand_syn = (deltas**2) * sed_syn # * weights
              integrand_ic  = (deltas**2) * sed_ic # * weights

              # integrazione su theta (trapz veloce)
              #flux_syn = weights* np.trapz(integrand_syn, thetas)
              flux_syn = 2*np.pi * np.trapz(integrand_syn, mu_vals)
              flux_ic  = 2*np.pi * np.trapz(integrand_ic,  mu_vals)

              synch_flux.append(flux_syn)
              ic_flux.append(flux_ic)

          
          self.synch_comp[i,:] = synch_flux* (u.erg / (u.cm**2 * u.s)) 
          self.ic_comp [i,:] =   ic_flux* (u.erg / (u.cm**2 * u.s)) 
          
          SSC_wo_abs = synch_flux* (u.erg / (u.cm**2 * u.s))  + ic_flux* (u.erg / (u.cm**2 * u.s)) 
          model_wo_abs += SSC_wo_abs
        
          #-------------------------- Gamma Gamma Absorption ----------------------------------------------------------------------
          # Optical depth in a shell of width R/(9*Gamma) after transformation of the gamma ray energy of the data in the grb frame
          tauval = tau_val(data['energy'] / Gamma[i] * redf, Esy, phn_sy, self.sizer[i] / (9 * self.gamma[i]) * u.cm)
          
          #--------------------------------METHOD 1 ---------------------------------------------------------------------
          #self.synch_compGG = self.synch_comp * np.exp(-tauval)
          #self.ic_compGG = self.ic_comp * np.exp(-tauval)
          #model = (self.synch_compGG + self.ic_compGG) 

          #-------------------------------- METHOD 2 --------------------------------------------------------------------
          """mask = tauval > 1.0e-4  # fixed level, you can choose another one
          self.synch_compGG2[i,:] = synch_comp.copy()
          self.ic_compGG2[i,:] = ic_comp.copy()
          self.synch_compGG2[i,:][mask] = synch_comp[mask] / (tauval[mask]) * (1. - np.exp(-tauval[mask]))
          self.ic_compGG2[i,:][mask] = ic_comp[mask] / (tauval[mask]) * (1. - np.exp(-tauval[mask]))
          SSC = self.synch_compGG2[i,:] + self.ic_compGG2[i,:]
          model += SSC"""
          
          #-------------------- save the electron distrivution ---------------------------
          ener = np.logspace(np.log10(emin.to('GeV').value), 8,500) * u.GeV  # Energy range to save the electron distribution from emin to 10^8 GeV
          eldis = ECBPL(ener)  # Compute the electron distribution
          electron_distribution = (ener, eldis)
          
        print("-------------------- Ending strucutured GRB computation --------------------")
        print("")
        
        return model,model_wo_abs, electron_distribution  # model returns model and electron distribution


    def get_Benergydensity(self):
        """
        Returns the magnetic field energy density in cgs system
        """

        bedens = (10. ** self.pars[4]) ** 2. / (8. * np.pi)  # free parameter 4 is the B field
        return bedens * u.erg / u.cm / u.cm / u.cm

    def get_Eltotenergy(self):
        """
        Returns total energy in the electron distribution
        """

        return self.Wesyn  # which is the total electron energy injected
      
    def plot_sed_fast(self, emin, emax, ymin, ymax):
        
        """ Parameters
          emin : float
            minimum energy of the interval (in eV)
          emax : float
            maximum energy of the interval (in eV)
          ymin : float
            minimum value for the y-axis (in erg/cm2/s)
          ymax : float
            maximum value for the y-axis (in erg/cm2/s)
        """
        
        bins = int(np.log10(emax/emin) * 20.)  # use 20 bins per decade
        newene = Table([np.logspace(np.log10(emin), np.log10(emax), bins) * u.eV], names=['energy'])  # energy in eV
        #self.naimamodel = self._SSSmodel_ind1fixed
  
        model = self.naimamodel(self.pars, newene)  # make sure we are computing the model for the new energy range
        SSC=model[0]
        
        #---------------------------------------------- Color ----------------------------------------------
        
        def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
            new_cmap = LinearSegmentedColormap.from_list(
                f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
                cmap(np.linspace(minval, maxval, n)))
            return new_cmap
          
        vmin = 0.1
        vmax = 1
        original_cmap = plt.cm.plasma
        cmap1 = truncate_colormap(original_cmap, vmin, vmax)

        #----------------------------------------------- Plot ----------------------------------------------
        plt.figure(figsize=(12,8))
        plt.rc('font', family='sans')
        plt.rc('mathtext', fontset='custom')

        plt.loglog(newene,SSC,lw=2,label='SSC',c=cmap1(0.2))
        plt.loglog(newene,self.synch_compGG2,lw=2,label='SYN',c=cmap1(0.4))
        plt.loglog(newene,self.ic_compGG2,lw=2,label='IC',c=cmap1(0.7))

        plt.xlabel('Photon energy [{0}]'.format(newene['energy'].unit.to_string('latex_inline')))
        plt.ylabel('$E^2 dN/dE$ [{0}]'.format(SSC.unit.to_string('latex_inline')))

        #plt.xlim(emin, emax)
        plt.ylim(ymin, ymax)
        plt.tight_layout()
        plt.legend(loc='lower left')


        plt.title(f"SSC test",fontsize=15)
        plt.grid(True, which="both", linestyle="--", alpha=0.6)
        plt.show
        #---------------------------------------------------------------------------------------------------- 
        
    def plot_sed(self, emin, emax,order_bottom=6,plot_approx_spectrum=True,plot_true_spectrum=False,plot_approx_shells=False,plot_true_shells=False,
                 Save=False,Name="SSC_plot2",path="./"):
        
        """ Parameters
          emin : float
            minimum energy of the interval (in eV)
          emax : float
            maximum energy of the interval (in eV)
        """
        
        bins = int(np.log10(emax/emin) * 20.)  # use 20 bins per decade
        newene = Table([np.logspace(np.log10(emin), np.log10(emax), bins) * u.eV], names=['energy'])  # energy in eV

        #---------------------------------------------- Color ----------------------------------------------
        
        def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
            new_cmap = LinearSegmentedColormap.from_list(
                f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
                cmap(np.linspace(minval, maxval, n)))
            return new_cmap
          
      
        cmap1 = truncate_colormap(plt.cm.plasma, 0.1, 1)
        cmap2 = truncate_colormap(plt.cm.viridis, 0.1, 0.9)
        
        #----------------------------------------------- Report --------------------------------------------
        print("------------------------------------ Short Report---------------------------------------------------")
        Gamma = '\u0393'
        eta = '\u03B7'
        print(f"{Gamma} factor = {self.gamma}")
        print(f"{eta}_B = {self.eta_b}")
        print(f"{eta}_e = {self.eta_e}")
        print(f"Shell Radius",self.sizer*u.cm)
        print("---------------------------------------------------------------------------------------------------")

        #----------------------------------------------- Plot ----------------------------------------------
        plt.figure(figsize=(12,8))
        plt.rc('font', family='sans')
        plt.rc('mathtext', fontset='custom')
        
        if plot_approx_spectrum==True:
            total_synch = np.sum(self.synch_comp_approx, axis=0)
            total_ic  = np.sum(self.ic_comp_approx, axis=0)
            SSC_approx = total_synch+ total_ic

            SSC_val = np.clip(SSC_approx.value, 1e-30, 1e50)  # limiti ragionevoli
            ymax=np.max(SSC_val)
            ymin=np.min(SSC_val)
        
            ordine = int(np.ceil(np.log10(ymax)))
            ymax = 10**(ordine)
            ymin = 10**(ordine - order_bottom)

            plt.loglog(newene,total_synch,lw=2,label='SYN_approx',ls='--',c="black")
            plt.loglog(newene,total_ic,lw=2,label='IC_approx',ls=':',c="gray")
            plt.loglog(newene,SSC_approx,lw=2,label='SSC_approx',c=cmap1(0.1))

        if plot_true_spectrum==True:

            total_synch_True = np.sum(self.synch_comp, axis=0)
            total_ic_True   = np.sum(self.ic_comp, axis=0)
            SSC_True = total_synch_True + total_ic_True

            SSC_val = np.clip(SSC_True.value, 1e-30, 1e50)  # limiti ragionevoli
            ymax=np.max(SSC_val)
            ymin=np.min(SSC_val)
        
            ordine = int(np.ceil(np.log10(ymax)))
            ymax = 10**(ordine)
            ymin = 10**(ordine - order_bottom)
            
            plt.loglog(newene,total_synch_True,lw=2,label='SYN_True',ls='--',c="black")
            plt.loglog(newene,total_ic_True,lw=2,label='IC_True',ls=':',c="gray")
            plt.loglog(newene,SSC_True,lw=2,label='SSC_True',c=cmap2(0.1))
        
        if plot_approx_shells==True:
            n_shells= self.shells
            for i in range(n_shells):
                shell_flux = (self.synch_comp_approx[i,:] + self.ic_comp_approx[i,:])
                plt.loglog(newene, shell_flux, lw=2, ls='--', color=cmap1(i / n_shells), label=f'shell{i} total')
                

        if plot_true_shells==True:
            n_shells= self.shells
            for i in range(n_shells):
                shell_flux = (self.synch_comp[i,:] + self.ic_comp[i,:])
                plt.loglog(newene, shell_flux, lw=2, ls='--', color=cmap2(i / n_shells), label=f'shell{i} total')
                
        
        sed_unit = (u.erg / (u.cm**2 * u.s * u.eV))
        plt.xlabel('Photon energy [{0}]'.format(newene['energy'].unit.to_string('latex_inline')),fontsize=15)
        plt.ylabel('$E^2 dN/dE$ [{0}]'.format(sed_unit.to_string('latex_inline')),fontsize=15)

        #plt.xlim(emin, emax)
        plt.ylim(ymin, ymax)
        plt.tight_layout()
        plt.legend(loc='lower left')


        plt.title(f"SSC test",fontsize=15)
        plt.grid(True, which="both", linestyle="--", alpha=0.6)

        if Save:
            plt.title(f"{Name}",fontsize=15)
            plt.savefig(f"{path}SED_{Name}.jpg", format="jpg", dpi=300)
        
            print(f"Plot saved as: {path}SED_{Name}.png/pdf")

        plt.show
        #---------------------------------------------------------------------------------------------------- 
      
    def plot_jet_profile(self,bottom_order=3,Save=False,Name="Jet_profile",path="./"):
        
        theta = np.linspace(0, 30, 500)
        plt.figure(figsize=(8,5))
        if self.energy_profile=='gaussian':
              E = self.E_theta_gaussian(theta, thetaw_deg=12)
              plt.plot(theta, E, label=r"Gaussiano troncato °" )
        elif self.energy_profile == 'powerlaw':
              E = self.E_theta_powerlaw(theta,thetaw_deg=20,b=4.5)
              plt.plot(theta, E, label=fr"Power-law, $b$ = {4.5}")
        else:
                raise ValueError(f"Unknown energy profile: {self.energy_profile}")   
        
        Emax = np.max(E)
        ymax = 10**np.ceil(np.log10(Emax))
        ymin = ymax / (10**bottom_order)
        
        plt.xlabel("θ (gradi)")
        plt.ylabel(r"$E_{\mathrm{iso}}(\theta)$ [erg]")
        plt.title("Profilo angolare dell'energia isotropica")
        plt.yscale("log")
        plt.ylim(ymin, ymax)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if Save:
            plt.title(f"{Name}",fontsize=15)
            plt.savefig(f"{path}SED_{Name}.jpg", format="jpg", dpi=300)
        
            print(f"Plot saved as: {path}SED_{Name}.png/pdf")
        
        plt.show()
          
        
    def print_GRB_status(self):
        
        print("")
        print("###############################   Structured GRB status - START  #########################################")
        print("")
        print(f"Isotropic Energy(0)-on axis: {self.Eiso_zero} erg")
        print(f"Ambient density around the burst units of cm-3: {self.density}")
        print(f"Average evaluation time: {self.avtime} s")
        print(f"Redshift: {self.redshift}")
        print(f"Luminosity Distance: {self.Dl}")
        print(f"Scenario: {self.scenario}")
        print(f"Magnetic field B: {self.B}")
        print(f"eta e: {self.eta_e}")
        print(f"eta B: {self.eta_b}")
        print(f"---------------------------------------------------------------------------------------")
        print(f"Energy profile: {self.energy_profile}")
        print(f"Number of concentric shells: {self.shells}")
        print(f"Gamma factor (Boosting): {self.gamma}")
        #radius=self.sizer*u.cm
        #deltaR=self.shell_dept*u.cm
        #volume=self.volume*u.cm**3
        """for r in radius:
          print(f"Radius of the shell: {r.to(u.pc):.3e} or {r.to(u.km):.3e}")
        for d in deltaR:
          print(f"Dept of the shell: {d.to(u.pc):.3e} or {d.to(u.km):.3e}")
        for v in volume:
          print(f"Volume of the shell: {v.to(u.pc**3):.3e} or {v.to(u.km**3):.3e}")"""
        print(f"Shock energy omega: {self.shock_energy}")
        print(f"Theta core: {self.thetacore}")
        print(f"Theta limits: {self.theta_limits}")
        print(f"Theta shells: {self.theta_shells}")
        print(f"Average Energies array: {self.Eavg_array}")
        print(f"Minimum injection energy for the particle distribution: {self.Emin}")
        print(f"Total energy in the electrons: {self.Wesyn}")
        print("")
        print("###############################   Structured GRB status - END   #########################################")
        print("")


    def plot_gamma_R_3D_range(self, t_min, t_max, theta_min, theta_max, n_t=100, n_theta=100,slice=False,elevation=30,azimut=45,
                              Save=False,Name="Gamma_R_3D",path="./"):
        """
        Plotta due superfici 3D:
        - Gamma(theta, time)
        - R(theta, time)
        """

        time_array = np.logspace(np.log10(t_min), np.log10(t_max), n_t)
        T_log = np.log10(time_array)

        theta_array = np.linspace(theta_min, theta_max, n_theta)

        # build the grid
        T, Theta = np.meshgrid(T_log, theta_array, indexing='ij')
        Gamma_grid = np.zeros_like(T)
        R_grid = np.zeros_like(T)

        # evaluate Gamma and R on the Grid 
        for i, t in enumerate(time_array):
            Gamma_row, R_row, Energy=self.compute_dynamics(avtime=t, theta=theta_array)  
            Gamma_grid[i, :] = Gamma_row
            R_grid[i, :] = R_row

        # ------------------------------- PLOT ------------------------------------------
        fig = plt.figure(figsize=(16, 8))
        fig.suptitle(f"Gamma and Radius vs Time: {self.energy_profile.capitalize()} profile",fontsize=16)
        
        # Plot Gamma
        ax1 = fig.add_subplot(121, projection='3d')
        surf1 = ax1.plot_surface(T, Theta, Gamma_grid,alpha=0.7, cmap='viridis')
        ax1.set_xlabel('log10(Time [s])')
        ax1.set_ylabel('Theta [rad]')
        ax1.set_zlabel('Gamma')
        ax1.set_title(r"$\Gamma(\theta, t)$")
        ax1.view_init(elev=elevation, azim=azimut)
        cbar1 = fig.colorbar(surf1, ax=ax1, shrink=0.8, aspect=20)
        cbar1.set_label('Gamma factor', fontsize=12)

        # Plot R
        ax2 = fig.add_subplot(122, projection='3d')
        surf2 = ax2.plot_surface(T, Theta, R_grid,alpha=0.7, cmap='plasma')
        ax2.set_xlabel('log10(Time [s])')
        ax2.set_ylabel('Theta [deg]')
        ax2.set_zlabel('Radius [cm]')
        ax2.set_title(r"$R(\theta, t)$")
        ax2.view_init(elev=elevation, azim=azimut)
        cbar2 = fig.colorbar(surf2, ax=ax2, shrink=0.8, aspect=20)
        cbar2.set_label('Radius', rotation=270, labelpad=15, fontsize=12)
        
        #----------------------------- Slice = True ---------------------------------------------
        if slice:
          
          theta_array=self.theta_shells
          norm = Normalize(vmin=np.min(theta_array), vmax=np.max(theta_array))
          colormap1 = cm.viridis  # puoi usare anche 'plasma', 'inferno', etc.
          colormap2 = cm.plasma
          
          for theta_val in theta_array:
            Gamma_vals, R_vals,E = self.compute_dynamics(avtime=time_array, theta=theta_val)
            
            
            color1 = colormap1(norm(theta_val))
            color2 = colormap2(norm(theta_val))
            
            # Gamma curve
            ax1.plot(np.log10(time_array),[theta_val]*len(time_array), Gamma_vals,
                color=color2, linestyle='-',linewidth=2.5,alpha=1.0, label=f'θ={theta_val:.0f}°')

            # R curve
            ax2.plot(np.log10(time_array),[theta_val]*len(time_array),R_vals,
                color=color1, linestyle='-', linewidth=2.5,alpha=1.0,label=f'θ={theta_val:.0f}°')

          ax1.legend(loc='best')
          ax2.legend(loc='best')

          plt.subplots_adjust(wspace=0.4)
          plt.tight_layout()


          if Save:
            plt.title(f"{Name}",fontsize=15)
            plt.savefig(f"{path}{Name}2.jpg", format="jpg", dpi=300)
        
            print(f"Plot saved as: {path}{Name}2.png/pdf")

          plt.show()
          
          fig_slice = plt.figure(figsize=(16, 8))
          fig_slice, (axg, axr) = plt.subplots(1, 2, figsize=(16, 7), sharex=True)
          #axg = fig.add_subplot(121)
          #axr = fig.add_subplot(122)
          
          fig_slice.suptitle(f"Slices of Gamma e Radius vs Time \nProfilo: {self.energy_profile.capitalize()}, θ ∈ [{np.min(theta_array):.0f}°, {np.max(theta_array):.0f}°]",
                            fontsize=16)

          for theta_val in theta_array:
              Gamma_vals, R_vals,E = self.compute_dynamics(avtime=time_array, theta=theta_val)

              label = f'θ={theta_val:.0f}°'
              color1 = colormap1(norm(theta_val))
              color2 = colormap2(norm(theta_val))

              axg.plot(time_array, Gamma_vals, label=label, linewidth=2.5,color=color2)
              axr.plot(time_array, R_vals, label=label, linewidth=2.5,color=color1)

          axg.set_xscale('log')
          axr.set_xscale('log')

          axg.set_xlabel('Time [s]')
          axg.set_ylabel('Gamma')
          axg.set_title(r"$\Gamma(\theta, t)$")

          axr.set_xlabel('Time [s]')
          axr.set_ylabel('Radius [cm]')
          axr.set_title(r"$R(\theta, t)$")

          axg.legend()
          axr.legend()
          
          axg.grid(True, which='both', linestyle='--', alpha=0.6)
          axr.grid(True, which='both', linestyle='--', alpha=0.6)

          plt.tight_layout()

          if Save:
            plt.title(f"{Name}",fontsize=15)
            plt.savefig(f"{path}{Name}1.jpg", format="jpg", dpi=300)
        
            print(f"Plot saved as: {path}{Name}1.png/pdf")

          plt.show()


