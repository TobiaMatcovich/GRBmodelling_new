

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



class GRBModel1:
    """
    The spectral modelling presented here is based on the picture of particle acceleration at the forward shock,
    which propagates outwards through the circumburst material (see   `R. D. Blandford, C. F. McKee,Physics of Fluids19, 1130 (1976)`).
    
    It is possible to choose between 3 scenario options:
      - ISM : homogeneous interstellar medium of density n
      - Wind : density of the material with r^-2 dependance and dictated by a certain value of mass loss rate
               of the star `mass_loss` (in solar masses per year) and a certain wind speed `wind_speed` (in km/s)
      - average : an ISM scenario but with parameters of the size of the shock that are an average of the 2 previous cases
      
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

    def __init__(self, eiso, dens, tstart, tstop, redshift, pars, labels,
                 scenario='ISM',
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
            
        self.Eiso = eiso  # Eiso of the burst
        self.density = dens  # ambient density around the burst units of cm-3
        self.tstart = tstart  # units of s
        self.tstop = tstop  # units of s
        self.energy_grid = np.asarray(data['energy'])
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
        self.doppler= 0
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
        self.synch_comp_approx = 0 
        self.ic_comp_apprx = 0  
        self.synch_comp = 0  # Spectrum of the synchrotron component
        self.ic_comp = 0  # Spectrum of the IC component
        self.synch_compGG = 0  # synchrotron component of the model with gammagamma absorption with METHOD 1
        self.ic_compGG = 0  # inverse compton component of the model with gammagamma absorption with METHOD 1
        self.synch_compGG2 = 0  # synchrotron component of the model with gammagamma absorption with METHOD 2
        self.ic_compGG2 = 0  # inverse compton component of the model with gammagamma absorption with METHOD 2
        
        
        
    def gammaval(self,avtime):
        """
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
        """            
        if (self.scenario == 'ISM'):
            gamma = (1. / 8.) ** (3. / 8.) * (3.0 * self.Eiso / (4.0 * np.pi * self.density * mpc2_erg * ((c * avtime) ** 3.0))) ** 0.125
            self.gamma=gamma
            radius = 8. * c * avtime * self.gamma ** 2.
            self.sizer = radius
            self.depthpar = 9. / 1.
        else:
            text = "Chosen scenario: %s\n" \
                   "The scenario indicated not found. Please choose 'ISM' scenario" % self.scenario
            raise ValueError(text)       
        
        return gamma,radius   
          
    def calc_photon_density(self, Lsy, sizereg):
        """
        This is a thin shell, we use the approximation that the radiation is emitted in a region with radius sizereg.
        (see e.g. Atoyan, Aharonian, 1996). No correction factor (2.24) needed because of thin shell.

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
        
        return Lsy / (4. * np.pi * sizereg ** 2. * c * u.cm / u.s)
      
    def load_model_and_prior(self):
        """
        Associates the bound methods
        naimamodel and lnprior to the chosen
        model and prior function.

        Modify here if you want to change the model
        or the priors
        """
        print("")
        print(" ------------------ Starting GRB initialization ------------------")
        Gamma,raduius=self.gammaval(self.avtime) # call the function to compute the basic GRB initialization parameters
        print("Gamma",Gamma)
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

        print(" ------------------ Ending GRB initialization ------------------")
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
        print("-------------------- Starting GRB computation --------------------")

        eta_e = 10. ** pars[0] # parameter 0: fraction of available energy ending in non-thermal electrons
        self.eta_e = eta_e
        ebreak = 10. ** pars[1] * u.TeV  # parameter 1: linked to break energy of the electron distribution (as log10)
        alpha1 = pars[2] - 1.  # fixed to be a cooling break
        alpha2 = pars[2]  # parameter 2: high energy index of the ExponentialCutoffBrokenPowerLaw
        e_cutoff = (10. ** pars[3]) * u.TeV  # parameter 3: High energy cutoff of the electron distribution (as log10)
        
        bfield = 10. ** (pars[4]) * u.G  # parameter 4: Magnetic field (as log10)
        self.B=bfield
        redf = 1. + self.redshift  # redshift factor
        
        #doppler = self.gamma  # assumption of doppler boosting ~ Gamma OLD
        #doppler = self.gamma 
        size_reg = self.sizer * u.cm  # size of the region as astropy quantity
        
        # ------------------- Volume shell where the emission takes place. The factor 9 comes from considering the shock in the ISM ----------------
        #  (Eq. 7 from GRB190829A paper from H.E.S.S. Collaboration)
        deltaR=self.sizer / (self.depthpar * self.gamma)
        self.shell_dept=deltaR
        
        solid_angle=4.0*np.pi
        vol = solid_angle * self.sizer ** 2. * deltaR
        self.volume=vol
        
        shock_energy = 2. * self.gamma ** 2. * self.density * mpc2_erg *u.erg #/u.cm**3 # available energy in the shock
        self.shock_energy = shock_energy
        
        eemax = e_cutoff.value * 1e13 # maximum energy of the electron distribution, based on 10 * cut-off value in eV (1 order more then cutoff)
        self.eta_b = (bfield.value ** 2 / (np.pi * 8.)) / shock_energy.value  # ratio between magnetic field energy and shock energy   
        
        #---------------------------------------------------------------------------------------------------------------------------
        ampl = 1. / u.eV  # temporary amplitude
        ECBPL = Models.ExponentialCutoffBrokenPowerLaw(ampl, 1. * u.TeV, ebreak, alpha1, alpha2, e_cutoff)
        #---------------------------------------------- E min iterative process ----------------------------------------------------
        Emin_0=1e9*u.eV
        Emin_0_exp=9
        energies = np.logspace(Emin_0_exp, np.log10(eemax), 100) * u.eV
        eldis = ECBPL(energies)
        fact1 = eta_e * self.gamma * mpc2
        E_medium = trapz_loglog(energies * eldis, energies) / trapz_loglog(eldis, energies)
        K=E_medium/fact1
        
        emin= Emin_0/K  # calculation of the minimum injection energy. See detailed model explanationn(! iteration)
        self.Emin = emin
        #--------------------------------------------- E min non iterative process -------------------------------------------------
        
        #emin=(p-2)/(p-1)*fact1 # p=? abbiamo una broken powerlaw, quindi sono 2 
        
        #----------- (https://www.cv.nrao.edu/~sransom/web/Ch5.html)------------------------
        SYN = Radiative.Synchrotron(ECBPL, B=bfield, Eemin=emin, Eemax=eemax * u.eV, nEed=20)
        #----------------------------------------------------------------------------------------------------------------------------
        
        amplitude = ((eta_e * shock_energy * vol) / SYN.compute_Etot(Eemin=emin, Eemax=eemax * u.eV)) / u.eV
        
        ECBPL = Models.ExponentialCutoffBrokenPowerLaw(amplitude, 1. * u.TeV, ebreak, alpha1, alpha2, e_cutoff)
        SYN = Radiative.Synchrotron(ECBPL, B=bfield, Eemin=emin, Eemax=eemax * u.eV, nEed=20)
  
        self.Wesyn = SYN.compute_Etot(Eemin=emin, Eemax=eemax * u.eV)           # E tot in the electron distribution
        
        #----- energy array to compute the target photon number density to compute IC radiation and gamma-gamma absorption -----------
        
        cutoff_charene = np.log10((synch_charene(bfield, e_cutoff)).value)      # characteristic energy at the electron cutoff
        min_synch_ene = -4                                                      # minimum energy to start sampling the synchrotron spectrum
        bins_per_decade = 20                                                    # 20 bins per decade to sample the synchrotron spectrum
        bins = int((cutoff_charene - min_synch_ene) * bins_per_decade)
        
        #---------------------------------------------------------------------------------------------------------------------
        Esy = np.logspace(min_synch_ene, cutoff_charene + 1, bins) * u.eV
        Lsy = SYN.flux(Esy, distance=0 * u.cm)  # number of synchrotron photons per energy per time (units of 1/eV/s)
        phn_sy = self.calc_photon_density(Lsy, size_reg)   # number density of synchrotron photons (dn/dE) units of 1/eV/cm3
        #----------------------------------------------------------------------------------------------------------------------
        self.esycool = (synch_charene(bfield, ebreak))
        self.synchedens = trapz_loglog(Esy * phn_sy, Esy, axis=0).to('erg / cm3')
        
        
        IC = Radiative.InverseCompton(ECBPL, seed_photon_fields=[['SSC', Esy, phn_sy]], 
                                      Eemin=emin, Eemax=eemax * u.eV, 
                                      nEed=20)
        
        #--------------------------- SYN and IC in detector frame-------------------------------------
        
        unit_syn = (u.erg / (u.cm**2 * u.s)) 
        
        self.synch_comp_approx = (self.gamma ** 2.) * SYN.sed(data['energy'] / self.gamma * redf, distance=self.Dl)*redf* unit_syn
        self.ic_comp_approx =    (self.gamma ** 2.) *  IC.sed(data['energy'] / self.gamma * redf, distance=self.Dl)*redf* unit_syn
        
        #print("Approx synch comp:", self.synch_comp_approx)
        #print("Approx IC comp:", self.ic_comp_approx)


        # --- Doppler factor ---
        def doppler(gamma, theta_rad):
            beta = np.sqrt(1 - 1 / gamma**2)
            return 1 / (gamma * (1 - beta * np.cos(theta_rad)))

        # --- Parametri ---
        Gamma = self.gamma
        Dl = self.Dl  # distanza con unità (es. Mpc)
        E_obs = data['energy'].quantity  # array di energie con unità (es. eV)

    

        # --- Griglia angolare per l'integrazione ---
        #theta_max = np.pi/2   # oppure il tuo angolo di jet (es. theta_j)
        
        theta_max = 4/Gamma   # oppure il tuo angolo di jet (es. theta_j)
        print("Theta max (deg):", theta_max*180/np.pi," = 4\Gamma")

        n_theta = 10         # risoluzione angolare, aumenta se serve
        thetas = np.linspace(0, theta_max, n_theta)
        weights = np.sin(thetas) * 2*np.pi
        deltas = doppler(Gamma, thetas)  # shape (n_theta,)
        self.doppler=deltas  # Doppler factor at theta=0

        #print("Doppler factor:", self.doppler)
        print("Gamma factor ISO :", self.gamma) 
        print(" Doppler factor(theta)",self.doppler) 

        # --- Loop sulle energie, vettoriale sugli angoli ---
        synch_flux = []
        ic_flux = []

        for E in E_obs:
            # energia comovente per ogni angolo
            E_com = (E * redf / deltas).to(u.eV)

            # calcolo delle SED (array 1D)
            sed_syn = SYN.sed(E_com, distance=Dl).to(unit_syn).value  # shape (n_theta,)
            sed_ic  = IC.sed(E_com,  distance=Dl).to(unit_syn).value

            # integrandi
            integrand_syn = (deltas**2) * sed_syn * weights
            integrand_ic  = (deltas**2) * sed_ic  * weights

            # integrazione su theta (trapz veloce)
            flux_syn = np.trapz(integrand_syn, thetas)
            flux_ic  = np.trapz(integrand_ic,  thetas)

            synch_flux.append(flux_syn)
            ic_flux.append(flux_ic)


        self.synch_comp = synch_flux* unit_syn
        self.ic_comp    = ic_flux  * unit_syn

        #print("self.synch_comp:", self.synch_comp)
        #print("self.ic_comp:", self.ic_comp)

        model_wo_abs = self.synch_comp + self.ic_comp

        #-------------------------- Gamma Gamma Absorption ----------------------------------------------------------------------
        # Optical depth in a shell of width R/(9*Gamma) after transformation of the gamma ray energy of the data in the grb frame
        
        tauval = tau_val(data['energy'] / self.doppler[0] * redf, Esy, phn_sy, self.sizer / (9 * self.gamma) * u.cm)
        #tauval = tau_val(data['energy'] / Gamma* redf, Esy, phn_sy, self.sizer / (9 * self.gamma) * u.cm)

        R_eff = (self.sizer / (9 * self.gamma)) * u.cm
        tauval = tau_val(data['energy'] / self.gamma * redf, Esy, phn_sy, R_eff)
        
        """print("type(synch_comp):", type(self.synch_comp))
        print("synch_comp.shape:", getattr(self.synch_comp, 'shape', None))
        print("type(ic_comp):", type(self.ic_comp))
        print("ic_comp.shape:", getattr(self.ic_comp, 'shape', None))
        print("type(tauval):", type(tauval))
        print("tauval.shape:", getattr(tauval, 'shape', None))"""


        
        #--------------------------------METHOD 1 ---------------------------------------------------------------------
        #self.synch_compGG = self.synch_comp * np.exp(-tauval)
        #self.ic_compGG = self.ic_comp * np.exp(-tauval)
        #model = (self.synch_compGG + self.ic_compGG) 

        #-------------------------------- METHOD 2 --------------------------------------------------------------------
        mask = tauval > 1.0e-4  # fixed level, you can choose another one
        self.synch_compGG2 = self.synch_comp.copy()
        self.ic_compGG2 = self.ic_comp.copy()
        self.synch_compGG2[mask] = self.synch_comp[mask] / (tauval[mask]) * (1. - np.exp(-tauval[mask]))
        self.ic_compGG2[mask] = self.ic_comp[mask] / (tauval[mask]) * (1. - np.exp(-tauval[mask]))
        model = (self.synch_compGG2 + self.ic_compGG2)
        
        #-------------------- save the electron distrivution ---------------------------
        ener = np.logspace(np.log10(emin.to('GeV').value), 8,500) * u.GeV  # Energy range to save the electron distribution from emin to 10^8 GeV
        eldis = ECBPL(ener)  # Compute the electron distribution
        electron_distribution = (ener, eldis)

        print("-------------------- ending GRB computation --------------------")
        print("")

        return model,model_wo_abs, electron_distribution  # returns model and electron distribution


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
        
    def plot_sed(self, emin, emax,order_bottom,Save=False,Path='',Name=''):
        
        """ Parameters
          emin : float
            minimum energy of the interval (in eV)
          emax : float
            maximum energy of the interval (in eV)
        """
        energy=self.energy_grid
        # controlla che il range richiesto sia dentro la griglia
        if emin < energy.min() or emax > energy.max():
            raise ValueError(
                f"Range richiesto [{emin}, {emax}] eV fuori dalla griglia del modello "
                f"[{energy.min()}, {energy.max()}] eV."
            )
        #bins = int(np.log10(emax/emin) * 20.)  # use 20 bins per decade
        #newene = Table([np.logspace(np.log10(emin), np.log10(emax), bins) * u.eV], names=['energy'])  # energy in eV
  
        #model = self.naimamodel(self.pars, newene)  # make sure we are computing the model for the new energy range
        
        #SSC=model[0]
        #SSC_no_abs=model[1]


        mask = (energy >= emin) & (energy <= emax)
        energy_plot = energy[mask]
 

        SSC= self.synch_comp + self.ic_comp
        SSC_approx= self.synch_comp_approx + self.ic_comp_approx 
    
        SSC_val = np.clip(SSC_approx.value, 1e-30, 1e50)  # limiti ragionevoli
        ymax = np.max(SSC_val)
        ymin = np.min(SSC_val)

        #ymax=np.max(SSC).value
        #ymin=np.min(SSC).value
        
        ordine = int(np.ceil(np.log10(ymax)))
        ymax = 10**ordine
        ymin = 10**(ordine - order_bottom)

        #---------------------------------------------- Color ----------------------------------------------
        
        def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
            new_cmap = LinearSegmentedColormap.from_list(
                f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
                cmap(np.linspace(minval, maxval, n)))
            return new_cmap
          
      
        cmap1 = truncate_colormap(plt.cm.plasma, 0.1, 1)
        #cmap2 = truncate_colormap(plt.cm.viridis, 0.1, 0.9)
        
        #----------------------------------------------- Report --------------------------------------------
        print("---------------------------------------------------------------------------------------------------")
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

        plt.loglog(energy_plot,self.synch_comp[mask],lw=2,label='SYN',c=cmap1(0.4))
        plt.loglog(energy_plot,self.ic_comp[mask],lw=2,label='IC',c=cmap1(0.7))
        plt.loglog(energy_plot,SSC[mask],lw=2,label='SSC',c=cmap1(0.2))
        plt.loglog(energy_plot,SSC_approx[mask],lw=2,label='SSC_approx',c=cmap1(0.9))
        #plt.loglog(newene,SSC_no_abs,lw=2,label='SSC no abs',c=cmap1(0.1))
        
        #plt.loglog(newene,self.synch_comp,lw=2,label='SYN-no absorbtion',c=cmap2(0.4))
        #plt.loglog(newene,self.ic_comp,lw=2,label='IC-no absorbtion',c=cmap2(0.7))
        #plt.loglog(newene,SSC_no_abs,lw=2,label='SSC-no absorbtion',c=cmap2(0.2))
        

        #plt.xlabel('Photon energy [{0}]'.format(energy_plot['energy'].unit.to_string('latex_inline')),fontsize=15)
        plt.xlabel("Photon energy [eV]", fontsize=15)
        plt.ylabel('$E^2 dN/dE$ [{0}]'.format(SSC_approx.unit.to_string('latex_inline')),fontsize=15)

        #plt.xlim(emin, emax)
        plt.ylim(ymin, ymax)
        plt.tight_layout()
        plt.legend(loc='lower left')

        plt.grid(True, which="both", linestyle="--", alpha=0.6)
      
        if Save:
            plt.title(f"{Name}",fontsize=15)
            plt.savefig(f"{Path}SED_{Name}.jpg", format="jpg", dpi=300)
        
            print(f"Plot saved as: {Path}SED_{Name}.png/pdf")
        plt.show
        #---------------------------------------------------------------------------------------------------- 
      
    def print_GRB_status(self):
        print("")
        print("###############################   GRB status - START   #########################################")
        print("")
        print(f"Isotropic Energy: {self.Eiso} erg")
        print(f"Ambient density around the burst units of cm-3: {self.density}")
        print(f"Average evaluation time: {self.avtime} s")
        print(f"Redshift: {self.redshift}")
        print(f"Luminosity Distance: {self.Dl}")
        print(f"Scenario: {self.scenario}")
        print(f"Magnetic field B: {self.B}")
        print(f"eta e: {self.eta_e}")
        print(f"eta B: {self.eta_b}")
        print("---------------------------------------------------------------------------------------")
        print(f"Gamma factor (Boosting): {self.gamma}")
        radius=self.sizer*u.cm
        deltaR=self.shell_dept*u.cm
        volume=self.volume*(u.cm)**3
        print(f"Radius of the shell: {radius.to(u.pc):.3e} or {radius.to(u.km):.3e}")
        print(f"Dept of the shell: {deltaR.to(u.pc):.3e} or {deltaR.to(u.km):.3e}")
        print(f"Volume of the shell: {volume.to(u.pc**3):.3e} or {volume.to(u.km**3):.3e}")
        print(f"Shock energy density (omega): {self.shock_energy}")
        print(f"Minimum injection energy for the particle distribution: {self.Emin}")
        print(f"Total energy in the electrons: {self.Wesyn}")
        print("")
        print("###############################   GRB status - END   #########################################")
        print("")
 
    def plot_gamma_radius_vs_time(self, tmin=1, tmax=1e7, num=200,Save=False,path='',Name=''):
      """
      Plotta Gamma(t) e R(t) usando la funzione self.gammaval()
      
      Parametri:
      - tmin, tmax: intervallo di tempo in secondi
      - num: numero di punti nel plot
      """

      times = np.logspace(np.log10(tmin), np.log10(tmax), num) * u.s
      gammas = []
      radii = []

      for t in times:
          gamma, radius = self.gammaval(t.to_value(u.s))
          gammas.append(gamma)
          radii.append(radius)

      radii_pc = (np.array(radii) * u.cm).to(u.pc).value

      # Crea sottografi
      fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
      fig.suptitle("Shell evolution in time (2D)", fontsize=16)

      # Plot Gamma(t)
      ax1.plot(times.to_value(u.s), gammas, color='tab:blue')
      ax1.set_ylabel(r"$\Gamma$")
      ax1.set_xscale('log')
      ax1.grid(True)
      ax1.set_title("Lorentz factor evolution")

      # Plot R(t)
      ax2.plot(times.to_value(u.s), radii_pc, color='tab:green')
      ax2.set_xlabel("Tempo [s]")
      ax2.set_ylabel("Raggio [pc]")
      ax2.set_xscale('log')
      ax2.grid(True)
      ax2.set_title("Radius evolution")

      plt.tight_layout(rect=[0, 0, 1, 0.95])  # spazio per il titolo globale

      if Save:
            plt.title(f"{Name}",fontsize=15)
            plt.savefig(f"{path}SED_{Name}.jpg", format="jpg", dpi=300)
        
            print(f"Plot saved as: {path}SED_{Name}.png/pdf")
      
      plt.show()
