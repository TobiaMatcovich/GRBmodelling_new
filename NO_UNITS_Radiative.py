# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst
from log_config import setup_logger
log = setup_logger(__name__)  


import warnings
from collections import OrderedDict
from multiprocessing import Pool

import numpy as np
from astropy import units as u
from astropy.constants import alpha, c, e, hbar, m_e, m_p, sigma_sb
from astropy.utils.data import get_pkg_data_filename
from astropy.table import Table

from Validator import (
    validate_array,
    validate_physical_type,
    validate_scalar,
    check_e_in_eV,
    check_B_in_Gauss
)
#from .model_utils import memoize
from Utils import trapz_loglog,trapz_loglog_nd,simpson_logspace,simpson_logspace_nb,trapz_loglog_nd_fast

from scipy.special import cbrt

import time
from numba import njit,vectorize,prange

#######################################################################################################
__all__ = [
    "Synchrotron",
    "InverseCompton"
]

e = e.gauss
mec2 = (m_e * c**2).cgs
mec2_eV = (m_e * c**2).to("eV").value
mec2_unit = u.Unit(mec2)
eV_to_erg = u.eV.to(u.erg)
erg_to_eV = u.erg.to(u.eV)

#ar = (4 * sigma_sb / c).to("erg/(cm3 K4)")  # costante di radiazione
#r0 = (e**2 / mec2).to("cm")  #raggio classico dell'elettrone 

######################################################################################################
def _validate_ene(ene):
    

    if isinstance(ene, dict) or isinstance(ene, Table):
        try:
            ene = validate_array(
                "energy", u.Quantity(ene["energy"]), physical_type="energy"
            )
        except KeyError:
            raise TypeError("Table or dict does not have 'energy' column")
    else:
        if not isinstance(ene, u.Quantity):
            ene = u.Quantity(ene)
        validate_physical_type("energy", ene, physical_type="energy")

    return ene



def heaviside(x):
    return (np.sign(x) + 1) / 2.0

def simps_points(a,b,N,loglog=False):
    
    if loglog:
        log_a, log_b = np.log10(a), np.log10(b)
        log_range = log_b - log_a
        h_T = log_range / (N - 1)
        h_S = h_T ** 0.5
        N_S = int(np.round(log_range / h_S)) + 1
        #print(N_S)
        x_s = np.logspace(log_a, log_b, N_S)

    else:
        lin_range=a-b
        h_T = (lin_range) / (N - 1)
        h_S = h_T ** 0.5
        N_S = int(np.round((lin_range) / h_S)) + 1
        x_s = np.linspace(a, b, N_S)
        #print(N_S)
    return x_s

####################################################################################################

class BaseRadiative:
    """Base class for radiative models

    This class implements the flux, sed methods and subclasses must implement
    the spectrum method which returns the intrinsic differential spectrum.
    """

    def __init__(self, particle_distribution):
        self.particle_distribution = particle_distribution
        check_e_in_eV(1/self.particle_distribution.amplitude)


    # @memoize
    def flux(self, photon_energy, distance=1 * u.kpc):
        """Differential flux at a given distance from the source.

        Parameters
        ----------
        photon_energy : :class:`~astropy.units.Quantity` float or array
            Photon energy array.

        distance : :class:`~astropy.units.Quantity` float, optional
            Distance to the source. If set to 0, the intrinsic differential
            luminosity will be returned. Default is 1 kpc.
        """
        flux = self._spectrum(photon_energy)#[0]

        if distance != 0:
            flux =flux/ (4 * np.pi * distance.to("cm").value ** 2)
            return flux

        return flux

    def sed(self, photon_energy, distance=1 * u.kpc):
        """Spectral energy distribution at a given distance from the source.

        Parameters
        ----------
        photon_energy : :class:`~astropy.units.Quantity` float or array
            Photon energy array.

        distance : :class:`~astropy.units.Quantity` float, optional
            Distance to the source. If set to 0, the intrinsic luminosity will
            be returned. Default is 1 kpc.
        """
        log.info("")
        log.info("SED ...")
        
        photon_energy_adim=check_e_in_eV(photon_energy)
        sed = self.flux(photon_energy, distance) * (photon_energy_adim**2.0)* eV_to_erg

        return sed
    
###########################################################################################################################    
    
class BaseElectron(BaseRadiative):
    """Implements gamma and nelec properties"""

    def __init__(self, particle_distribution):
        super().__init__(particle_distribution)
        self.param_names = ["Eemin", "Eemax", "nEed"]

    @property
    def _gamma(self): 
        """Lorentz factor array (cached)"""
        
        if hasattr(self, "__gamma_cache"):
            return self.__gamma_cache  # già calcolato → restituisci
        

        log10gmin = np.log10(self.Eemin / mec2).value
        log10gmax = np.log10(self.Eemax / mec2).value
        N = int(np.maximum(10, self.nEed * (log10gmax - log10gmin)))
        
        simpson = False
        if simpson:
            points = simps_points(10**(log10gmin), 10**(log10gmax), N, loglog=True)
        else:
            points = np.logspace(log10gmin, log10gmax, N)
 
        self.__gamma_cache = points
        return points

    @property
    def _nelec(self):
        """Particles per unit lorentz factor (cached)"""
        if hasattr(self, "__nelec_cache"):
            return self.__nelec_cache  # già calcolato → restituisci
    
        pd = self.particle_distribution(self._gamma* mec2)
        #print(f"Particle distribution shape: {pd.shape}")
        return pd.to(1 / mec2_unit).value
    
    @property
    def Etot(self):
        """Total energy in electrons used for the radiative calculation"""
        simpson=False
        if simpson:
            Etot=simpson_logspace(self._gamma * self._nelec, self._gamma * mec2)
        else:
            Etot = trapz_loglog_nd_fast(self._gamma * self._nelec, self._gamma * mec2)
        return Etot

    def compute_Etot(self, Eemin=None, Eemax=None):
        """Total energy in electrons between energies Eemin and Eemax

        Parameters
        ----------
        Eemin : :class:`~astropy.units.Quantity` float, optional
            Minimum electron energy for energy content calculation.

        Eemax : :class:`~astropy.units.Quantity` float, optional
            Maximum electron energy for energy content calculation.
        """
        if Eemin is None and Eemax is None:
            Etot = self.Etot
        else:
            if Eemax is None:
                Eemax = self.Eemax
            if Eemin is None:
                Eemin = self.Eemin

            log10gmin = np.log10(Eemin / mec2).value
            log10gmax = np.log10(Eemax / mec2).value
            gamma = np.logspace(
                log10gmin, log10gmax, max(10, int(self.nEed * (log10gmax - log10gmin)))
            )
            nelec = self.particle_distribution(gamma * mec2).to(1 / mec2_unit).value
            Etot = trapz_loglog(gamma * nelec, gamma * mec2)

        return Etot
    
    
    def set_Etot(self,Etot, Eemin=None, Eemax=None, amplitude_name=None):
        
        """Normalize particle distribution so that the total energy in electrons
        between Eemin and Eemax is Etot

        Parameters
        ----------
        Etot : :class:`~astropy.units.Quantity` float
            Desired energy in electrons.

        Eemin : :class:`~astropy.units.Quantity` float, optional
            Minimum electron energy for energy content calculation.

        Eemax : :class:`~astropy.units.Quantity` float, optional
            Maximum electron energy for energy content calculation.

        amplitude_name : str, optional
            Name of the amplitude parameter of the particle distribution. It
            must be accesible as an attribute of the distribution function.
            Defaults to ``amplitude``.
        """

        Etot = validate_scalar("Etot", Etot, physical_type="energy")
        oldEtot = self.compute_Etot(Eemin=Eemin, Eemax=Eemax)

        if amplitude_name is None:
            try:
                self.particle_distribution.amplitude *= (Etot / oldEtot).decompose()
            except AttributeError:
                log.error(
                    "The particle distribution does not have an attribute"
                    " called amplitude to modify its normalization: you can"
                    " set the name with the amplitude_name parameter of set_Etot"
                )
        else:
            oldampl = getattr(self.particle_distribution, amplitude_name)
            setattr(
                self.particle_distribution,
                amplitude_name,
                oldampl * (Etot / oldEtot).decompose(),  # decompose in fondamental units
            )

###########################################################################################################################
@njit(cache=True) 
def Gtilde(x):
    """
    AKP10 Eq. D7

    Factor ~2 performance gain in using cbrt(x)**n vs x**(n/3.)
    Invoking crbt only once reduced time by ~40%
    """
    #cb = np.cbrt(x)
    cb= np.sign(x) * np.abs(x) ** (1/3.)
    gt1 = 1.808 * cb / np.sqrt(1 + 3.4 * cb**2.0)
    gt2 = 1 + 2.210 * cb**2 + 0.347 * cb**4
    gt3 = 1 + 1.353 * cb**2 + 0.217 * cb**4
    return gt1 * (gt2 / gt3) * np.exp(-x)

@vectorize(['float64(float64)'], target='parallel', fastmath=True)
def Gtilde_vec(x):
    cb = np.sign(x) * abs(x)**(1/3)
    gt1 = 1.808 * cb / np.sqrt(1 + 3.4 * cb**2)
    gt2 = 1 + 2.210 * cb**2 + 0.347 * cb**4
    gt3 = 1 + 1.353 * cb**2 + 0.217 * cb**4
    return gt1 * (gt2 / gt3) * np.exp(-x)

class Synchrotron(BaseElectron):
    """Synchrotron emission from an electron population.

    This class uses the approximation of the synchrotron emissivity in a
    random magnetic field of Aharonian, Kelner, and Prosekin 2010, PhysRev D
    82, 3002 (`arXiv:1006.1045 <http://arxiv.org/abs/1006.1045>`_).

    Parameters
    ----------
    particle_distribution : function
        Particle distribution function, taking electron energies as a
        `~astropy.units.Quantity` array or float, and returning the particle
        energy density in units of number of electrons per unit energy as a
        `~astropy.units.Quantity` array or float.

    B : :class:`~astropy.units.Quantity` float instance, optional
        Isotropic magnetic field strength. Default: equipartition
        with CMB (3.24e-6 G)

    Other parameters
    ----------------
    Eemin : :class:`~astropy.units.Quantity` float instance, optional
        Minimum electron energy for the electron distribution. Default is 1
        GeV.

    Eemax : :class:`~astropy.units.Quantity` float instance, optional
        Maximum electron energy for the electron distribution. Default is 510
        TeV.

    nEed : scalar
        Number of points per decade in energy for the electron energy and
        distribution arrays. Default is 100.
    """
    
    def __init__(self, particle_distribution, B=3.24e-6 * u.G, **kwargs):
        super().__init__(particle_distribution)
        self.B = check_B_in_Gauss(B)
        self.Eemin = 1e9 * u.eV
        self.Eemax = 5.11*1e14*u.eV # the same of 1e9 * mec2
        self.nEed = 100
        self.param_names += ["B"]
        self.__dict__.update(**kwargs)
    

        
    def _spectrum(self, photon_energy):
        """Compute intrinsic synchrotron differential spectrum for energies in
        ``photon_energy``

        Compute synchrotron for random magnetic field according to
        approximation of Aharonian, Kelner, and Prosekin 2010, PhysRev D 82,
        3002 (`arXiv:1006.1045 <http://arxiv.org/abs/1006.1045>`_)."""
        
        #print("")
        log.info("SYN Spectrum...")
        start=time.time()
        
        if not photon_energy.unit.is_equivalent(u.eV):
            raise ValueError("e must have units equivalent to energy (eV)")
        photon_energy.to(u.eV)

        photon_energy_erg = photon_energy.to("erg").value

        Num= np.sqrt(3) * e.value**3 * self.B
        Den = 2 * np.pi * m_e.cgs.value * c.cgs.value**2 * hbar.cgs.value * photon_energy_erg
        factor=Num/Den
        #print(factor[30])
        
        # Critical energy in erg 
        Ec = (3 * e.value * hbar.cgs.value * self.B * self._gamma**2)/ (2 * (m_e * c).cgs.value)    # Broadcast: photon_energy 2D / Ec 1D
        #Ec_eV = Ec / u.eV.to(u.erg)
        
        EpEc = photon_energy_erg[..., np.newaxis] / Ec[np.newaxis, np.newaxis, :]  # shape: (theta, phi, gamma)ma
                             
        dN_dEdt = factor[..., np.newaxis] * Gtilde_vec(EpEc)  # shape (theta, phi, gamma)
        
        nelec = np.array(self._nelec)  # shape (gamma,)
        
        #print("dNdEdt.shape:", dNdEdt.shape)   
        #print("nelec.shape:",nelec .shape)
        #print("gamma.shape:",self._gamma.shape)

        #start1=time.time()
        #spectrum = trapz_loglog(nelec * dNdEdt, self._gamma, axis=-1) / u.s / u.erg
        #time_trapz_loglog=time.time()-start1
        #log.info(f"SYN trapz_loglog:{time.time()-start1:.5f} s")
        #spectrum = spectrum.to("1/(s eV)")

        #start=time.time()
        #spectrum1 = trapz_loglog_nd(nelec * dNdEdt, self._gamma) / u.s / u.erg
        #time_trapz_loglog_nd=time.time()-start
        #print("SYN trapz_loglog_nd:",time.time()-start)
        #spectrum1 = spectrum.to("1/(s eV)")
        
        #start1=time.time()
        #spectrum2 = trapz_loglog_nd_fast(nelec * dNdEdt, self._gamma) / u.s / u.erg
        #log.info(f"SYN trapz_loglog_nd_fast:{time.time()-start1:.5f} s")
        #spectrum2 = spectrum.to("1/(s eV)")
        
        start1=time.time()
        spectrum = simpson_logspace(nelec * dN_dEdt, self._gamma) #/(u.s*u.erg)
        log.info(f"SYN simpson logspace time:{time.time()-start1:.5f} s")
        #spectrum = spectrum.to("1/(s eV)")
        
        spectrum_eV = spectrum/erg_to_eV
        
        #print("scpectrum.shape:",spectrum.shape)
        log.info(f"_spectrum time:{time.time()-start:.5f} s")
        
        return spectrum_eV #,spectrum1,spectrum2,spectrum3,time_trapz_loglog,time_trapz_loglog_nd,time_trapz_loglog_nd_fast,time_simps_loglog
    
############################################################################################################################################

'''@njit(cache=True)
def G12(x, param):
    """
    Eqs 18,19,20 of Khangulyan et al (2014)
    """
    alpha, a, beta, b = param
    G0 = (np.pi**2 / 6.0 + x) * np.exp(-x)
    tmp = 1 + b * x**beta
    g = 1.0 / (a * x**alpha / tmp + 1.0)
    return G0 * g

@njit(cache=True)
def G34(x, param):
    """
    Eqs 20, 24, 25 of Khangulyan et al (2014)
    """
    alpha, a, beta, b, c = param
    pi26 = np.pi**2 / 6.0
    tmp = (1 + c * x) / (1 + pi26 * c * x)
    G0 = pi26 * tmp * np.exp(-x)
    tmp = 1 + b * x**beta
    g = 1.0 / (a * x**alpha / tmp + 1.0)
    return G0 * g  '''


@njit(cache=True)
def fic_element(g, e, p):
        
    b = 4 * p * e
    w = g / e
    den = b*(1-w)
    if den <= 0:
        return 0.0
    q = w / den
    
    if 1.0/(4*e**2) < q < 1.0:
        return 2*q*np.log(q) + (1 + 2*q)*(1 - q) + 0.5*(b*q)**2*(1 - q)/(1 + b*q)
    else:
        return 0.0
        
@njit(cache=True)
def Fic_element(E1,gamma,E0):
    """ This functions copute the Fic Kernel as done in Blumenthal-Gould 1970 (BG70)

    Args:
        gamma (_type_): lorentz factor of the electron (adimensional)
        E0 (_type_): seed photon energy (before teh scattering)
        E1 (_type_): photon energy (after the scattering)

    Returns:
        _type_: BG70 Kernel (no dimension)
    """
        
    Gamma = 4 * gamma * E0 
    w =E1 / gamma
    den = Gamma*(1-w)
    if den <= 0:
        return 0.0
    q = w / den
    
    if 1.0/(4*gamma**2) < q < 1.0:
        return 2*q*np.log(q) + (1 + 2*q)*(1 - q) + 0.5*(Gamma*q)**2*(1 - q)/(1 + Gamma*q)
    else:
        return 0.0




@njit(parallel=True, fastmath=True)
def compute_fic_numba(E1,gamma, E0):
    """
    Versione ultra-ottimizzata:
    - parallel su gamma flattenato
    - memoria contigua
    - zero bounds-checks
    - reshape finale alla forma originale (N-D gamma, Ne, Np)
    """

    # salva la forma originale di gamma
    original_shape = E1.shape
    E1_flat = E1.ravel()         # contiguo in memoria

    NE1 = E1_flat.shape[0]
    Ngamma = gamma.shape[0]
    NE0 = E0.shape[0]

    # array contiguo 3D per massima efficienza
    fic = np.zeros((NE1, Ngamma, NE0),dtype=np.float64)

    # loop esterno parallelo
    for i in prange(NE1):
        E_1 =E1_flat[i]  # carica una volta per tutte
        # loop interni puri → ottimizzabili
        for k in range(Ngamma):
            g = gamma[k]

            for l in range(NE0):
                E_0= E0[l]
                # kernel IC
                fic[i, k, l] = Fic_element(E_1,g,E_0)
    # ricostruisci la forma originale
    return fic.reshape(original_shape + (Ngamma, NE0))


from numba import guvectorize, float64
# Kernel completamente vettorizzato
@guvectorize(
    [(float64, float64, float64, float64[:])],
    '(),(),()->()', target='parallel', fastmath=True
)
def Fic_element_vec(E1, gamma, E0, out):
    Gamma = 4.0 * gamma * E0
    w = E1 / gamma
    den = Gamma * (1.0 - w)

    if den <= 0.0:
        out[0] = 0.0
    else:
        q = w / den
        if (1.0 / (4.0 * gamma**2)) < q < 1.0:
            out[0] = 2.0 * q * np.log(q) + (1.0 + 2.0*q)*(1.0 - q) + 0.5*(Gamma*q)**2*(1.0 - q)/(1.0 + Gamma*q)
        else:
            out[0] = 0.0

# Funzione di interfaccia identica a compute_fic_numba
def compute_fic_numba2(E1, gamma, E0):
    """
    Compute Fic Kernel in fully vectorized form on all axes.
    
    Args:
        E1 : array-like, photon energy after scattering
        gamma : array-like, lorentz factor
        E0 : array-like, photon energy before scattering
    Returns:
        fic : ndarray, shape (len(E1), len(gamma), len(E0))
    """
    E1 = np.asarray(E1, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    E0 = np.asarray(E0, dtype=np.float64)

    # Creo una meshgrid virtuale di tutte le combinazioni
    E1_grid, gamma_grid, E0_grid = np.meshgrid(E1, gamma, E0, indexing='ij')

    # Chiamata vettorizzata su tutti gli elementi simultaneamente
    fic = Fic_element_vec(E1_grid, gamma_grid, E0_grid)

    return fic


class InverseCompton(BaseElectron):
    """Inverse Compton emission from an electron population.

    If you use this class in your research, please consult and cite
    `Khangulyan, D., Aharonian, F.A., & Kelner, S.R.  2014, Astrophysical
    Journal, 783, 100 <http://adsabs.harvard.edu/abs/2014ApJ...783..100K>`_

    Parameters
    ----------
    particle_distribution : function
        Particle distribution function, taking electron energies as a
        `~astropy.units.Quantity` array or float, and returning the particle
        energy density in units of number of electrons per unit energy as a
        `~astropy.units.Quantity` array or float.

    seed_photon_fields : string or iterable of strings (optional)
        A list of gray-body or non-thermal seed photon fields to use for IC
        calculation. Each of the items of the iterable can be either:

        * A string equal to radiation fields:
        ``CMB`` (default, Cosmic Microwave Background),2.72 K, energy densitiy of 0.261 eV/cm³
        ``NIR`` (Near Infrared Radiation),  30 K, energy densitiy 0.5 eV/cm³
        ``FIR`` (Far Infrared Radiation), 3000 K,energy densitiy 1 eV/cm³
        (these are the GALPROP values for a location at a distance of 6.5 kpc from the galactic center).

        * A list of length three (isotropic source) or four (anisotropic
          source) composed of:

            1. A name for the seed photon field.
            2. Its temperature (thermal source) or energy (monochromatic or
               non-thermal source) as a :class:`~astropy.units.Quantity`
               instance.
            3. Its photon field energy density as a
               :class:`~astropy.units.Quantity` instance.
            4. Optional: The angle between the seed photon direction and the
               scattered photon direction as a :class:`~astropy.units.Quantity`
               float instance.

    Other parameters
    ----------------
    Eemin : :class:`~astropy.units.Quantity` float instance, optional
        Minimum electron energy for the electron distribution. Default is 1
        GeV.

    Eemax : :class:`~astropy.units.Quantity` float instance, optional
        Maximum electron energy for the electron distribution. Default is 510
        TeV.

    nEed : scalar
        Number of points per decade in energy for the electron energy and
        distribution arrays. Default is 100.
    """

    def __init__(self, particle_distribution, seed_photon_fields=["CMB"], **kwargs):
        super().__init__(particle_distribution)
        self.seed_photon_fields = self._process_input_seed(seed_photon_fields)
        self.Eemin = 1e9 * u.eV
        self.Eemax = 5.11*1e14*u.eV # the same of 1e9 * mec2
        self.nEed = 100
        self.param_names += ["seed_photon_fields"]
        self.__dict__.update(**kwargs)
        

    @staticmethod
    def _process_input_seed(seed_photon_fields):
        """
        Process seed_photon_fields input and return a standardized OrderedDict.
        """

        #log.info("_process input seed...")
        #start=time.time()
        result = OrderedDict()

        for inseed in seed_photon_fields:
            seed = {}

            # Caso: definizione avanzata come lista [name, T, u, theta?]
            if isinstance(inseed, list) and len(inseed) in (3, 4):

                name, energy, uu = inseed[:3]
                theta = inseed[3] if len(inseed) == 4 else None
                isotropic = theta is None

                #energy = validate_array(f"{name}-energy", u.Quantity(T).flatten(), domain="positive", physical_type="energy")
                energy=check_e_in_eV(energy)
                density = u.Quantity(uu).flatten()
                
                if density.unit.physical_type == "pressure":
                    
                    density /= energy**2
                density = validate_array(f"{name}-density", density, domain="positive")

                seed.update({
                    "type": "array",
                    "energy": energy,
                    "photon_density": density,
                    "isotropic": isotropic
                })

                if not isotropic:
                    seed["theta"] = validate_scalar(f"{name}-theta", theta, physical_type="angle")

            else:
                raise TypeError(f"Unable to process seed photon field: {inseed}")

            result[name] = seed
        
        #log.info(f"_process_input_seed time: {time.time() - start:.5f} s")
        
        #print(result)
        return result
    

    # Funzione principale
    @staticmethod
    def _iso_ic_on_monochromatic(electron_energy, seed_energy, seed_edensity, photon_energy):

        """
        electron_energy: electron energy array
        seed_energy: seed photon energy array or monocromatic
        seed_edensity: seed photon spectral density
        gamma_energy: final photons energy array
        
        """
        #print("")
        log.info("_iso_ic_on_monochromatic ...")
        start = time.time()

        # Assicura array 1D
        electron_energy = np.atleast_1d(electron_energy)
        seed_energy     = np.atleast_1d(seed_energy)
        seed_edensity   = np.atleast_1d(seed_edensity)
        photon_energy = np.asarray(photon_energy) # shape (..., Ng)


        #seed_E_mc2 = seed_energy / mec2
        
        '''start3 = time.time()
        kernel = compute_fic_numba(gamma_energy, electron_energy, photE0)
        print(f"Tempo impiegato fic_numba : {time.time() - start3:.5f} s")

        start3 = time.time()
        kernel = compute_fic_numba_flat(gamma_energy, electron_energy, photE0)
        print(f"Tempo impiegato fic_numba flat : {time.time() - start3:.5f} s")'''

        start4 = time.perf_counter()
        kernel = compute_fic_numba(photon_energy,electron_energy,seed_energy)
        log.info(f"Time for fic_numba : {time.perf_counter() - start4:.5f} s")

        start4 = time.perf_counter()
        kernel = compute_fic_numba2(photon_energy,electron_energy,seed_energy)
        log.info(f"Time for fic_numba2 : {time.perf_counter() - start4:.5f} s")


        # ---- Normalizzazione e fattori fisici ----
        elec = electron_energy[None, :, None]  # broadcasting per norm
        sigt = 6.652458734983284e-25   # cm^2
        c = 29979245800.0  # cm/s
        norm = (3.0/4.0) * sigt * c / (elec**2)
        gamint = kernel * norm

        # ---- Integrazione sul seed photon field ----
        if seed_edensity.size > 1:
            factor = 5.3668543576e-34
            density = seed_edensity * factor
            start1=time.time()
            print(density[None, None, :].shape)
            print(seed_energy[None, None, :].shape)
            print(gamint.shape)
            IC_kernel = simpson_logspace_nb(gamint * density[None, None, :] / seed_energy[None, None, :], seed_energy)
            log.info(f"Time for per gamint = simpson_logspace_nb() : {time.time() - start1:.5f} s")


            #print("IC_kernel.shape:",IC_kernel.shape)
            #print("")
        else:
            factor = 5.3668543576e-34
            density = seed_edensity * factor
            gamint *=density / seed_energy**2
            IC_kernel = gamint.squeeze()
      
        log.info(f"Time for _iso_ic_on_monochromatic : {time.time() - start:.3f} s")
        #print("")
        return IC_kernel

    '''@njit(parallel=True, cache=True)
    def _iso_ic_on_monochromatic_parallel(electron_energy, seed_energy, seed_edensity, gamma_energy):
        
        electron_energy = np.atleast_1d(electron_energy)
        seed_energy = np.atleast_1d(seed_energy)
        seed_edensity = np.atleast_1d(seed_edensity)
        
        photE0 = seed_energy
        phn = seed_edensity
        gamma_energy = np.asarray(gamma_energy)

        kernel = compute_fic_numba(gamma_energy, electron_energy, photE0)
        elec_ = electron_energy[None, :, None]
        sigt = 6.652458734983284e-25
        c = 29979245800.0
        norm = (3.0/4.0) * sigt * c / (elec_**2)
        gamint = kernel * norm

        # integrazione parallela su gamma_energy (ultima dimensione batch)
        IC_kernel = np.empty((gamint.shape[0], gamint.shape[1], photE0.shape[0]))
        for i in prange(gamint.shape[0]):
            for j in prange(gamint.shape[1]):
                IC_kernel[i, j, :] = simpson_logspace_nb(gamint[i, j, :] * phn / photE0, photE0)

        return IC_kernel'''
    
    def _calc_specic(self, seed, Eph1):
        
        #log.info("IC _calc_specic...")
        #start=time.time()
        
        Eph1_mec2 = Eph1/mec2_eV
        
        Eph0_mec2=self.seed_photon_fields[seed]["energy"]/mec2_eV

        start=time.time()
        IC_kernel = self._iso_ic_on_monochromatic(
            self._gamma,                                     #gamma of electrons
            Eph0_mec2,                                       #seed photons energy (0)
            self.seed_photon_fields[seed]["photon_density"], #seed photon density
            Eph1_mec2,                                            #outgoing photon energy (1)
        ) 
        log.info(f"Time for _iso_ic_on_monochromatic : {time.time() - start:.3f} s")

        #start1=time.time()
        #lum = Eph * trapz_loglog(self._nelec * IC_kernel, self._gamma)
        #print(f"Tempo impiegato per lum = trapz_loglog() : {time.time() - start1:.3f} s")

        #print(f"nelec.shape:{self._nelec.shape}")
        #print(f"ICkernel.shape:{IC_kernel.shape}")
        #print(f" self._gamma.shape:{ self._gamma.shape}")
        #start1=time.time()
        #lum =  Eph * trapz_loglog_nd_fast(self._nelec * IC_kernel, self._gamma)
        #log.info(f"Time for lum = trapz_loglog numba() : {time.time() - start1:.5f} s")

        start1=time.time()
        lum =   Eph1_mec2 * simpson_logspace(self._nelec * IC_kernel, self._gamma)
        log.info(f"Time for lum = simpson_logspace() : {time.time() - start1:.5f} s")

        start1=time.time()
        lum =   Eph1_mec2 * simpson_logspace_nb(self._nelec * IC_kernel, self._gamma)
        log.info(f"Time for lum = simpson_logspace_nb() : {time.time() - start1:.5f} s")

        specic=lum/Eph1 # dN/(dt dEgamma)

        #log.info(f"Time for _calc_specic : {time.time() - start:.5f} s")
        return specic  # return differential spectrum in 1/s/eV
    
    def _spectrum(self, photon_energy):
        """Compute differential IC spectrum for energies in ``photon_energy``.

        Compute IC spectrum using IC cross-section for isotropic interaction
        with a blackbody photon spectrum following Khangulyan, Aharonian, and
        Kelner 2014, ApJ 783, 100 (`arXiv:1310.7971
        <http://www.arxiv.org/abs/1310.7971>`_).

        Parameters
        ----------
        photon_energy : :class:`~astropy.units.Quantity` instance
            Photon energy array.
        """
        log.info("IC _spectrum")
        start = time.time()

        
        photon_energy=check_e_in_eV(photon_energy)

        self.specic = []

        for seed in self.seed_photon_fields:
            lum = self._calc_specic(seed, photon_energy)
            self.specic.append(lum)


        result = np.sum(self.specic,axis=0)
        #result = result * (1 / (u.s * u.eV))   # ora result è Quantity con unità 1/(s eV)

        log.info(f"Time for IC _spectrum : {time.time() - start:.5f} s")
        return result
        
    
    
    def flux(self, photon_energy, distance=1 * u.kpc, seed=None):
        """Differential flux at a given distance from the source from a single
        seed photon field

        Parameters
        ----------
        photon_energy : :class:`~astropy.units.Quantity` float or array
            Photon energy array.

        distance : :class:`~astropy.units.Quantity` float, optional
            Distance to the source. If set to 0, the intrinsic luminosity will
            be returned. Default is 1 kpc.

        seed : int, str or None
            Number or name of seed photon field for which the IC contribution
            is required. If set to None it will return the sum of all
            contributions (default).
        """
        
        
        if seed is None:
            flux = super().flux(photon_energy, distance=distance)
        else:

            if distance != 0:
                dfac = 4 * np.pi * distance.to("cm").value ** 2
                #out_unit = 1 / (u.s * u.cm**2 * u.eV)
            else:
                dfac = 1
                #out_unit = 1 / (u.s * u.eV)

            flux = (self.specic[seed] / dfac)#*out_unit
        return flux 

    

    def sed(self, photon_energy, distance=1 * u.kpc, seed=None):
        """Spectral energy distribution at a given distance from the source

        Parameters
        ----------
        photon_energy : :class:`~astropy.units.Quantity` float or array
            Photon energy array.

        distance : :class:`~astropy.units.Quantity` float, optional
            Distance to the source. If set to 0, the intrinsic luminosity will
            be returned. Default is 1 kpc.

        seed : int, str or None
            Number or name of seed photon field for which the IC contribution
            is required. If set to None it will return the sum of all
            contributions (default).
        """
        log.info("")
        log.info("IC SED...")
        
        photon_energy_adim=check_e_in_eV(photon_energy)

        if seed is None:
            # total SED
            sed = super().sed(photon_energy, distance=distance)
        else:
            # SED di un seed specifico
            #out_unit = u.erg/(u.cm**2*u.s) if distance != 0 else u.erg/u.s
            sed = self.flux(photon_energy, distance=distance, seed=seed,check_units=False)*(photon_energy_adim**2.0) * eV_to_erg

        return sed
    


