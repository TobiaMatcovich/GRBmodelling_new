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
)
#from .model_utils import memoize
from Utils import trapz_loglog,trapz_loglog_nd,trapz_loglog_nd_fast
from Utils import simpson_logspace,simpson_logspace_nb

from scipy.special import cbrt

import time
from numba import njit,vectorize,prange,guvectorize, float64

#######################################################################################################
__all__ = [
    "Synchrotron",
    "InverseCompton"
]

e = e.gauss
mec2 = (m_e * c**2).cgs
mec2_unit = u.Unit(mec2)

ar = (4 * sigma_sb / c).to("erg/(cm3 K4)")  # costante di radiazione
r0 = (e**2 / mec2).to("cm")  #raggio classico dell'elettrone 

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
        try:
            # Check first for the amplitude attribute, which will be present if
            # the particle distribution is a function from naima.models
            pd = self.particle_distribution.amplitude
            validate_physical_type(
                "Particle distribution",
                pd,
                physical_type="differential energy",
            )
        except (AttributeError, TypeError):
            # otherwise check the output
            pd = self.particle_distribution([0.1, 1, 10] * u.TeV)
            validate_physical_type(
                "Particle distribution",
                pd,
                physical_type="differential energy",
            )

    #  @memoize
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

        #log.info("FLUX ...")
        #start=time.time()
        flux = self._spectrum(photon_energy)#[0]
        #print("flux.shape",flux.shape)

        if distance != 0:
            distance = validate_scalar("distance", distance, physical_type="length")
            flux =flux/ (4 * np.pi * distance.to("cm") ** 2)
            out_unit = "1/(s cm2 eV)"
 
        else:
            out_unit = "1/(s eV)"
        
        #print("FLUX time:",time.time()-start)
        #print("")
        return flux.to(out_unit)

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
        if distance != 0:
            out_unit = "erg/(cm2 s)"
        else:
            out_unit = "erg/s"

        photon_energy = _validate_ene(photon_energy)

        sed = (self.flux(photon_energy, distance) * photon_energy**2.0).to(out_unit)
        #print("Sed.shape",sed.shape)
        #print("SED time:",time.time()-start)
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
        
        #print("")
        #print("_gamma computation")
        #start = time.time()

        log10gmin = np.log10(self.Eemin / mec2).value
        log10gmax = np.log10(self.Eemax / mec2).value
        N = int(np.maximum(10, self.nEed * (log10gmax - log10gmin)))
        
        simpson = False
        if simpson:
            points = simps_points(10**(log10gmin), 10**(log10gmax), N, loglog=True)
        else:
            points = np.logspace(log10gmin, log10gmax, N)

        #print(f"Time gamma computation: {time.time() - start:.3f}s")
        #print(f"points: {len(points)}")
        #print("_gamma computation end\n")
 
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
    cb= np.sign(x) * np.abs(x) ** (1.0 / 3.0)
    gt1 = 1.808 * cb / np.sqrt(1 + 3.4 * cb**2.0)
    gt2 = 1 + 2.210 * cb**2.0 + 0.347 * cb**4.0
    gt3 = 1 + 1.353 * cb**2.0 + 0.217 * cb**4.0
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
        self.B = validate_scalar("B", B, physical_type="magnetic flux density")
        self.Eemin = 1 * u.GeV
        self.Eemax = 1e9 * mec2
        self.nEed = 100
        self.param_names += ["B"]
        self.__dict__.update(**kwargs)
    

        
    def _spectrum(self, photon_energy):
        """Compute intrinsic synchrotron differential spectrum for energies in
        ``photon_energy``

        Compute synchrotron for random magnetic field according to
        approximation of Aharonian, Kelner, and Prosekin 2010, PhysRev D 82,
        3002 (`arXiv:1006.1045 <http://arxiv.org/abs/1006.1045>`_)."""
        
        #log.info("SYN Spectrum...")
        #start=time.time()

        validated_energy = _validate_ene(photon_energy)
        energies = validated_energy.to("erg").value
        
        Num= np.sqrt(3) * e.value**3 * self.B.to("G").value
        Den = 2 * np.pi * m_e.cgs.value* c.cgs.value**2* hbar.cgs.value* energies
        factor=Num/Den

        #print(factor[30])
        
        # Critical energy in erg 
        Ec = (3 * e.value * hbar.cgs.value * self.B.to("G").value * self._gamma**2)/ (2 * (m_e * c).cgs.value)    # Broadcast: photon_energy 2D / Ec 1D)
        EgEc = energies[..., np.newaxis] / Ec[np.newaxis, np.newaxis, :]  # shape: (theta, phi, gamma)
                             # 1D version
        
        '''start1=time.time()
        dNdEdt = factor[..., np.newaxis] * Gtilde(EgEc)  # shape (theta, phi, gamma)
        log.info(f"Gtilde:{time.time()-start1:.5f} s")
        
        start1=time.time()
        dNdEdt = factor[..., np.newaxis]*Gtilde(EgEc.ravel()).reshape(EgEc.shape)
        log.info(f"Gtilde2:{time.time()-start1:.5f} s")'''
        
        start1=time.time()
        dNdEdt = factor[..., np.newaxis]*Gtilde_vec(EgEc)
        log.info(f"Gtilde vec:{time.time()-start1:.5f} s")
        
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
        spectrum = simpson_logspace_nb(nelec * dNdEdt, self._gamma) / u.s / u.erg
        log.info(f"SYN simpson logspace nb:{time.time()-start1:.5f} s")

        '''start1=time.time()
        spectrum = simpson_logspace(nelec * dNdEdt, self._gamma) / u.s / u.erg
        log.info(f"SYN simpson logspace time:{time.time()-start1:.5f} s")'''


        spectrum = spectrum.to("1/(s eV)")
        
        #print("scpectrum.shape:",spectrum.shape)
        #log.info(f"_spectrum time:{time.time()-start:.5f} s")
        
        return spectrum #,spectrum1,spectrum2,spectrum3,time_trapz_loglog,time_trapz_loglog_nd,time_trapz_loglog_nd_fast,time_simps_loglog
    
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
 
from numba import float32
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
def compute_fic_numba(E1, gamma, E0):
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

    Ng = E1_flat.shape[0]
    Ne = gamma.shape[0]
    Np = E0.shape[0]

    # array contiguo 3D per massima efficienza
    fic = np.zeros((Ng, Ne, Np),dtype=np.float64)

    # loop esterno parallelo
    for i in prange(Ng):
        g = E1_flat[i]  # carica una volta per tutte
        # loop interni puri → ottimizzabili
        for k in range(Ne):
            e = gamma[k]

            for l in range(Np):
                p = E0[l]
                # kernel IC
                fic[i, k, l] = Fic_element(g, e, p)
    # ricostruisci la forma originale
    return fic.reshape(original_shape + (Ne, Np))




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
        self.Eemin = 1 * u.GeV
        self.Eemax = 1e9 * mec2
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

                name, T, uu = inseed[:3]
                theta = inseed[3] if len(inseed) == 4 else None
                isotropic = theta is None

                energy = validate_array(f"{name}-energy", u.Quantity(T).flatten(), domain="positive", physical_type="energy")
                density = u.Quantity(uu).flatten()
                
                if density.unit.physical_type == "pressure":
                    
                    density /= energy**2
                density = validate_array(f"{name}-density", density, domain="positive", physical_type="differential number density")

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
    def _iso_ic_on_monochromatic(electron_energy, seed_energy, seed_edensity, gamma_energy):

        """
        electron_energy: electron energy array
        seed_energy: seed photon energy array or monocromatic
        seed_edensity: seed photon spectral density
        gamma_energy: final photons energy array
        
        """

        # Assicura array 1D
        electron_energy = np.atleast_1d(electron_energy)
        seed_energy     = np.atleast_1d(seed_energy)
        seed_edensity   = np.atleast_1d(seed_edensity)

        #photE0 = seed_energy / mec2.value
        photE0 = (seed_energy / mec2).decompose().value
        phn = seed_edensity
        #print("phn:",phn)
        #print(f"phn.shape:{phn.shape}")
        gamma_energy = np.asarray(gamma_energy)  # shape (..., Ng)
        
        ## ---- Costruzione array fic con Numba ----
        #log.info(f"gamma_energy.shape:{gamma_energy.shape}")
        #log.info(f"electron_.shape:{electron_energy.shape}")
        #log.info(f"photE0.shape:{photE0.shape}")
        
        start4 = time.perf_counter()
        kernel = compute_fic_numba(gamma_energy, electron_energy, photE0)
        log.info(f"Time for fic_numba : {time.perf_counter() - start4:.5f} s")


        # ---- Normalizzazione e fattori fisici ----
        elec_ = electron_energy[None, :, None]  # broadcasting per norm
        sigt = 6.652458734983284e-25   # cm^2
        c = 29979245800.0  # cm/s
        norm = (3.0/4.0) * sigt * c / (elec_**2)
        gamint = kernel * norm

        #print("norm.shape:",norm.shape)
        #print("Kernel.shape:",kernel.shape)
        #print("")

        # ---- Integrazione sul seed photon field ----
        if phn.size > 1:
            phn_ = phn.to(1 / (mec2_unit * u.cm**3)).value
            #start1 = time.time()
            #IC_kernel = trapz_loglog(gamint * phn_[None, None, :] / photE0[None, None, :], photE0, axis=-1)
            #print(f"Tempo impiegato per gamint = trapz_loglog() : {time.time() - start1:.3f} s")
            #print("gamint.shape:",gamint.shape)

            '''start1=time.time()
            IC_kernel = trapz_loglog_nd(gamint * phn_[None, None, :] / photE0[None, None, :], photE0)
            log.info(f"Time for gamint = trapz_loglog_nd() : {time.time() - start1:.5f} s")'''

            start1=time.time()
            IC_kernel = simpson_logspace_nb(gamint * phn_[None, None, :] / photE0[None, None, :], photE0)
            log.info(f"Time for gamint = simpson_logspace_nb() : {time.time() - start1:.5f} s")


            #print("IC_kernel.shape:",IC_kernel.shape)
            #print("")
        else:
            phn_ = phn.to(mec2_unit / u.cm**3).value
            #phn_=(phn/mec2).value
            gamint *= phn_ / photE0**2
            IC_kernel = gamint.squeeze()
      


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
    
    def _calc_specic(self, seed, outspecene):
        
        """ 
        outspecene= E gamma """
        #log.info("IC _calc_specic...")
        #start=time.time()
        
        Eph = (outspecene / mec2).decompose().value
        
        # Catch numpy RuntimeWarnings of overflowing exp (which are then
        # discarded anyway)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            #print("self._gamma.shape:",self._gamma.shape)
            #print("self.seed_photon_fields[seed]['energy'].shape:",self.seed_photon_fields[seed]["energy"].shape)
            #print("self.seed_photon_fields[seed]['photon_density'].shape:",self.seed_photon_fields[seed]["photon_density"].shape)
            #print("Eph.shape:",Eph.shape)
            #print("")
            start=time.time()
            IC_kernel = self._iso_ic_on_monochromatic(
                self._gamma,      #gamma of electrons
                self.seed_photon_fields[seed]["energy"],   #seed photons energy
                self.seed_photon_fields[seed]["photon_density"], #seed photon density
                Eph,  #outgoing photon energy
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

            '''start1=time.time()
            lum =  Eph * simpson_logspace(self._nelec * IC_kernel, self._gamma)
            log.info(f"Time for lum = simpson_logspace() : {time.time() - start1:.5f} s")'''

            start1=time.time()
            lum =  Eph * simpson_logspace_nb(self._nelec * IC_kernel, self._gamma)
            log.info(f"Time for lum = simpson_logspace_nb() : {time.time() - start1:.5f} s")

            #lum =  Eph * trapz_loglog(self._nelec *IC_kernel, self._gamma)

        #lum = lum * u.Unit("1/s")
        lum = lum / u.s
        specic=lum/outspecene # dN/(dt dEgamma)
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
        start = time.time()
        validated_energy = _validate_ene(photon_energy)
        self.specic = []

        for seed in self.seed_photon_fields:
            lum = self._calc_specic(seed, validated_energy).to("1/(s eV)")
            self.specic.append(lum)


        result = np.sum(u.Quantity(self.specic), axis=0)
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
        #print("")
        #log.info("IC flux")
        #start=time.time()
        model = super().flux(photon_energy, distance=distance)

        # Assicurati che la spectrum sia calcolata
        #if not hasattr(self, "specic"):
        #    _ = self._spectrum(photon_energy)
            
        # Ora gestiamo il seed
        if seed is not None:
            # --- seleziona il seed ---
            if not isinstance(seed, int):
                if seed not in self.seed_photon_fields:
                    raise ValueError("Seed non valido")
                seed = list(self.seed_photon_fields.keys()).index(seed)

            if distance != 0:
                distance = validate_scalar("distance", distance, physical_type="length")
                dfac = 4 * np.pi * distance.to("cm") ** 2
                out_unit = "1/(s cm2 eV)"
            else:
                dfac = 1
                out_unit = "1/(s eV)"

            model = (self.specic[seed] / dfac).to(out_unit)
        #print("flux time:",time.time()-start)
        return model  


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

        if seed is None:
            # total SED
            sed = super().sed(photon_energy, distance=distance)
        else:
            # SED di un seed specifico
            out_unit = "erg/(cm2 s)" if distance != 0 else "erg/s"
            sed = (self.flux(photon_energy, distance=distance, seed=seed)
                        * photon_energy**2.0).to(out_unit)

        return sed