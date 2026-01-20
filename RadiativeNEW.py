# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import os
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
from Utils import trapz_loglog,trapz_loglog_nd_fast,simpson_uniform,simpson_logspace,trapz_loglog_nd

from scipy.special import cbrt

import time
from numba import njit,prange

#######################################################################################################
__all__ = [
    "Synchrotron",
    "InverseCompton"
]

# Get a new logger to avoid changing the level of the astropy logger
log = logging.getLogger("Radiative")
log.setLevel(logging.INFO)

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
        #x_s = np.logspace(log_a, log_b, N_S)

    else:
        lin_range=a-b
        h_T = (lin_range) / (N - 1)
        h_S = h_T ** 0.5
        N_S = int(np.round((lin_range) / h_S)) + 1
        #x_s = np.linspace(a, b, N_S)
        #print(N_S)
    return N_S

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
        #print("Flux start")
        #print("Photon energy.shape",photon_energy.shape)
        flux = self._spectrum(photon_energy)
        #print("flux.shape",flux.shape)

        if distance != 0:
            distance = validate_scalar("distance", distance, physical_type="length")
            flux =flux/ (4 * np.pi * distance.to("cm") ** 2)
            out_unit = "1/(s cm2 eV)"
            #print("flux unit:", flux.unit)
            #print("expected unit:", out_unit)
 
        else:
            out_unit = "1/(s eV)"
            #print("flux unit:", flux.unit)
            #print("expected unit:", out_unit)

        #print("Flux end")
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
        #print("Sed start")
        if distance != 0:
            out_unit = "erg/(cm2 s)"
        else:
            out_unit = "erg/s"

        photon_energy = _validate_ene(photon_energy)

        sed = (self.flux(photon_energy, distance) * photon_energy**2.0).to(out_unit)
        #print("Sed.shape",sed.shape)
        #print("Sed end")
        return sed
    
    
    
class BaseElectron(BaseRadiative):
    """Implements gamma and nelec properties"""

    def __init__(self, particle_distribution):
        super().__init__(particle_distribution)
        self.param_names = ["Eemin", "Eemax", "nEed"]

    @property
    def _gamma(self):
        """Lorentz factor array"""
        
        Eemin=(self.Eemin / mec2).value
        Eemax=(self.Eemax / mec2).value
        log10gmin = np.log10(Eemin)
        log10gmax = np.log10(Eemax)

        #N=int(np.maximum(10, self.nEed * (log10gmax - log10gmin)))
        N=max(10, int(self.nEed * (log10gmax - log10gmin)))
        simpson=False

        if simpson:
            N_s=simps_points(Eemin,Eemax,N,loglog=True)
            print(f"N_S:{N_s}")
            return np.logspace(log10gmin, log10gmax,N_s)
        else:
            #print(f"N:{N}")
            return np.logspace(log10gmin, log10gmax,N)
        #return np.logspace(log10gmin, log10gmax, max(10, int(self.nEed * (log10gmax - log10gmin)))) OLD

    @property
    def _nelec(self):
        """Particles per unit lorentz factor"""
        pd = self.particle_distribution(self._gamma* mec2)
        print(f"Particle distribution shape: {pd.shape}")
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
@njit    
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
    
    def __init__(self, particle_distribution, B=3.24e-6 * u.G,nEed=100, **kwargs):
        super().__init__(particle_distribution)
        self.B = validate_scalar("B", B, physical_type="magnetic flux density")
        self.Eemin = 1 * u.GeV# (1 * u.GeV).to(u.erg)  # old # 1 * u.GeV
        self.Eemax = 1e9 * mec2
        self.nEed = nEed
        self.param_names += ["B"]
        self.__dict__.update(**kwargs)
    

        
    def _spectrum(self, photon_energy):
        """Compute intrinsic synchrotron differential spectrum for energies in
        ``photon_energy``

        Compute synchrotron for random magnetic field according to
        approximation of Aharonian, Kelner, and Prosekin 2010, PhysRev D 82,
        3002 (`arXiv:1006.1045 <http://arxiv.org/abs/1006.1045>`_)."""
        
        #print("Spectrum start")
        validated_energy = _validate_ene(photon_energy)
        energies = validated_energy.to("erg").value

        #print("_gamma.shape:",self._gamma.shape)
        #print("Energies.shape:",energies.shape)

        Num= np.sqrt(3) * e.value**3 * self.B.to("G").value
        Den = 2 * np.pi * m_e.cgs.value* c.cgs.value**2* hbar.cgs.value* energies
        factor=Num/Den
        
        # Critical energy in erg 
        Ec = (3 * e.value * hbar.cgs.value * self.B.to("G").value * self._gamma**2)/ (2 * (m_e * c).cgs.value)    # Broadcast: photon_energy 2D / Ec 1D
        
        EgEc = energies[..., np.newaxis] / Ec[np.newaxis, np.newaxis, :]  # shape: (theta, phi, gamma)
        
        #print("EgEc.shape:",EgEc.shape)
        #EgEc=validated_energy.to("erg").value/np.vstack(Ec) # 1D version
        #dNdEdt = factor * Gtilde(EgEc)                      # 1D version
        dNdEdt = factor[..., np.newaxis] * Gtilde(EgEc)  # shape (theta, phi, gamma)
        
        #print("dNdEdt.shape:",dNdEdt.shape)
        #spectrum = (trapz_loglog(np.vstack(self._nelec) * dNdEdt, self._gamma, axis=0) / u.s / u.erg ) # 1D version

        # Elettroni: _nelec deve avere dimensione gamma
        nelec = np.array(self._nelec)  # shape (gamma,)
        #print("nelec.shape:",nelec.shape)
        print("nelec*dNdEdt.shape:",(nelec * dNdEdt).shape)
        print("gamma.shape:",self._gamma.shape)
        spectrum = trapz_loglog(nelec * dNdEdt, self._gamma, axis=-1) / u.s / u.erg
        
        #print("spectrum.shape:",spectrum.shape)
        spectrum = spectrum.to("1/(s eV)")

        #print(" Spectrum End")
        
        return spectrum
@njit
def G12(x, param):
    """
    Eqs 18,19,20 of Khangulyan et al (2014)
    """
    alpha, a, beta, b = param
    G0 = (np.pi**2 / 6.0 + x) * np.exp(-x)
    tmp = 1 + b * x**beta
    g = 1.0 / (a * x**alpha / tmp + 1.0)
    return G0 * g

@njit
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
    return G0 * g   
 
@njit
def fic_element(g, e, p):
        eps=1e-30   
        b = 4 * p * e
        w = g / e
        q = w / ((b * (1 - w))+ eps)
        
        if 1.0/(4*e**2) < q < 1.0:
            return 2*q*np.log(q) + (1 + 2*q)*(1 - q) + 0.5*(b*q)**2*(1 - q)/(1 + b*q)
        else:
            return 0.0

@njit
def compute_fic_numba(gamma, electron, phot):
    
    
    Ng_shape = gamma.shape
    Ne = electron.shape[0]
    Np = phot.shape[0]

    # Array di output: gamma_dim..., Ne, Np
    fic = np.zeros(Ng_shape + (Ne, Np))

    for idx in np.ndindex(Ng_shape):
        for k in range(Ne):
            for l in range(Np):
                fic[idx + (k,l)] = fic_element(gamma[idx], electron[k], phot[l])
                

    return fic

@njit(parallel=True)
def compute_fic_integrated(gamma, electron, phot):
    Ng = gamma.shape[0]
    Ne = electron.shape[0]
    Np = phot.shape[0]
    fic = np.zeros(Ng)
    for i in prange(Ng):
        s = 0.0
        for k in range(Ne):
            for l in range(Np):
                s += fic_element(gamma[i], electron[k], phot[l])
        fic[i] = s
    return fic



"""@njit
def compute_fic_numba_parallel(gamma, electron, phot):
    Ng_shape = gamma.shape
    Ne = electron.shape[0]
    Np = phot.shape[0]
    fic = np.zeros(Ng_shape + (Ne, Np))

    for idx in prange(np.prod(Ng_shape)):  # loop "flattened"
        gamma_idx = np.unravel_index(idx, Ng_shape)
        for k in range(Ne):
            for l in range(Np):
                fic[gamma_idx + (k,l)] = fic_element(gamma[gamma_idx], electron[k], phot[l])
    
    return fic"""

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
        # Definizioni predefinite per i campi noti
        known_fields = {
            "CMB": {"T": 2.72548 * u.K, "u": None},
            "FIR": {"T": 30 * u.K, "u": 0.5 * u.eV / u.cm**3},
            "NIR": {"T": 3000 * u.K, "u": 1.0 * u.eV / u.cm**3},
        }

        if not isinstance(seed_photon_fields, list):
            seed_photon_fields = seed_photon_fields.split("-")

        result = OrderedDict()

        for inseed in seed_photon_fields:
            seed = {}

            # Caso: stringa nota (CMB, FIR, NIR)
            if isinstance(inseed, str):
                name = inseed
                if name not in known_fields:
                    raise ValueError(f"Unknown seed field: {name}")
                T = known_fields[name]["T"]
                u_val = known_fields[name]["u"] or (ar * T**4)
                seed.update({
                    "type": "thermal",
                    "T": T,
                    "u": u_val,
                    "isotropic": True
                })

            # Caso: definizione avanzata come lista [name, T, u, theta?]
            elif isinstance(inseed, list) and len(inseed) in (3, 4):
                name, T, uu = inseed[:3]
                theta = inseed[3] if len(inseed) == 4 else None
                isotropic = theta is None

                thermal = T.unit.physical_type == "temperature"

                if thermal:
                    validate_scalar(f"{name}-T", T, domain="positive", physical_type="temperature")
                    u_val = ar * T**4 if uu == 0 else validate_scalar(f"{name}-u", uu, domain="positive", physical_type="pressure")
                    seed.update({
                        "type": "thermal",
                        "T": T,
                        "u": u_val,
                        "isotropic": isotropic
                    })
                else:
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

        return result

            
    @staticmethod
    def _iso_ic_on_planck(electron_energy, soft_photon_temperature, gamma_energy):
        """
        IC cross-section for isotropic interaction with a blackbody photon
        spectrum following Eq. 14 of Khangulyan, Aharonian, and Kelner 2014,
        ApJ 783, 100 (`arXiv:1310.7971 <http://www.arxiv.org/abs/1310.7971>`_).

        `electron_energy` and `gamma_energy` are in units of m_ec^2
        `soft_photon_temperature` is in units of K
        """
        Ktomec2 = 1.6863699549e-10   # conversion factor form Kelvin to m-ec^2 units
        soft_photon_temperature *= Ktomec2  # convert the temperature

        gamma_energy = gamma_energy[:, None]
        # Parameters from Eqs 26, 27
        a3 = [0.606, 0.443, 1.481, 0.540, 0.319]
        a4 = [0.461, 0.726, 1.457, 0.382, 6.620]
        
        # gamma_energy is the upscattered photon energy 
        z = gamma_energy / electron_energy
        x = z / (1 - z) / (4.0 * electron_energy * soft_photon_temperature)
        # Eq. 14
        cross_section = z**2 / (2 * (1 - z)) * G34(x, a3) + G34(x, a4)
        tmp = (soft_photon_temperature / electron_energy) ** 2
        
        # r0 = (e**2 / m_e / c**2).to('cm')
        # (2 * r0 ** 2 * m_e ** 3 * c ** 4 / (pi * hbar ** 3)).cgs
        tmp *= 2.6318735743809104e16 
        cross_section = tmp * cross_section
        # gamma energy less then the production elctron energy and a relativistiv elctron
        validity_condition= (gamma_energy < electron_energy) * (electron_energy > 1)  
        return np.where(validity_condition, cross_section, np.zeros_like(cross_section))


    @staticmethod
    def _ani_ic_on_planck(electron_energy, soft_photon_temperature, gamma_energy, theta):
        """
        IC cross-section for anisotropic interaction with a blackbody photon
        spectrum following Eq. 11 of Khangulyan, Aharonian, and Kelner 2014,
        ApJ 783, 100 (`arXiv:1310.7971 <http://www.arxiv.org/abs/1310.7971>`_).

        `electron_energy` and `gamma_energy` are in units of m_ec^2
        `soft_photon_temperature` is in units of K
        `theta` is in radians
        """
        Ktomec2 = 1.6863699549e-10         # conversion factor form Kelvin to m-ec^2 units
        soft_photon_temperature *= Ktomec2 # convert the temperature

        gamma_energy = gamma_energy[:, None]
        # Parameters from Eqs 21, 22
        a1 = [0.857, 0.153, 1.840, 0.254]
        a2 = [0.691, 1.330, 1.668, 0.534]
        z = gamma_energy / electron_energy
        ttheta = 2.0 * electron_energy * soft_photon_temperature * (1.0 - np.cos(theta))
        x = z / (1 - z) / ttheta
        # Eq. 11
        cross_section = z**2 / (2 * (1 - z)) * G12(x, a1) + G12(x, a2)
        tmp = (soft_photon_temperature / electron_energy) ** 2
        
        # r0 = (e**2 / m_e / c**2).to('cm')
        # (2 * r0 ** 2 * m_e ** 3 * c ** 4 / (pi * hbar ** 3)).cgs
        tmp *= 2.6318735743809104e16       
        cross_section = tmp * cross_section
        # gamma energy less then the production elctron energy and a relativistiv elctron
        validity_conditon = (gamma_energy < electron_energy) * (electron_energy > 1)
        return np.where(validity_conditon, cross_section, np.zeros_like(cross_section))

    


    
    @staticmethod
    def _iso_ic_on_monochromatic(electron_energy, seed_energy, seed_edensity, gamma_energy):
        start = time.time()

        # Assicura array 1D
        electron_energy = np.atleast_1d(electron_energy)
        seed_energy     = np.atleast_1d(seed_energy)
        seed_edensity   = np.atleast_1d(seed_edensity)

        photE0 = (seed_energy / mec2).decompose().value
        phn = seed_edensity
        gamma_energy = np.asarray(gamma_energy)

        # ---- Kernel integrato (numba parallelo) ----
        #start2 = time.time()
        fic = compute_fic_numba(gamma_energy, electron_energy, photE0)
        #print(f"Tempo compute_fic_integrated: {time.time() - start2:.3f} s")

        # ---- Normalizzazione ----
        sigt = 6.652458734983284e-25   # cm^2
        c = 29979245800.0  # cm/s
        norm = (3.0/4.0) * sigt * c / np.mean(electron_energy**2)
        gamint = fic * norm

        # ---- Peso fotoni seed ----
        if phn.size > 1:
            phn_ = phn.to(1 / (mec2_unit * u.cm**3)).value
            #gamint *= np.trapz(phn_ / photE0, photE0)
            start1 = time.time()
            #gamint = trapz_loglog(gamint * phn_[None, None, :] / photE0[None, None, :], photE0, axis=-1)
            gamint = trapz_loglog_nd(gamint * phn_[None, None, :] / photE0[None, None, :], photE0)
            print(f"Tempo impiegato per gamint = trapz_loglog() : {time.time() - start1:.3f} s")
        else:
            #phn_ = phn.to(mec2_unit / u.cm**3).value
            gamint *= phn_ / photE0**2
            gamint = gamint.squeeze()

        print(f"Tempo totale _iso_ic_on_monochromatic: {time.time() - start:.3f} s")
        return gamint
    
    
    def _calc_specic(self, seed, outspecene):
        start= time.time()
        log.debug("_calc_specic: Computing IC on {0} seed photons...".format(seed))

        Eph = (outspecene / mec2).decompose().value
        # Catch numpy RuntimeWarnings of overflowing exp (which are then
        # discarded anyway)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.seed_photon_fields[seed]["type"] == "thermal":
                T = self.seed_photon_fields[seed]["T"]
                uf = (self.seed_photon_fields[seed]["u"] / (ar * T**4)).decompose()
                
                if self.seed_photon_fields[seed]["isotropic"]:
                    gamint = self._iso_ic_on_planck(self._gamma, T.to("K").value, Eph)
                else:
                    theta = self.seed_photon_fields[seed]["theta"].to("rad").value
                    gamint = self._ani_ic_on_planck(
                        self._gamma, T.to("K").value, Eph, theta
                    )
            else:
                #print("self._gamma.shape:",self._gamma.shape)
                #print("self.seed_photon_fields[seed]['energy'].shape:",self.seed_photon_fields[seed]["energy"].shape)
                #print("self.seed_photon_fields[seed]['photon_density'].shape:",self.seed_photon_fields[seed]["photon_density"].shape)
                #print("Eph.shape:",Eph.shape)
                
                uf = 1
                gamint = self._iso_ic_on_monochromatic(
                    self._gamma,
                    self.seed_photon_fields[seed]["energy"],
                    self.seed_photon_fields[seed]["photon_density"],
                    Eph,
                )
            print("gamma:", self._gamma.shape)
            print("nelec:", self._nelec.shape)
            print("gamint:", gamint.shape)
            lum = uf * Eph * trapz_loglog(self._nelec * gamint, self._gamma)
        lum = lum * u.Unit("1/s")
        #print("lum.shape:",lum.shape)
        #print("lum/outspecene.shape:",(lum / outspecene).shape)
        
        print(f"Tempo impiegato per _calc_specic : {time.time() - start:.3f} s")
        return lum / outspecene  # return differential spectrum in 1/s/eV
    
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
        validated_energy = _validate_ene(photon_energy)

        #original_shape = validated_energy.shape
        #flattened_energy = validated_energy.flatten()

        self.specic = []

        for seed in self.seed_photon_fields:
            # Call actual computation, detached to allow changes in subclasses
            self.specic.append(self._calc_specic(seed, validated_energy).to("1/(s eV)"))

            #lum_flat = self._calc_specic(seed, flattened_energy).to("1/(s eV)")
            #self.specic.append(lum_flat)

        #total_flat = np.sum(self.specic, axis=0)
        #total = total_flat.reshape(original_shape) * u.Unit("1/(s eV)")

        return np.sum(u.Quantity(self.specic), axis=0)
        #return total
    
    
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
        model = super().flux(photon_energy, distance=distance)

        if seed is not None:
            # Test seed argument
            if not isinstance(seed, int):
                if seed not in self.seed_photon_fields:
                    raise ValueError(
                        "Provided seed photon field name is not in"
                        " the definition of the InverseCompton instance"
                    )
                else:
                    seed = list(self.seed_photon_fields.keys()).index(seed)
            elif seed > len(self.seed_photon_fields):
                raise ValueError(
                    "Provided seed photon field number is larger"
                    " than the number of seed photon fields defined in the"
                    " InverseCompton instance"
                )

            if distance != 0:
                distance = validate_scalar("distance", distance, physical_type="length")
                dfac = 4 * np.pi * distance.to("cm") ** 2
                out_unit = "1/(s cm2 eV)"
            else:
                dfac = 1
                out_unit = "1/(s eV)"

            model = (self.specic[seed] / dfac).to(out_unit)

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
        sed = super().sed(photon_energy, distance=distance)

        if seed is not None:
            if distance != 0:
                out_unit = "erg/(cm2 s)"
            else:
                out_unit = "erg/s"

            sed = (
                self.flux(photon_energy, distance=distance, seed=seed)
                * photon_energy**2.0
            ).to(out_unit)

        return sed