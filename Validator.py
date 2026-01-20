# from astrofrog/sedfitter
# Copyright (c) 2013-14, Thomas P. Robitaille

import numpy as np
from astropy import units as u


def validate_physical_type(name, value, physical_type):
    """_summary_

    Args:
        name (_type_): _description_
        value (_type_): _description_
        physical_type (_type_): _description_

    Raises:
        TypeError: _description_
        TypeError: _description_
        TypeError: _description_
    """
    if physical_type is not None:
        if not isinstance(value, u.Quantity):
            raise TypeError("{0} should be given as a Quantity object".format(name))
        if isinstance(physical_type, str):
            if value.unit.physical_type != physical_type:
                raise TypeError(
                    "{0} should be given in units of {1}".format(name, physical_type)
                )
        else:
            if value.unit.physical_type not in physical_type:
                raise TypeError(
                    "{0} should be given in units of {1}".format(
                        name, ", ".join(physical_type)
                    )
                )
########################################### NEW  #################################################       

def check_e_in_eV(e):
    if isinstance(e, u.Quantity):
        if not e.unit.is_equivalent(u.eV):
            raise ValueError("The energy must be in units of eV")
        # Se vuoi, puoi anche restituire solo il valore numerico
        return np.asanyarray(e.to_value(u.eV))
    else:
        raise TypeError("e must be a Quantity in units of eV")
    

def check_B_in_Gauss(B):
    if isinstance(B, u.Quantity):
        if not B.unit.is_equivalent(u.G):
            raise ValueError("The B field must be in units of G")
        # Se vuoi, puoi anche restituire solo il valore numerico
        return np.asanyarray(B.to_value(u.G))
    else:
        raise TypeError("B must be a Quantity in units of G")

########################################### NEW  #################################################  
def validate_scalar(name, value, domain=None, physical_type=None):
    validate_physical_type(name, value, physical_type)

    if not physical_type:
        if not np.isscalar(value) or not np.isreal(value):
            raise TypeError("{0} should be a scalar floating point value".format(name))

    if domain == "positive":
        if value < 0.0:
            raise ValueError("{0} should be positive".format(name))
    elif domain == "strictly-positive":
        if value <= 0.0:
            raise ValueError("{0} should be strictly positive".format(name))
    elif domain == "negative":
        if value > 0.0:
            raise ValueError("{0} should be negative".format(name))
    elif domain == "strictly-negative":
        if value >= 0.0:
            raise ValueError("{0} should be strictly negative".format(name))
    elif type(domain) in [tuple, list] and len(domain) == 2:
        if value < domain[0] or value > domain[-1]:
            raise ValueError(
                "{0} should be in the range [{1}:{2}]".format(
                    name, domain[0], domain[-1]
                )
            )

    return value


def validate_array(name, value, domain=None, ndim=1, shape=None, physical_type=None):
    validate_physical_type(name, value, physical_type)

    # First convert to a Numpy array:
    if type(value) in [list, tuple]:
        value = np.array(value)

    # Check the value is an array with the right number of dimensions
    if not isinstance(value, np.ndarray) or value.ndim != ndim:
        if ndim == 1:
            raise TypeError("{0} should be a 1-d sequence".format(name))
        else:
            raise TypeError("{0} should be a {1:d}-d array".format(name, ndim))

    # Check that the shape matches that expected
    if shape is not None and value.shape != shape:
        if ndim == 1:
            raise ValueError(
                "{0} has incorrect length (expected {1} but found {2})".format(
                    name, shape[0], value.shape[0]
                )
            )
        else:
            raise ValueError(
                "{0} has incorrect shape (expected {1} but found {2})".format(
                    name, shape, value.shape
                )
            )

    return value