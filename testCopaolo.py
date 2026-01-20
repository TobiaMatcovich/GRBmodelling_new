import ast
import warnings
import math
import astropy.units as u
import numpy as np
from astropy import log
from astropy.table import QTable, Table

from Validator import validate_array, validate_scalar
from numba import njit,prange
import time 

from log_config import setup_logger
log = setup_logger(__name__)  

#from scipy.integrate import simpson
###############################################################
#                                                             #
#                     INTEGRATION                             #
#                                                             #
###############################################################

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


def trapz_loglog(y, x, axis=-1, intervals=False):
    """
    Integrate along the given axis using the composite trapezoidal rule in
    loglog space.

    Integrate `y` (`x`) along given axis in loglog space.

    Parameters
    ----------
    y : array_like
        Input array to integrate.
    x : array_like, optional
        Independent variable to integrate over.
    axis : int, optional
        Specify the axis.

    Returns
    -------
    trapz : float
        Definite integral as approximated by trapezoidal rule in loglog space.
    """
    try:
        y_unit = y.unit
        y = y.value
    except AttributeError:
        y_unit = 1.0
    try:
        x_unit = x.unit
        x = x.value
    except AttributeError:
        x_unit = 1.0

    y = np.asanyarray(y)
    x = np.asanyarray(x)

    slice1 = [slice(None)] * y.ndim
    slice2 = [slice(None)] * y.ndim
    slice1[axis] = slice(None, -1)
    slice2[axis] = slice(1, None)

    slice1 = tuple(slice1)
    slice2 = tuple(slice2)

    if x.ndim == 1:
        shape = [1] * y.ndim
        shape[axis] = x.shape[0]
        x = x.reshape(shape)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Compute the power law indices in each integration bin
        b = np.log10(y[slice2] / y[slice1]) / np.log10(x[slice2] / x[slice1])

        # if local powerlaw index is -1, use \int 1/x = log(x); otherwise use
        # normal powerlaw integration
        trapzs = np.where(
            np.abs(b + 1.0) > 1e-10,
            (y[slice1] * (x[slice2] * (x[slice2] / x[slice1]) ** b - x[slice1]))
            / (b + 1),
            x[slice1] * y[slice1] * np.log(x[slice2] / x[slice1]),
        )

    tozero = (y[slice1] == 0.0) + (y[slice2] == 0.0) + (x[slice1] == x[slice2])
    trapzs[tozero] = 0.0

    if intervals:
        return trapzs * x_unit * y_unit

    ret = np.add.reduce(trapzs, axis) * x_unit * y_unit

    return ret

##################################### TRAPZ ##############################################
@njit(cache=True,fastmath=True)
def trapz_numba(y, x):
    """
    Trapezoidal integration along the last axis.
    y : 2D array
    x : 1D array (same length as last axis of y)
    """
    n = x.shape[0]
    out_shape = y.shape[:-1]
    out = np.zeros(out_shape)

    # Scorri tutti gli indici tranne l’ultimo asse
    for idx in np.ndindex(out_shape):
        s = 0.0
        for i in range(n - 1):
            s += 0.5 * (y[idx + (i+1,)] + y[idx + (i,)]) * (x[i+1] - x[i])
        out[idx] = s

    return out

@njit(cache=True,fastmath=True)
def trapz_loglog_nd(y, x):
    out_shape = y.shape[:-1]
    res = np.zeros(out_shape)

    n = x.shape[0]
    for idx in np.ndindex(out_shape):
        s = 0.0
        for i in range(n - 1):
            x1 = x[i]; x2 = x[i+1]
            y1 = y[idx + (i,)]; y2 = y[idx + (i+1,)]
            if y1 <= 0.0 or y2 <= 0.0 or x1 <= 0.0 or x2 <= 0.0:
                continue
            b = np.log(y2 / y1) / np.log(x2 / x1)
            if np.abs(b + 1.0) > 1e-10:
                s += (y1 * (x2 * (x2 / x1)**b - x1)) / (b + 1.0)
            else:
                s += x1 * y1 * np.log(x2 / x1)
        res[idx] = s

    return res

@njit(cache=True,fastmath=True)
def trapz_loglog_nd_fast(y, x):

    out_shape = y.shape[:-1]
    n = x.shape[0]
    res = np.zeros(out_shape)

    # 🔹 Precalcola i logaritmi di x per evitare ripetizioni costose
    logx = np.log(x)

    # 🔹 Loop su tutti gli indici tranne l'ultimo asse
    for idx in np.ndindex(out_shape):
        s = 0.0
        for i in range(n - 1):
            x1 = x[i]; x2 = x[i+1]
            y1 = y[idx + (i,)]; y2 = y[idx + (i+1,)]

            # Salta intervalli non validi
            if y1 <= 0.0 or y2 <= 0.0 or x1 <= 0.0 or x2 <= 0.0:
                continue

            # 🔹 Calcola b in modo stabile
            b = np.log(y2 / y1) / (logx[i+1] - logx[i])

            # 🔹 Versione senza 'if': aggiungi piccolo offset per evitare divisione per zero
            denom = b + 1.0 + 1e-14
            s += (y1 * (x2 * (x2 / x1)**b - x1)) / denom
        res[idx] = s
    return res

##################################### SIMPSON ##############################################

@njit(cache=True,fastmath=True)
def simpson_uniform(y, x):
    """
    Regola di Simpson per griglie uniformi (corretta anche per N pari).
    """
    n = x.shape[0]
    h = (x[-1] - x[0]) / (n - 1)

    out_shape = y.shape[:-1]
    res = np.zeros(out_shape)

    for idx in np.ndindex(out_shape):
        # se n è pari → usa Simpson 3/8 sugli ultimi 4 punti
        if n % 2 == 0:
            # togli l’ultimo intervallo dalla parte principale
            n_main = n - 3
            s = y[idx + (0,)] + y[idx + (n_main-1,)]
            for i in range(1, n_main-1, 2):
                s += 4.0 * y[idx + (i,)]
            for i in range(2, n_main-1, 2):
                s += 2.0 * y[idx + (i,)]
            I = s * h / 3.0

            # aggiungi Simpson 3/8 sugli ultimi 3 intervalli
            I += (3*h/8.0) * (y[idx + (n-4,)] + 3*y[idx + (n-3,)] +
                            3*y[idx + (n-2,)] + y[idx + (n-1,)])
        else:
            # normale Simpson 1/3
            s = y[idx + (0,)] + y[idx + (n-1,)]
            for i in range(1, n-1, 2):
                s += 4.0 * y[idx + (i,)]
            for i in range(2, n-1, 2):
                s += 2.0 * y[idx + (i,)]
            
            I = s * h / 3.0

        res[idx] = I

    return res


#@njit(cache=True)
def simpson_logspace(y, x):
    """
    Regola di Simpson per griglie uniformi in log10(x),
    corretta anche per N pari (usa 3/8 sugli ultimi 3 intervalli).
    """

    n = x.shape[0]
    # coordinate logaritmiche
    t0 = np.log10(x[0])
    t1 = np.log10(x[-1])
    h = (t1 - t0) / (n - 1)

    out_shape = y.shape[:-1]
    res = np.zeros(out_shape)

    log.info(type(y))
    log.info(type(x))
    log.info("--------")
    log.info(y.shape)
    log.info(x.shape)
    log.info(f"out:{out_shape}")

    #log10=np.log(10.0)

    for idx in np.ndindex(out_shape):
        if n < 3:
            res[idx] = 0.0
            continue

        # funzione trasformata: f_i = y_i * x_i * ln(10)
        # perché dx = x * ln(10) * dt
        if n % 2 == 0:
            # n pari → 1/3 fino a n-4, poi 3/8 finale
            n_main = n - 3
            s = y[idx + (0,)] * x[0] * np.log(10.0) + y[idx + (n_main-1,)] * x[n_main-1] * np.log(10.0)
            for i in range(1, n_main-1, 2):
                s += 4.0 * y[idx + (i,)] * x[i] * np.log(10.0)
            for i in range(2, n_main-1, 2):
                s += 2.0 * y[idx + (i,)] * x[i] * np.log(10.0)
            I = s * h / 3.0

            # Simpson 3/8 sugli ultimi 4 punti
            I += (3*h/8.0) * (
                y[idx + (n-4,)] * x[n-4] * np.log(10.0)
                + 3.0 * y[idx + (n-3,)] * x[n-3] * np.log(10.0)
                + 3.0 * y[idx + (n-2,)] * x[n-2] * np.log(10.0)
                + y[idx + (n-1,)] * x[n-1] * np.log(10.0)
            )

        else:
            # n dispari → normale Simpson 1/3 su tutti i punti
            s = y[idx + (0,)] * x[0] * np.log(10.0) + y[idx + (n-1,)] * x[n-1] * np.log(10.0)
            for i in range(1, n-1, 2):
                s += 4.0 * y[idx + (i,)] * x[i] * np.log(10.0)
            for i in range(2, n-1, 2):
                s += 2.0 * y[idx + (i,)] * x[i] * np.log(10.0)
            I = s * h / 3.0

        res[idx] = I

    return res

#@njit(cache=True, parallel=True, fastmath=True)
def simpson_logspace_nb2(y, x):

    n = x.shape[0]

    t0 = np.log10(x[0])
    t1 = np.log10(x[-1])
    h = (t1 - t0) / (n - 1)

    shape = y.shape
    B = y.size // n
    y2 = y.reshape(B, n)

    out = np.empty(B, dtype=np.float64)

    log.info(type(y))
    log.info(type(x))
    log.info("--------")
    log.info(y.shape)
    log.info(x.shape)
    log.info(f"B:{B}")
    log.info(f"y2:{y2}")
    log.info(f"out:{out}")


    for b in prange(B):
        s = y2[b, 0] + y2[b, n-1]

        for k in range(2, n-1, 2):
            s += 2.0 * y2[b, k]

        for k in range(1, n-1, 2):
            s += 4.0 * y2[b, k]

        if (n % 2) == 0:
            s -= 4.0 * y2[b, n-3]
            s += (3.0*y2[b, n-3] + 3.0*y2[b, n-2] + y2[b, n-1]) * (3.0/8.0)

        out[b] = (h / 3.0) * s

    log.info(f"out reshape:{out.reshape(shape[:-1])}")

    return out.reshape(shape[:-1])

######################## TEST ###################################################################


@njit(parallel=True, fastmath=True)
def simpson_logspace_nb(y, x):

    n = x.shape[0]

    # log10 grid step
    t0 = np.log10(x[0])
    t1 = np.log10(x[-1])
    h = (t1 - t0) / (n - 1)

    # flatten last axis
    B = y.size // n
    y2 = y.reshape(B, n)

    out = np.empty(B, dtype=np.float64)

    ln10 = np.log(10.0)

    for b in prange(B):

        # cambio di variabile: f = y * x * ln(10)
        f = y2[b, :] * x * ln10

        # caso con n dispari → Simpson 1/3 standard
        if (n % 2) != 0:
            s = f[0] + f[n-1]

            for k in range(1, n-1, 2):
                s += 4.0 * f[k]

            for k in range(2, n-1, 2):
                s += 2.0 * f[k]

            out[b] = (h/3.0) * s
            continue

        # caso con n pari → Simpson 1/3 fino a n-4 (incluso), poi 3/8 finale
        # 1) Simpson 1/3 sui primi n-3 punti
        n_main = n - 3

        s = f[0] + f[n_main-1]

        for k in range(1, n_main-1, 2):
            s += 4.0 * f[k]

        for k in range(2, n_main-1, 2):
            s += 2.0 * f[k]

        I = (h/3.0) * s

        # 2) Simpson 3/8 finale su: (n-4, n-3, n-2, n-1)
        I += (3*h/8.0) * (
            f[n-4] +
            3.0 * f[n-3] +
            3.0 * f[n-2] +
            f[n-1]
        )

        out[b] = I

    return out.reshape(y.shape[:-1])




##################################################################################################


def integrate_auto(y, x, method="trapz", log_int=False):
    """
    Integra y(x) scegliendo tra integrazione lineare o logaritmica.
    Se method='simpson', riduce i punti per migliorare la velocità.
    
    Parameters
    ----------
    y : array
        Valori della funzione.
    x : array
        Ascisse (lineari o logaritmiche).
    method : str
        'trapz' o 'simpson'
    log_int : bool
        True se x è logaritmico (es. np.logspace), False se lineare.
    """

    a, b = x[0], x[-1]
    N = len(x)

    if method == "trapz":
        if log_int:
            # integrazione log-log
            return trapz_loglog_nd_fast(y ,x)
        else:
            return trapz_numba(y, x)

    elif method == "simpson":
        if log_int:
            # riscalo punti in scala log
            log_a, log_b = np.log10(a), np.log10(b)
            log_range = log_b - log_a
            h_T = log_range / (N - 1)
            h_S = h_T ** 0.5
            N_S = int(np.round(log_range / h_S)) + 1
            x_s = np.logspace(log_a, log_b, N_S)
            y_s = np.interp(x_s, x, y)
            return simpson_uniform(y,x)
        else:
            h_T = (b - a) / (N - 1)
            h_S = h_T ** 0.5
            N_S = int(np.round((b - a) / h_S)) + 1
            x_s = np.linspace(a, b, N_S)
            y_s = np.interp(x_s, x, y)
            return simpson_logspace(y_s, x_s)

    else:
        raise ValueError(f"Metodo '{method}' non riconosciuto.")

##########################################################################################################à

def adaptive_shells_simple(f, theta_min_deg, theta_max_deg, tol=0.1, step_deg=0.1):
    """
    Divide [theta_min_deg, theta_max_deg] in shell tali che
    |f(theta) - f(theta_start)| > tol determina il bordo di una shell.

    Parametri
    ----------
    f : funzione f(theta_deg)
        La funzione da controllare (es. Gamma(theta)).
    theta_min_deg, theta_max_deg : float
        Intervallo in gradi.
    tol : float
        Tolleranza sulla variazione di f.
    step_deg : float
        Passo angolare per avanzare in gradi.

    Ritorna
    --------
    np.ndarray : array dei bordi in gradi (inclusi min e max)
    """

    edges = [theta_min_deg]
    theta_start = theta_min_deg
    f_start = f(theta_start)
    theta = theta_min_deg

    while theta < theta_max_deg:
        theta += step_deg
        if theta > theta_max_deg:
            theta = theta_max_deg

        f_now = f(theta)
        diff = abs(f_now - f_start)/f_start

        if diff > tol or theta >= theta_max_deg:
            # chiudo la shell e ricomincio
            edges.append(theta)
            theta_start = theta
            f_start = f_now

    return np.array(edges)


def funz(x):
    return 1/x


if __name__:

    import numpy as np
    import matplotlib.pyplot as plt



# Esempio di funzione test su griglia logaritmica
    def test_function(x):
        """
        Funzione di test morbida e integrabile,
        così si vede subito se l'integratore commette errori.
        """
        return np.exp(-x) * np.sin(10*x)**2 + x**0.3

    N = 501                   # numero punti (modifica se vuoi)
    xmin = 1e-4
    xmax = 1e2
    x = np.logspace(np.log10(xmin), np.log10(xmax), N)
    y = test_function(x)

    # Reshape per simpson_logspace_nb
    y_nb = y.reshape(1, -1)   # y con una dimensione batch fittizia

    # ===============================================================
    # INTEGRAZIONE
    # ===============================================================

    I1 = simpson_logspace(y, x)
    I2 = simpson_logspace_nb(y_nb, x)[0]

    print("---------------------------------------------------")
    print("   CONFRONTO TRA I DUE INTEGRATORI")
    print("---------------------------------------------------")
    print(f"Simpson logspace (Python):      {I1:.12e}")
    print(f"Simpson logspace (Numba) :      {I2:.12e}")
    print("---------------------------------------------------")
    print(f"Diff assoluta:                  {abs(I1 - I2):.3e}")
    print(f"Diff relativa:                  {abs(I1 - I2)/abs(I1):.3e}")
    print("---------------------------------------------------")


    # ===============================================================
    # GRAFICI
    # ===============================================================

    plt.figure(figsize=(9,5))
    plt.loglog(x, y, label="f(x) test")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Funzione di test su griglia logaritmica")
    plt.grid(True, which="both")
    plt.legend()
    plt.show()

    plt.figure(figsize=(9,5))
    plt.bar(["Simpson","Simpson Numba"], [I1, I2])
    plt.title("Confronto dei due integrali")
    plt.show()

    plt.figure(figsize=(9,5))
    plt.bar(["|I1 - I2|"], [abs(I1-I2)])
    plt.title("Differenza assoluta tra gli integrali")
    plt.show()