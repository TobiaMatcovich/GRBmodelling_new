import logging
import os

#LOG_FILE = "grb_debug.log"

'''def setup_logger():
    """
    Configura il logger principale del progetto.
    Sovrascrive il file ad ogni esecuzione.
    """
    # Rimuove eventuali handler duplicati (utile per Jupyter)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        filename=LOG_FILE,
        filemode="w",               # sovrascrive ogni volta
        level=logging.INFO,        # livello minimo
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Output anche su console
    console = logging.StreamHandler()
    console.setLevel(logging.ERROR)   # più “pulito” sul terminale
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logging.getLogger("").addHandler(console)

    logging.info("Logger inizializzato. File: %s", LOG_FILE)'''



def setup_logger(name=None, log_dir="logs", level=logging.INFO):
    """
    Configura un logger principale o per modulo, sovrascrivendo il file ad ogni esecuzione.

    name: nome del logger, di default usa il modulo che lo importa
    log_dir: cartella dove salvare i log
    level: livello minimo di log
    """
    if name is None:
        name = "__main__"  # default se non specificato

    # Rimuove handler duplicati (utile per Jupyter o reload multipli)
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(level)

    # Crea cartella log se non esiste
    os.makedirs(log_dir, exist_ok=True)

    # File log specifico per modulo
    log_file = os.path.join(log_dir, f"{name}.log")
    #log_file = os.path.join(log_dir, "main.log")
    fh = logging.FileHandler(log_file, mode="w")  # sovrascrive ogni volta
    fh.setLevel(level)
    fh_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(fh_formatter)

    # Console output (solo errori per non intasare il terminale)
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    # Aggiunge handler al logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info("Logger inizializzato. File: %s", log_file)

    return logger
