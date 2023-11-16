# UNUSED
import logging


def create_logger(log_to_file=False, verbose=False):
    """Create a new global logger for this file.

    `logger.critical()`, `logger.error()`, `logger.warning()` and `logger.info()` will always log the message.
    `logger.debug()` will log the message if `verbose=True`.
    `logger.trace()` will log the message *only to file* if `log_to_file=True`.

    Args:
       log_to_file (bool, default=False): Whether you want to also log everything to file `log.txt`. Note that *all* logs are logged to file, even trace-level ones.
       verbose (bool, default=False): If True, also log debug-level logs.
    """
    logger = logging.getLogger(__name__)

    def _addLogLevel(level_name, level_num):
        """Create a new logging level for the logger.

        Args:
           level_name (str): Name of the new level.
           level_num (int): Logging priority of the new level. For reference, consult https://docs.python.org/3/library/logging.html#logging-levels.
        """
        logging.Logger.trace = lambda self, msg, *args, **kws: logger._log(
            level_num, msg, args, **kws
        )
        setattr(logging, level_name, level_num)
        logging.addLevelName(level_num, level_name)

    _addLogLevel("TRACE", 5)
    logger.setLevel(logging.INFO)
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setLevel("DEBUG")
    logger.addHandler(consoleHandler)
    if log_to_file:
        logFormatter = logging.Formatter("[%(levelname)-5.5s]  %(message)s")
        fileHandler = logging.FileHandler("log.txt")
        fileHandler.setFormatter(logFormatter)
        fileHandler.setLevel(logging.TRACE)
        logger.addHandler(fileHandler)
    if verbose:
        logger.setLevel(logging.DEBUG)

    return logger
