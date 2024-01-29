import os
import logging
import datetime
import tensorflow as tf


class TBManager:
    def __init__(self, tb_folder, run_name, tmp=False, interval=None, active=True):
        self.tb_folder = tb_folder
        self.run_name = run_name
        self.tmp = tmp
        DEFAULT_INTERVAL = 100
        self.interval = DEFAULT_INTERVAL if interval is None else interval
        self.active = active

        # set tensorboard path and create folders
        self.tb_path = None
        if not os.path.exists(self.tb_folder):
            os.makedirs(self.tb_folder)
        if not self.tmp:
            self.tb_path = os.path.join(
                self.tb_folder,
                self.run_name or datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
            )
        else:
            self.tb_path = os.path.join(self.tb_folder, "tmp")

        # remove existing files in folder if it exists
        if os.path.exists(self.tb_path):
            for filename in os.listdir(self.tb_path):
                os.remove(os.path.join(self.tb_path, filename))

        # initialize counter variable
        self.counter = tf.Variable(0, trainable=False, dtype=tf.int64)

        # initialize tf file writer
        self.tb_writer = tf.summary.create_file_writer(self.tb_path)

    def __getattr__(self, key):
        res = None
        if key == "hot":
            res = self.active and self.counter % 100 == 0
        else:
            raise ValueError()
        return res

    def step(self):
        self.counter.assign_add(tf.constant(1, dtype=tf.int64))

    def enable(self):
        self.active = True

    def disable(self):
        self.active = False

    def scalar(self, name, data):
        with self.tb_writer.as_default():
            tf.summary.scalar(name, data, step=self.counter)

    def image(self, name, data):
        with self.tb_writer.as_default():
            tf.summary.image(name, data, step=self.counter)

    def histogram(self, name, data):
        with self.tb_writer.as_default():
            tf.summary.histogram(name, data, step=self.counter)


@DeprecationWarning
def _create_logger(log_to_file=False, verbose=False):
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
