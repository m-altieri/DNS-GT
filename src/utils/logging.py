import os
import logging
import datetime
import tensorflow as tf
import tensorboard as tb

import utils.nn


class TBManager:
    def __init__(
        self, tb_folder, run_name, tmp=False, interval=None, enabled=True, verbose=False
    ):
        self.tb_folder = tb_folder
        self.run_name = run_name
        self.tmp = tmp
        DEFAULT_INTERVAL = 100
        self.interval = DEFAULT_INTERVAL if interval is None else interval
        self.enabled = enabled
        self.force_write = False
        self.verbose = verbose
        self.scalars = {}
        self.images = {}
        self.histograms = {}

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

        # start tensorboard instance
        self.tb_instance = None

        # initialize counter variable
        self.counter = tf.Variable(0, trainable=False, dtype=tf.int64)

        # initialize tf file writer
        self.tb_writer = tf.summary.create_file_writer(self.tb_path)

    def run(self):
        self.tb_instance = tb.program.TensorBoard()
        self.tb_instance.configure(logdir=self.tb_path)
        tb_url = self.tb_instance.launch()
        print(f"[INFO] TensorBoard instance launched at {tb_url}")

    def is_hot(self):
        if self.force_write:
            if self.enabled:
                print("[DEBUG] TBManager is hot because it is forced.")
            else:
                print(
                    "[WARN] You are forcing TBManager to stay hot but it is not enabled."
                )
        return self.enabled and (self.counter % 100 == 0 or self.force_write)

    def step(self):
        if self.enabled:
            self.counter.assign_add(tf.constant(1, dtype=tf.int64))

    def force(self, value, /):
        """Make is_hot() return True until you call force(False), unless
        the TBManager is disabled (as if interval is temporarily 1).
        Note: this can have unexpected behaviour. Use only for debugging.

        :param value: Whether to force the TBManager to stay hot.
        :type value: bool
        """
        self.force_write = value

    def scalar(self, name, data):
        if self.enabled:
            if name not in self.scalars:
                self.scalars[name] = {"step": 0}

            logging_step = self.scalars[name]["step"]

            if self.verbose:
                print(
                    f"[INFO] Writing scalar with name {name} (global step: {self.counter}, logging step: {logging_step})"
                )

            with self.tb_writer.as_default():
                tf.summary.scalar(name, data, step=logging_step)

            self.scalars[name]["step"] = logging_step + 1

    def image(self, name, data, minmax=False):
        if self.enabled:
            if name not in self.images:
                self.images[name] = {"step": 0}

            logging_step = self.images[name]["step"]

            if self.verbose:
                print(
                    f"[INFO] Writing image with name {name} (global step: {self.counter}, logging step: {logging_step})"
                )
            # if the channel (color) axis is missing, add it with dim 1 (greyscale)
            data = tf.cond(
                tf.math.equal(tf.rank(data), tf.constant(3)),
                lambda: tf.expand_dims(data, -1),
                lambda: data,
            )
            # if minmax, apply minmax normalization to the image
            data = tf.cond(
                tf.constant(minmax),
                lambda: utils.nn.minmax(data),
                lambda: data,
            )

            with self.tb_writer.as_default():
                tf.summary.image(name, data, step=logging_step)

            self.images[name]["step"] = logging_step + 1

    def histogram(self, name, data):
        if self.enabled:
            if name not in self.histograms:
                self.histograms[name] = {"step": 0}

            logging_step = self.histograms[name]["step"]

            if self.verbose:
                print(
                    f"[INFO] Writing histogram with name {name} (global step: {self.counter}, logging step: {logging_step})"
                )

            with self.tb_writer.as_default():
                tf.summary.histogram(name, data, step=logging_step)

            self.histograms[name]["step"] = logging_step + 1


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
