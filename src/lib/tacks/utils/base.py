# -*- coding: utf-8 -*-
"""Util functions.

Author: Ronan Hamon <ronan.hamon@ec.europa.eu>
"""
import argparse
import configparser
import hashlib
import logging
import random
import shutil
import subprocess
import sys
import traceback
import warnings
from pathlib import Path

import numpy as np
import yaml


def check_path(path, logger=None):
    """Checks the existence of a path.

    Parameters
    ----------
    path : pathlib.Path or str
        Path to check.
    logger : logging.Logger or None
        Logging system.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    """
    path = Path(path)
    if not path.exists():
        errmsg = "Path does not exist: {}"
        if logger:
            logger.critical(errmsg.format(path))
        raise FileNotFoundError(errmsg.format(path))


def compute_hash(chunk):
    """Computes a hash value of a chunk of text.

    Parameters
    ----------
    chunk : str
        Chunk of text to hash.

    Returns
    -------
    str
        Hash computed using MD5.
    """
    return hashlib.md5(chunk.encode()).hexdigest()


def get_argparser(
    description, flags=None, pos_args=None, opt_args=None, default_values=None, **kwargs
):
    """Return a plain argument parser.

    Parameters
    ----------
    description : str
        Description of the argparser.
    flags : list of str or None, optional
        List of flags.
    pos_args : list of str or None, optional
        Ordered list of positional arguments (default: None).
    opt_args : list of str or None, optional
        List of optional arguments (default: None).
    default_values : dict or None, optional
        Default values for parameters (default: None).

    Returns
    -------
    argparse.ArgumentParser
    """

    ARGPARSER_FLAGS = {
        "clearml": ("c", "use clearml"),
        "debug": ("d", "debug mode."),
        "force": ("f", "force mode."),
        "launch": ("l", "launch computation."),
        "silent": ("i", "silent mode."),
        "plot": ("p", "plot figures."),
        "publish": ("q", "publish the model."),
        "reset": ("r", "reset computation."),
        "save": ("s", "save data."),
    }

    # define list of arguments for argparser
    ARGPARSER_ARGS = {
        # main group
        "device": (
            "main",
            {
                "default": None,
                "type": str,
                "help": "Device (cuda|cpu).",
            },
        ),
        "name": (
            "main",
            {
                "metavar": "NAME",
                "default": None,
                "type": str,
                "help": "Name of the run.",
            },
        ),
        "params": (
            "main",
            {
                "metavar": "PARAMS_PATH",
                "default": None,
                "type": Path,
                "help": "Path to the param file.",
            },
        ),
        "seed": (
            "main",
            {"default": None, "type": int, "help": "Seed for random generator."},
        ),
        "workspace_name": (
            "main",
            {"default": None, "type": str, "help": "Name of the workspace."},
        ),
        # data group
        "data": (
            "data",
            {
                "metavar": "DATA_PATH",
                "default": None,
                "type": Path,
                "help": "Path to data.",
            },
        ),
        "split_name": (
            "data",
            {
                "metavar": "NAME",
                "default": "train",
                "type": str,
                "help": "Name of the split.",
            },
        ),
        "batch_size": (
            "data",
            {
                "default": 128,
                "type": int,
                "metavar": "N",
                "help": "Size of a batch.",
            },
        ),
        "n_workers": (
            "data",
            {
                "metavar": "N",
                "default": 0,
                "type": int,
                "help": "Number of data loading workers.",
            },
        ),
        # model group
        # to be defined at initialization time
        "arch": None,
        "weights": (
            "model",
            {"metavar": "WEIGHTS_PATH", "type": Path, "help": "path to weights."},
        ),
        "model": (
            "model",
            {"metavar": "MODEL_PATH", "type": str, "help": "model name."},
        ),
        # training group
        "n_epochs": (
            "training",
            {
                "metavar": "N",
                "default": 10,
                "type": int,
                "help": "Number of epochs.",
            },
        ),
        "optimizer": (
            "training",
            {
                "metavar": "STR",
                "default": "adam",
                "type": str,
                "help": "Name of the optimizer (adam|sgd).",
            },
        ),
        "max_iter": (
            "training",
            {
                "metavar": "INT",
                "type": int,
                "default": 100,
                "help": "Number of iterations.",
            },
        ),
        "lr": (
            "training",
            {
                "metavar": "FLOAT",
                "default": 1e-4,
                "type": float,
                "help": "Learning rate.",
            },
        ),
        "wd": (
            "training",
            {
                "metavar": "FLOAT",
                "default": 1e-4,
                "type": float,
                "help": "Weight decay.",
            },
        ),
        "n_gpus": (
            "training",
            {"metavar": "INT", "type": int, "help": "Number of GPUs"},
        ),
        "eval_interval": (
            "training",
            {
                "metavar": "INT",
                "type": int,
                "help": "Interval at which an evaluation is performed.",
            },
        ),
        "save_interval": (
            "training",
            {
                "metavar": "INT",
                "type": int,
                "help": "Interval at which a save is performed.",
            },
        ),
    }

    class CustomFormatter(
        argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
    ):
        pass

    if flags is None:
        flags = []
    if pos_args is None:
        pos_args = []
    if opt_args is None:
        opt_args = []

    # get extra parameters
    archs = kwargs["archs"] if "archs" in kwargs else [""]
    models = kwargs["models"] if "models" in kwargs else [""]

    ARGPARSER_ARGS["arch"] = (
        "model",
        {
            "metavar": "ARCH",
            "default": models[0],
            "choices": models,
            "help": "Model architecture.",
        },
    )

    # create the parser
    argparser = argparse.ArgumentParser(description, formatter_class=CustomFormatter)

    parser_groups = {
        "main": argparser,
        "data": argparser.add_argument_group("Data"),
        "model": argparser.add_argument_group("Model"),
        "training": argparser.add_argument_group("Training"),
        "eval": argparser.add_argument_group("Eval"),
    }

    for flag in flags:
        if flag not in ARGPARSER_FLAGS:
            warnings.warn(f"Unknown flag: {flag}")
        else:
            flag_short, help_desc = ARGPARSER_FLAGS[flag]
            parser_groups["main"].add_argument(
                f"-{flag_short}", f"--{flag}", action="store_true", help=help_desc
            )

    for arg in pos_args + opt_args:
        if arg not in ARGPARSER_ARGS:
            warnings.warn(f"Unknown arguments: {arg}")
        else:
            group, params = ARGPARSER_ARGS[arg]

            # set default value
            if default_values is not None and arg in default_values:
                params["default"] = default_values[arg]

            # add hyphens if optional arguments
            if arg in opt_args:
                arg = "--{}".format(arg.replace("_", "-"))

            # add argument in parser
            parser_groups[group].add_argument(arg, **params)

    return argparser


def get_config():
    """Returns the config of the user.

    Returns
    -------
    ConfigParser
    """
    HOME_PATH = Path("~").expanduser()

    return ConfigParser(HOME_PATH / ".config" / "tacksrc")


def get_logger(
    name,
    to_file=True,
    to_console=True,
    file_level=logging.INFO,
    console_level=logging.INFO,
    log_path=None,
):
    """Return an instance of logger.

    Parameters
    ----------
    name : str
        Name of the logger.
    to_file : bool, optional
        Indicates if the logger writes in a file.
    to_console : bool, optional
        Indicates if the logger displays in console.
    file_level : int
        Level of messages in the file.
    console_level : int
        Level of messages in the console.
    log_path : pathlib.Path or str or None
        Path where to save the log file. If None, save in the current directory.

    Returns
    -------
    logging.Logger
    """

    class LoggerFormatter(logging.Formatter):
        """Custom formatter for logger."""

        GREY = "\x1b[38;20m"
        YELLOW = "\x1b[33;20m"
        RED = "\x1b[31;20m"
        BOLD_RED = "\x1b[31;1m"
        RESET = "\x1b[0m"
        # logger_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
        logger_fmt = "%(asctime)s|%(name)s|%(levelname)s - %(message)s"

        FORMATS = {
            logging.DEBUG: GREY + logger_fmt + RESET,
            logging.INFO: GREY + logger_fmt + RESET,
            logging.WARNING: YELLOW + logger_fmt + RESET,
            logging.ERROR: RED + logger_fmt + RESET,
            logging.CRITICAL: BOLD_RED + logger_fmt + RESET,
        }

        def format(self, record):
            log_fmt = self.FORMATS.get(record.levelno)
            formatter = logging.Formatter(log_fmt)
            return formatter.format(record)

    LP_INFO = 15
    MP_INFO = 25
    HP_INFO = 35

    logging.addLevelName(LP_INFO, "LP")
    logging.addLevelName(MP_INFO, "MP")
    logging.addLevelName(HP_INFO, "HP")

    def lp_info(self, message, *args, **kws):
        if self.isEnabledFor(LP_INFO):
            self._log(LP_INFO, message, args, **kws)

    def mp_info(self, message, *args, **kws):
        if self.isEnabledFor(MP_INFO):
            self._log(MP_INFO, message, args, **kws)

    def hp_info(self, message, *args, **kws):
        if self.isEnabledFor(HP_INFO):
            self._log(HP_INFO, message, args, **kws)

    logging.Logger.lp_info = lp_info
    logging.Logger.mp_info = mp_info
    logging.Logger.hp_info = hp_info

    logger = logging.Logger(name)
    logger.setLevel(logging.DEBUG)

    if to_file:
        if log_path is None:
            log_path = Path(".").absolute()

        log_path.mkdir(exist_ok=True, parents=True)
        handler_path = log_path / "{}.log".format(name)

        # check if a file handler already exists
        file_handler = None
        for handler in logger.handlers:
            if type(handler) == logging.FileHandler:
                file_handler = handler
                break

        # otherwise, create a new handler
        if file_handler is None:
            file_handler = logging.FileHandler(str(handler_path))

        # set level of verbosity
        file_handler.setLevel(file_level)
        # set formatting
        file_handler.setFormatter(LoggerFormatter())

        # add the new handler if new
        if repr(file_handler) not in map(repr, logger.handlers):
            logger.addHandler(file_handler)

    if to_console:
        # check if a file handler already exists
        console_handler = None
        for handler in logger.handlers:
            if type(handler) == logging.StreamHandler:
                console_handler = handler
                break

        if console_handler is None:
            console_handler = logging.StreamHandler(sys.stdout)

        # set level of verbosity
        console_handler.setLevel(console_level)
        # set formatting
        console_handler.setFormatter(LoggerFormatter())

        if repr(console_handler) not in map(repr, logger.handlers):
            logger.addHandler(console_handler)

    logger.to_file = to_file
    logger.to_console = to_console

    # dofine custom exception handler
    def exception_handler(exc_type, value, tb):
        logger.exception(
            "{0}".format("\n".join([item.strip() for item in traceback.format_tb(tb)])),
            exc_info=False,
        )
        logger.exception(
            "{0}: {1}".format(exc_type.__name__, str(value)), exc_info=False
        )

    logger.exception_handler = exception_handler

    # install exception handler
    sys.excepthook = logger.exception_handler

    return logger


def run_bash_command(
    cmd, cwd=".", stdout_file=None, stderr_file=None, feed_input=None, wait=True
):
    """Runs a bash command.

    Parameters
    ----------
    cmd : str
        Command to run.
    cwd : pathlib.Path or path, optional
        Working directory.
    stdout_file : pathlib.Path or path or None, optional
        Path to the stdout file, if redirection.
    stderr_file : pathlib.Path or path or None, optional
        Path to the stderr file, if redirection.
    feed_input : str or None, optional
        Input to feed during the execution of the command.
    wait : bool, optional
        Wait till the end of the process.

    Returns
    -------
    subprocess.Popen
    """
    stdout = open(str(stdout_file), "w") if stdout_file is not None else subprocess.PIPE
    stderr = open(str(stderr_file), "w") if stderr_file is not None else subprocess.PIPE

    stdin = subprocess.PIPE if feed_input is not None else None

    process = subprocess.Popen(
        cmd.split(" "), cwd=str(cwd), stdin=stdin, stdout=stdout, stderr=stderr
    )

    if feed_input:
        # convert input into bytes
        feed_input = bytes(feed_input, "utf-8")
        stdout, stderr = process.communicate(input=feed_input)
    elif wait:
        stdout, stderr = process.communicate()
    else:
        stdout = stderr = None

    if stdout is not None:
        stdout = stdout.decode("utf-8")
    if stderr is not None:
        stderr = stderr.decode("utf-8")

    return process, (stdout, stderr)


def set_seed(seed, deterministic=False):
    """Set seed for reproducibility for torch and numpy.

    Deterministic operation increases the reproducibility but has a negative impact on
    performances. See https://pytorch.org/docs/stable/notes/randomness.html for details.

    Parameters
    ----------
    seed : int
        Seed to use.
    deterministic : bool, optional
        Indicates if deterministic setup is used.
    """

    import torch

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic

    torch.manual_seed(seed)
    np.random.seed(seed)

    random.seed(seed)


class ConfigParser(configparser.ConfigParser):
    """Custom file parser.

    This class inherits from ConfigParser in module configparser. It adds a
    method to properly read a path in the config file and a logging system.

    The name of the config file should have 'cfg' extension, and not contain an
    underscore. Amended experiments can be used by adding '_X' as suffix, with
    X an integer.

    Parameters
    ----------
    config_path : pathlib.Path or str or None, optional
        Path to the config file. If None, no writing is possible.
    exists : bool, optional
        Indicates if the config already exists or not. Not used if
        `config_path` is None.
    logger : logging.Logger or None
        Logging system. If None, create a new logger.

    Raises
    ------
    FileNotFoundError
        if the parameter 'exists' is set to True and the config file does not
        exist.
    """

    def __init__(self, config_path=None, exists=True, logger=None):
        super().__init__()

        self.logger = logger
        self.config_path = Path(config_path) if config_path is not None else None
        # get file name
        if self.config_path:
            filename = self.config_path.stem.split("_")
            with_amend = len(filename) == 2

            # load the config
            if exists:
                check_path(self.config_path)

            # read the config file (or the root config file if amended config
            # file)
            if with_amend:
                amended_config_path = self.config_path
                self.config_path = self.config_path.parent / "{}.cfg".format(
                    filename[0]
                )
            self.read(str(self.config_path))

            # update the config file with the amended config file
            if with_amend:
                amended_config_parser = configparser.ConfigParser()
                amended_config_parser.read(amended_config_path)

                for section_name, section in amended_config_parser.items():
                    for option_name, option_value in section.items():
                        self[section_name][option_name] = option_value

    def get_path(
        self,
        section,
        option,
        check_exists=True,
        *,
        raw=False,
        vars=None,
        fallback=configparser._UNSET,
        **kwargs,
    ):
        """Get the path filled in a given option of a given section.

        Parameters
        ----------
        section : str
            Name of the section.
        option : str
            Name of the option.
        check_exists : bool, optional
            Indicate if the existence of the path is checked.

        Returns
        -------
        pathlib.Path or None

        Raises
        ------
        FileNotFoundError
            if the parameter 'check_exists' is set to True and the path does
            not exist.
        """
        path = self.get(
            section, option, raw=raw, vars=vars, fallback=fallback, **kwargs
        )

        if path is None:
            errmsg = "Option {} of section {} is empty."
            self.logger.error(errmsg.format(option, section))

        path = Path(path).expanduser()

        if check_exists:
            check_path(path, self.logger)

        return path

    def get_list(
        self,
        section,
        option,
        separator=",",
        *,
        raw=False,
        vars=None,
        fallback=configparser._UNSET,
        **kwargs,
    ):
        """Get a list value.

        Parameters
        ----------
        section : str
            Name of the section.
        option : str
            Name of the option.
        """
        list_value = self.get(
            section, option, raw=raw, vars=vars, fallback=fallback, **kwargs
        )
        return list_value.split(separator)

    def set(self, section, option, value=None):
        """Set an option.

        Extends configparser.ConfigParser.set by converting a non-string value
        into string.
        """
        super().set(section, option, str(value))

    def write(self):
        """Write the config file.

        Raises
        ------
        ValueError
            If `config_path` is set to None.
        """
        if not self.config_path:
            errmsg = "Config path not provided."
            self.logger.error(errmsg)

        with open(str(self.config_path.absolute()), "w") as in_file:
            super().write(in_file)


class Workspace:
    """Workspace object.

    Parameters
    ----------
    name : str
        Name of the workspace.
    instance_name : str
        Name of the instance.
    args : Namespace or None, optional
        Arguments as returned by :class:`argparser.ArgumentParser` (default: None).
    workspace_path : str or pathlib.Path, optional
        Workspace path. Overrides default settings (default: None).
    check_args : bool, optional
        Indicates if arguments are checked with previous values (default: False).
    reset : bool, optional
        Indicates if the workspace is reset to empty (default: False).
    """

    path_names = ["imgs", "logs", "outputs", "weights", "xps"]

    def __init__(
        self,
        name,
        instance_name,
        args=None,
        workspace_path=None,
        check_args=False,
        full_reset=False,
    ):
        self.name = name
        self.instance_name = instance_name
        self.args = (
            args if isinstance(args, dict) else (args.__dict__.copy() if args else {})
        )
        self.check_args = check_args

        flags = {
            flag: self.args.pop(flag) for flag in ARGPARSER_FLAGS if flag in self.args
        }

        # convert pathlib.Path into str
        for key, value in self.args.items():
            if isinstance(value, Path):
                self.args[key] = str(value)

        debug = flags["debug"] if "debug" in flags else False
        reset = flags["reset"] if "reset" in flags else False
        silent = flags["silent"] if "silent" in flags else False
        force = flags["force"] if "force" in flags else False

        self.verbose = not silent

        # load config
        config = get_config()

        # define the path to the workspace
        if workspace_path is None:
            workspace_path = config.get_path("paths", "workspaces")
        self.workspace_path = Path(workspace_path) / self.name

        # define log path in workspace folder
        self.logs_path = self.workspace_path / "logs"

        self.workspace_path.mkdir(exist_ok=True, parents=True)

        # define directories is sbworkspace folder
        for path_name in self.path_names:
            # define path
            path = self.workspace_path / path_name

            # store as attribute
            self.__dict__[f"{path_name}_path"] = path

            # remove directories if full reset
            if full_reset and path.exists():
                shutil.rmtree(path)

            # create directory
            path.mkdir(exist_ok=True)

        # init logger
        console_level = (
            (logging.DEBUG if debug else logging.INFO)
            if self.verbose
            else logging.ERROR
        )
        self.logger = get_logger(
            instance_name,
            log_path=self.logs_path,
            file_level=logging.DEBUG if debug else logging.INFO,
            console_level=console_level,
        )
        self.logger.info("Initializing workspace %s", self.name)
        self.logger.info("   Instance: %s", self.instance_name)

        # check that args of current workspace are the same
        if (
            not self.is_empty()
            and (self.workspace_path / "args.yaml").exists()
            and not reset
        ):
            with open(self.workspace_path / "args.yaml", "r") as outfile:
                saved_args = yaml.safe_load(outfile)

            if self.check_args and not force and self.args != saved_args:
                err_msg = "Mismatch between current args and saved args. {}\n{}"
                raise ValueError(err_msg.format(self.args, saved_args))
        else:
            # save args in run path
            with open(self.workspace_path / "args.yaml", "w") as outfile:
                yaml.safe_dump(self.args, outfile)

        self.logger.info("Workspace %s ready.", self.name)

    def is_empty(self):
        """Check whether the workspace is empty or not.

        Logs and args are ignored.

        Returns
        -------
        bool
            Indicates if the workspace is empty or not.
        """
        return all(
            [
                len(list(self.__dict__[f"{path_name}_path"].glob("*"))) < 0
                for path_name in self.path_names
            ]
        )


class Debugger:
    """Basic debugger."""

    def __init__(self):
        self.counters = {}
        self.traces = {}

    def trace(self, name, *values):
        if name not in self.counters:
            self.counters[name] = 0
        else:
            self.counters[name] += 1

        idn = self.counters[name]
        self.traces[f"{name}_{idn:d}"] = values
