import datetime
import logging
import os
import pathlib
import sys


def setup_logger(
    log_level: str = "info",
    log_to_console: bool = True,
    log_file: pathlib.Path = pathlib.Path(os.devnull),
):
    """
    Set up logging objects based on user input.

    Parameters
    ----------
    log_level : str, default = info
        The level of logging messages: "critical", "warning", "info", "debug"

    log_to_console : bool, default = True
        If True, log messages are displayed on the screen in addition
        to the log file (if configured)

    log_file : pathlib.Path, default = pathlib.Path(os.devnull)
        The path on the disk where log files are written

    Raises
    ------
    ValueError
        If the log_file specified already exists or if an invalid value for
        log_level is provided
    """
    supported_log_levels = {
        "critical": logging.CRITICAL,
        "warning": logging.WARNING,
        "info": logging.INFO,
        "debug": logging.DEBUG,
    }
    if log_level not in supported_log_levels:
        raise ValueError(
            f"Invalid value for log_level: {log_level}. "
            f"Acceptable values are: crticial, warning, info, debug."
        )

    handlers = []
    if log_to_console:
        stdout_handler = logging.StreamHandler(sys.stdout)
        handlers.append(stdout_handler)

    if log_file != pathlib.Path(os.devnull):
        # if os.path.exists(log_file):
        #     raise ValueError(
        #         f"Log file: {str(log_file)} already exists. Please specify new log file."
        #     )
        file_handler = logging.FileHandler(log_file)
        handlers.append(file_handler)

    if log_level != "debug":
        # Show a simple logger for general purposes
        logger_format = "pystar: %(levelname)s: %(message)s"
        logger_date = None

    else:
        # Show detailed log messages only in debug mode
        logger_format = (
            "%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s"
        )
        logger_date = "%d-%b-%y %H:%M:%S"

    logging.basicConfig(
        level=supported_log_levels[log_level],
        format=logger_format,
        datefmt=logger_date,
        handlers=handlers,
    )

    # Prevents double log output when the solver is called
    logger = logging.getLogger("gurobipy")
    logger.propagate = False

    # pylint: disable = logging-fstring-interpolation
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info(f"Configuring a new pystar logger: [{datetime.datetime.now()}] ")
