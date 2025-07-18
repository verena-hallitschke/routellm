"""Logging setup for the RouteLLM project."""
import logging
import sys


def set_up_logging(output_file: str, name: str | None) -> logging.Logger:
    """
    Set up the logger for the RouteLLM project.

    Args:
        output_file (str): Logging file path.
        name (str | None): Name of the logger. If None, will use the root logger.

    Returns:
        logging.Logger: Logger instance configured with the specified output \
            file and name.

    """
    logger = logging.getLogger()
    formatter = logging.Formatter(
        "%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.name = str(name)
    logger.level = logging.DEBUG

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(output_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger
