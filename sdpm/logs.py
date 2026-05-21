import logging
from rich.logging import RichHandler

def get_logger(name: str, filename: str | None = None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger
    console_formatter = logging.Formatter(
        "[%(name)s] %(message)s",
        datefmt="%d.%m.%Y %H:%M:%S"
    )
    console_handler = RichHandler(
        level=logging.INFO,
        show_path=False,
        enable_link_path=False,
        markup=True,
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    if filename is not None:
        file_handler = logging.FileHandler(filename, encoding='utf-8')
        file_formatter = logging.Formatter(
            "[ %(name)s | %(levelname)s | %(asctime)s ] %(message)s",
            datefmt="%d.%m.%Y %H:%M:%S"
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    return logger
