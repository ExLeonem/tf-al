import os, sys, logging


def setup_logger(debug, name="Runner", log_level=logging.DEBUG, default_log_level=logging.CRITICAL):
    """
        Setup a logger for the active learning loop

        Parameters:
            debug (bool): activate logging output in console?
            name (str): The name of the logger to use. (default='Runner')
            log_level (logging.level): The log level to use when debug==True. (default=logging.DEBUG)
            default_log_level (logging.level): The default log level to use when debug==False. (default=logging.CRITICAL)

        Returns:
            (logging.Logger) a configured logger object.
    """

    logger = logging.Logger(name)
    log_level = log_level if debug else default_log_level

    logger.handler = logging.StreamHandler(sys.stdout)
    logger.handler.setLevel(log_level)
    
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    logger.handler.setFormatter(formatter)
    logger.addHandler(logger.handler)
    return logger