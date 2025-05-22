"""
Module to wrap some logging features

Copyright (C) 2024 Witteveen+Bos (https://www.witteveenbos.com)

Licensed under the EUPL, Version 1.2 or - as soon they will be approved by
the European Commission - subsequent versions of the EUPL (the "Licence");
You may not use this work except in compliance with the Licence.
You may obtain a copy of the Licence at:
https://joinup.ec.europa.eu/software/page/eupl

Unless required by applicable law or agreed to in writing, software
distributed under the Licence is distributed on an "AS IS" basis,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the Licence for the specific language governing permissions and
limitations under the Licence.
"""

# External modules
import os
import sys
import datetime
import logging
import logging.handlers


# %% ===========================================================================
def defLogFile(logdir, prefix='', timeformat='%Y%m%d-%H%M%S'):
    """Determine the name of the logfile with optional prefix and incorporated timestamp.


    Parameters
    ----------
    logdir : str
        filepath of the directory that should contain the logfile
    prefix : str, optional
        optional prefix for the name of the logfile
    timeformat : str, default='%Y%m%d-%H%M%S'
        used timeformat for logging


    Returns
    -------
    str
        directory of the logfile

    """

    # Determine the complete filename of the log-file, based on time
    curTime = datetime.datetime.now().strftime(timeformat)
    logfullfile = os.path.join(logdir, prefix + curTime + '.log')

    return logfullfile


# %% ===========================================================================
def startLogging(logfullfile=None, onScreen=True, level=logging.DEBUG, header=None):
    """
    Starting the log-process. Should be called before the first logging.entry


    Parameters
    ----------
    logfullfile : str, default=None
        fullfilename of the logfile\n
        setting this parameter to None (default) only prints towards console
    level : log-depth, default=logging.DEBUG
        logging depth of particular logger
    header : ASCII-art, optional
        optional ASCII-art header for log-file



    :return: None
    """

    # Start the logging process with appropriate level
    myLogger = logging.getLogger()
    myLogger.setLevel(level)

    # First: a handler for console messages
    # Write messages towards the console, unless a StreamHandler already exists
    # Remark: a FileHandler inherits from StreamHandler. Therefore check with
    # 'type' and not with 'isinstance'
    hasStreamHandler = any([type(handler) == logging.StreamHandler
                            for handler in logging.getLogger().handlers])

    if onScreen and not hasStreamHandler:
        consoleHandler = logging.StreamHandler()
        # Determine a (slightly different) log-format, since this is on screen, not in file
        strmFmtr = logging.Formatter(fmt='%(asctime)s.%(msecs)03d -- %(levelname)s -- %(message)s',
                                     datefmt='%H:%M:%S')
        consoleHandler.setFormatter(strmFmtr)
        myLogger.addHandler(consoleHandler)

    # Check if a particular file handler exists, and remove them
    hasFileHandler = any([type(handler) == logging.FileHandler
                          for handler in myLogger.handlers])
    if hasFileHandler:
        # If a file handler already exists, then this particular file handler
        # must be closed (otherwise, all messages will be duplicated and printed
        # twice into the predefined file)
        oldFileHandlers = [handler for handler in myLogger.handlers
                           if type(handler) == logging.FileHandler]
        [myLogger.removeHandler(oldFH) for oldFH in oldFileHandlers]

    # Second: if a filename is given, make another (new) handler to write towards file
    if not logfullfile is None:
        # It is assumed that the logfile not necessarily exists, but its parent folder do
        fileHandler = logging.FileHandler(logfullfile)

        # Log message formatting
        fileFmtr = logging.Formatter(fmt='[%(asctime)s]; [%(levelname)-8s]; %(funcName)s -> %(message)s',
                                     datefmt='%Y-%m-%d %H:%M:%S')
        fileHandler.setFormatter(fileFmtr)
        myLogger.addHandler(fileHandler)

        # And: uncaught exceptions should be written towards logger
        sys.excepthook = logException

    # Put a header for the logfile, if this is given
    if header is not None:
        # Printing the header in style
        logging.info('{:=^60}'.format(''))
        logging.info('{:=^60}'.format(header.upper()))
        logging.info('{:=^60}'.format(''))

    return  # nothing: the logging process is started


# %% ===========================================================================
def stopLogging(msg=None):
    """
    stop the logging process

    Parameters
    ----------
    msg : str, default=None
        message format string


    :return: None
    """

    # Get logger object
    logger = logging.getLogger()
    # Closing the file handlers before shutdown log
    while logger.hasHandlers():
        # Throw them away, one by one, and close them
        handler = logger.handlers.pop()
        handler.close()
    logging.info('log handlers closed')

    # Add custom end-message to log
    if msg is not None:
        logging.info('{:=^60}'.format(msg.upper()))

    logging.shutdown()
    return  # nothing: logging is stopped


# %% ===========================================================================
def logException(exc_type, exc_value, exc_traceback):
    """
    For printing the failure (error stack) towards the logger.
    If a mail-handler is attached towards the logger, a logfile will be mailed.

    Parameters
    ----------
    exc_type : str
        type of the exception
    exc_value : str
        value of the exception
    exc_traceback :
        traceback of the exception


    :return: None

    """

    # Do not catch on Keyboad Interrupt: behaviour should then be default
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    # However, the log, including the stack, should be printed
    logging.critical('uncaught exception',
                     exc_info=(exc_type, exc_value, exc_traceback))

    # And close particular file neatly
    stopLogging('closed on error')
