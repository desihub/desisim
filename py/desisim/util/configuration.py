"""
Setup the global variable with all the configuration options
"""
import ConfigParser


def setup_survey(configfile="test.cfg"):
    """
    Loads the global variables that define the survey.

    input: configuration filename
    
    output: None

    notes: the global variable __CONFIG__ is set from None to 
           a variable of the type returned by ConfigParser.SafeConfigParser
    """
    global __CONFIG__ 
    try:
        __CONFIG__ = ConfigParser.SafeConfigParser()
        __CONFIG__.read(configfile)        
    except  Exception, err:
        raise err
    return
        

