import os
from os.path import join, dirname

MAIN_DIRECTORY = dirname(dirname(__file__))

def get_models_path():

    # check if directory exists
    dir = join(MAIN_DIRECTORY, "saved_models")
    if not os.path.isdir(dir):
        os.mkdir(dir)
    return dir