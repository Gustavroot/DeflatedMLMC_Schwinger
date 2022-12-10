from gateway import G101,G102
from gateway import G201,G202

import os

# to run :
# python3 main.py




if __name__=='__main__':

    # Schwinger 16^2
    #os.environ['OMP_NUM_THREADS'] = '1'
    #G201() # deflated MLMC
    #G101() # deflated Hutchinson

    # Schwinger 128^2
    os.environ['OMP_NUM_THREADS'] = '1'
    #G202() # deflated MLMC
    G102() # deflated Hutchinson
