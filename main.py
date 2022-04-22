from gateway import G101
from gateway import G201

import os

# to run :
# python3 main.py



if __name__=='__main__':

    # Schwinger 16^2
    os.environ['OMP_NUM_THREADS'] = '1'
    G201()
    G101()
