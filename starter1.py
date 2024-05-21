#Main python modules.
#All external packages, none of my stuff.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colorbar as cb

import astropy.io.fits as pyfits
import platform
ver=platform.python_version()
python_version=  int(ver[0])
if python_version == 3:
    from importlib import reload
import sys
import re
import copy
import h5py
import os
import copy
import warnings
import numpy as np
import math
nar = np.array
import pdb
import matplotlib.pyplot as plt
import matplotlib as mpl
import pylab
import time
import glob
#ef=execfile
from collections import defaultdict

import scipy

x_dict = [1,0,0]
y_dict = [2,2,1]
plot_dir = "%s/PigPen"%os.environ['HOME']

