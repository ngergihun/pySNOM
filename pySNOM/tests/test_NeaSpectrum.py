import unittest
import os

import numpy as np

from pySNOM import NeaSpectrum

class test_Neaspectrum(unittest.TestCase):
    def test_readfile(self):
        test_dir = os.path.dirname(__file__)

        neasp = NeaSpectrum()
        neasp.readNeaSpectrum(test_dir+'/sp.txt')

        np.testing.assert_almost_equal(neasp.data['O1A'][0], 129.49686)

        
# if __name__ == '__main__':
#     print('running as main')
#     unittest.main()