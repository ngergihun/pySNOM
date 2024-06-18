import unittest
import os

import numpy as np

import pySNOM
from pySNOM import NeaImage

class test_NeaImage(unittest.TestCase):
    def test_readfile(self):
        f = 'datasets/image.gwy'
        im = NeaImage()
        im.read_from_gwyfile(os.path.join(pySNOM.__path__[0], f), 'M1A raw')

        np.testing.assert_almost_equal(im.data[7][7], 70.70956420)
        np.testing.assert_almost_equal(im.data[8][8], 70.34189605)
        np.testing.assert_almost_equal(im.data[1][9], 70.69984436)


if __name__ == '__main__':
    unittest.main()
