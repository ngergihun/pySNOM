import unittest
import os

import numpy as np

import pySNOM
from pySNOM import Reader, Image

class test_NeaImage(unittest.TestCase):
    def test_readfile(self):
        f = 'datasets/testPsHetData.gwy'
        file_reader = Reader(os.path.join(pySNOM.__path__[0], f))
        channeldata = file_reader.read_gwychannel('O3A raw')

        im = Image(channeldata=channeldata)

        np.testing.assert_almost_equal(im.data[7][7], 15.213262557983398)
        np.testing.assert_almost_equal(im.data[8][8], 15.736936569213867)
        np.testing.assert_almost_equal(im.data[1][9], 13.609171867370605)

    def test_readgsf(self):
        f = 'datasets/testPsHet O3A raw.gsf'
        file_reader = Reader(os.path.join(pySNOM.__path__[0], f))
        channeldata = file_reader.read_gsffile()
        im = Image(channeldata=channeldata)
        im.setChannel('O3A raw')

        np.testing.assert_almost_equal(im.data[7][7], 15.213262557983398)
        np.testing.assert_almost_equal(im.data[8][8], 15.736936569213867)
        np.testing.assert_almost_equal(im.data[1][9], 13.609171867370605)

if __name__ == '__main__':
    unittest.main()
