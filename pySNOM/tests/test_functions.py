import unittest
import os

import numpy as np

import pySNOM
from pySNOM import Image, Reader


class test_public_functions(unittest.TestCase):
    def test_normalize_simple(self):
        f = 'datasets/image.gwy'
        file_reader = Reader(os.path.join(pySNOM.__path__[0], f))
        channeldata = file_reader.read_gwychannel('M1A raw')

        im = Image(channeldata=channeldata)
        im.setChannel('M1A raw')

        res = im.processor.normalize_simple(datatype=im.datatype)

        np.testing.assert_almost_equal(res[4][8], -0.018852233886)
        np.testing.assert_almost_equal(res[7][5],  2.023254394531)

    def self_reference(self):
        # TODO for testing these we need to have a better test file with actual optical data

        pass


if __name__ == '__main__':
    unittest.main()
