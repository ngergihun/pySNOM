import unittest

import numpy as np

from pySNOM.images import LineLevel, DataTypes


class TestLineLevel(unittest.TestCase):

    def test_median(self):
        d = np.arange(12).reshape(3, -1)[:, [0, 1, 3]]
        l = LineLevel(method="median", datatype=DataTypes.Phase)
        out = l.transform(d)
        np.testing.assert_almost_equal(out, [[-1.,  0.,  2.],
                                             [-1.,  0.,  2.],
                                             [-1.,  0.,  2.]])
        l = LineLevel(method="median", datatype=DataTypes.Amplitude)
        out = l.transform(d)
        np.testing.assert_almost_equal(out, [[0.       , 1.       , 3.       ],
                                             [0.8      , 1.       , 1.4      ],
                                             [0.8888889, 1.       , 1.2222222]])

    def test_mean(self):
        d = np.arange(12).reshape(3, -1)[:, [0, 1, 3]]
        l = LineLevel(method="mean", datatype=DataTypes.Phase)
        out = l.transform(d)
        np.testing.assert_almost_equal(out, [[-1.3333333, -0.3333333,  1.6666667],
                                             [-1.3333333, -0.3333333,  1.6666667],
                                             [-1.3333333, -0.3333333,  1.6666667]])
        l = LineLevel(method="mean", datatype=DataTypes.Amplitude)
        out = l.transform(d)
        np.testing.assert_almost_equal(out, [[0.       , 0.75     , 2.25     ],
                                             [0.75     , 0.9375   , 1.3125   ],
                                             [0.8571429, 0.9642857, 1.1785714]])

    def test_difference(self):
        d = np.arange(12).reshape(3, -1)[:, [0, 1, 3]]
        l = LineLevel(method="difference", datatype=DataTypes.Phase)
        out = l.transform(d)
        np.testing.assert_almost_equal(out, [[-4., -3., -1.],
                                             [ 0.,  1.,  3.]])
        l = LineLevel(method="difference", datatype=DataTypes.Amplitude)
        out = l.transform(d)
        np.testing.assert_almost_equal(out, [[0.       , 0.2      , 0.6      ],
                                             [2.2222222, 2.7777778, 3.8888889]])


if __name__ == '__main__':
    unittest.main()