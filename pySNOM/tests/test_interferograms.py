import unittest
import os

import numpy as np

import pySNOM
from pySNOM import readers
from pySNOM.interferograms import NeaInterferogram, ProcessSingleChannel, ProcessMultiChannels, ProcessAllPoints

class test_Neaspectrum(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        fdata = 'datasets/testifg_singlepoint.txt'
        data_reader = readers.NeaInterferogramReader(os.path.join(pySNOM.__path__[0], fdata))
        data, measparams = data_reader.read()

        self.ifg = NeaInterferogram(data,measparams,filename=fdata)

    def test_pointinterferogram_object(self):

        np.testing.assert_almost_equal(self.ifg.data['O2A'][100], 9.564073)
        np.testing.assert_string_equal(self.ifg.parameters["Scan"], "Fourier Scan")
        np.testing.assert_string_equal(self.ifg.scantype, "Point")
        np.testing.assert_equal(np.shape(self.ifg.data["O2A"])[0],46080)

    def test_singlechannel_process(self):

        a2, p2, wn2 = ProcessSingleChannel(order=2,simpleoutput=True).transform(self.ifg)

        np.testing.assert_almost_equal(a2[500],65.082045184048)
        np.testing.assert_almost_equal(p2[500],-2.2006946866734887)
        np.testing.assert_almost_equal(wn2[500],1273.6715150986495)

    def test_multichannel_process(self):
        s_multichannel = ProcessMultiChannels(apod=True).transform(self.ifg)

        np.testing.assert_almost_equal(s_multichannel.data["O3A"][500],24.262286272717716)
        np.testing.assert_almost_equal(s_multichannel.data["O3P"][500],0.3032580222572598)
        np.testing.assert_almost_equal(s_multichannel.data["Wavenumber"][500],1273.6715150986495)

    def test_fullauto_process(self):
        s_allpoints = ProcessAllPoints().transform(self.ifg)

        np.testing.assert_almost_equal(s_allpoints.data["O3A"][500],24.262286272717716)
        np.testing.assert_almost_equal(s_allpoints.data["O3P"][500],0.3032580222572598)
        np.testing.assert_almost_equal(s_allpoints.data["Wavenumber"][500],1273.6715150986495)

if __name__ == '__main__':
    unittest.main()