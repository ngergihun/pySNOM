import unittest
import os

import numpy as np

import pySNOM
from pySNOM import readers, spectra


class test_Neaspectrum(unittest.TestCase):
    def test_pointspectrum_object(self):
        f = "datasets/testspectrum_singlepoint.txt"
        file_reader = readers.NeaSpectralReader(os.path.join(pySNOM.__path__[0], f))
        data, params = file_reader.read()

        s = spectra.NeaSpectrum(data, params)

        np.testing.assert_almost_equal(s.data["O2A"][0], 0.1600194)
        np.testing.assert_string_equal(s.parameters["Scan"], "Fourier Scan")
        np.testing.assert_string_equal(s.scantype, "Point")
        np.testing.assert_equal(np.shape(s.data["O2A"])[0], 2048)

    def test_add_channel(self):
        f = "datasets/testspectrum_singlepoint.txt"
        file_reader = readers.NeaSpectralReader(os.path.join(pySNOM.__path__[0], f))
        data, params = file_reader.read()

        newchannel = np.zeros(np.shape(data["O3A"]))
        s = spectra.NeaSpectrum(data, params)
        s.add_channel(newchannel, "O6A",zerofilling=2)

        np.testing.assert_almost_equal(s.data["O6A"][0], 0)


    def test_multipointspectrum_object(self):
        f = "datasets/testspectrum_multipoint.txt"
        file_reader = readers.NeaSpectralReader(os.path.join(pySNOM.__path__[0], f))
        data, params = file_reader.read()

        s = spectra.NeaSpectrum(data, params)

        np.testing.assert_almost_equal(s.data["O2A"][1, 0, 0], 0.1600194)
        np.testing.assert_string_equal(s.parameters["Scan"], "Fourier Scan")
        np.testing.assert_string_equal(s.scantype, "LineScan")
        np.testing.assert_equal(np.shape(s.data["O2A"])[2], 4)

    def test_transfromations(self):
        f = "datasets/testspectrum_singlepoint.txt"
        file_reader = readers.NeaSpectralReader(os.path.join(pySNOM.__path__[0], f))
        data, params = file_reader.read()
        fref = "datasets/testspectrum_singlepoint_ref.txt"
        file_reader_ref = readers.NeaSpectralReader(
            os.path.join(pySNOM.__path__[0], fref)
        )
        data_ref, params_ref = file_reader_ref.read()

        s = spectra.NeaSpectrum(data, params)
        r = spectra.NeaSpectrum(data_ref, params_ref)

        channel = "O2A"
        normdata = spectra.NormalizeSpectrum(spectra.DataTypes.Amplitude).transform(
            s.data[channel], r.data[channel]
        )
        corrdata = spectra.LinearNormalize(
            wavenumber1=1000, wavenumber2=2200, datatype=spectra.DataTypes.Amplitude
        ).transform(normdata, s.data["Wavenumber"])

        np.testing.assert_almost_equal(normdata[1000], 0.7278023)
        np.testing.assert_almost_equal(corrdata[1000], 0.9795999)

    def test_constant_normalize_from_spectrum_amplitude(self):
        spectrum = np.array([2.0, 4.0, 8.0])
        wnaxis = np.array([900.0, 1000.0, 1100.0])

        # value=1005 selects the nearest wavenumber point at 1000 (index 1).
        normalized = spectra.ConstantNormalize(
            value=1005.0,
            from_spectrum=True,
            datatype=spectra.DataTypes.Amplitude,
        ).transform(spectrum, wnaxis)

        np.testing.assert_allclose(normalized, np.array([0.5, 1.0, 2.0]))

    def test_constant_normalize_from_spectrum_phase(self):
        spectrum = np.array([0.2, 0.5, 1.1])
        wnaxis = np.array([900.0, 1000.0, 1100.0])

        # value=1002 selects the nearest wavenumber point at 1000 (index 1).
        normalized = spectra.ConstantNormalize(
            value=1002.0,
            from_spectrum=True,
            datatype=spectra.DataTypes.Phase,
        ).transform(spectrum, wnaxis)

        np.testing.assert_allclose(normalized, np.array([-0.3, 0.0, 0.6]))

    def test_shift_phase_to_zero_linear_phase(self):
        wnaxis = np.array([900.0, 1000.0, 1100.0, 1200.0])
        # Linear phase with offset and slope; ShiftPhaseToZero should remove both.
        spectrum = np.array([0.0, 0.1, 0.2, 0.3])

        shifted = spectra.ShiftPhaseToZero(
            wavenumber1=1000.0,
            wavenumber2=1200.0,
        ).transform(spectrum, wnaxis)

        np.testing.assert_allclose(shifted, np.zeros_like(shifted), atol=1e-12)

    def test_rotate_phase_wavenumber_scaled_shift(self):
        wnaxis = np.array([500.0, 1000.0, 1500.0])
        spectrum = np.array([0.0, 0.0, 0.0])

        rotated = spectra.RotatePhase(
            degree=90.0,
            wn_ref=1000.0,
            constant_shift=False,
        ).transform(spectrum, wnaxis)

        expected = np.array([np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0])
        np.testing.assert_allclose(rotated, expected, atol=1e-12)

    def test_linear_normalize_phase(self):
        wnaxis = np.array([1000.0, 1500.0, 2000.0])
        # Exactly linear phase; removing the fitted line should return zeros.
        spectrum = np.array([1.0, 1.5, 2.0])

        normalized = spectra.LinearNormalize(
            wavenumber1=1000.0,
            wavenumber2=2000.0,
            datatype=spectra.DataTypes.Phase,
        ).transform(spectrum, wnaxis)

        np.testing.assert_allclose(normalized, np.zeros_like(spectrum), atol=1e-12)

    def test_linear_normalize_amplitude(self):
        wnaxis = np.array([1000.0, 1500.0, 2000.0])
        # Exactly linear amplitude; division by the fitted line should return ones.
        spectrum = np.array([2.0, 3.0, 4.0])

        normalized = spectra.LinearNormalize(
            wavenumber1=1000.0,
            wavenumber2=2000.0,
            datatype=spectra.DataTypes.Amplitude,
        ).transform(spectrum, wnaxis)

        np.testing.assert_allclose(normalized, np.ones_like(spectrum), atol=1e-12)

    def test_cut_transformation(self):
        spectrum = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        wnaxis = np.array([900.0, 1000.0, 1100.0, 1200.0, 1300.0])

        cut_spectrum, cut_wnaxis = spectra.Cut(
            wavenumber1=1000.0,
            wavenumber2=1300.0,
        ).transform(spectrum, wnaxis)

        np.testing.assert_allclose(cut_spectrum, np.array([20.0, 30.0, 40.0]))
        np.testing.assert_allclose(cut_wnaxis, np.array([1000.0, 1100.0, 1200.0]))

    def test_scale_transformation_phase(self):
        spectrum = np.array([0.0, np.pi / 4.0])

        scaled = spectra.Scale(
            factor=2.0,
            datatype=spectra.DataTypes.Phase,
        ).transform(spectrum)

        np.testing.assert_allclose(scaled, np.array([0.0, np.pi / 2.0]), atol=1e-12)

    def test_scale_transformation_amplitude(self):
        spectrum = np.array([1.0, 2.0, 3.0])

        scaled = spectra.Scale(
            factor=3.0,
            datatype=spectra.DataTypes.Amplitude,
        ).transform(spectrum)

        np.testing.assert_allclose(scaled, np.array([3.0, 6.0, 9.0]), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
