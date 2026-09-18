"""Classes and helpers for processing nanoFTIR interferogram data.

This module provides :class:`NeaInterferogram`, a container for raw
interferogram channels and their acquisition parameters, transformations for
interpolation and Fourier processing, and :class:`Tools` utilities for
reshaping data and analysing measurement steps.
"""

import numpy as np
import copy
from enum import Enum
from scipy import signal
from scipy.fft import fft, fftshift
from scipy.interpolate import CubicSpline, interp1d
from pySNOM.spectra import NeaSpectrum, SingleChannelSpectrum
import re

MeasurementModes = Enum("MeasurementModes", ["None", "nanoFTIR"])
DataTypes = Enum("DataTypes", ["Amplitude", "Phase", "Topography"])
ChannelTypes = Enum("ChannelTypes", ["None", "Optical", "Mechanical"])
ScanTypes = Enum("ScanTypes", ["Point", "LineScan", "HyperScan"])


# INTERFEROGRAMS ------------------------------------------------------------------------------------------------------------------
class NeaInterferogram(NeaSpectrum):
    """Container for raw nanoFTIR interferograms and acquisition parameters.

    Parameters
    ----------
    data : dict
        Dictionary mapping channel names to raw interferogram arrays. For
        line scans and hyperscans, arrays are reshaped to
        ``(PixelArea[0], PixelArea[1], PixelArea[2] * Averaging)``.
    parameters : dict
        Measurement parameter dictionary containing at least ``"PixelArea"``
        and ``"Averaging"`` entries.
    scantype : str, optional
        Name of a :data:`ScanTypes` member used when parameters do not resolve
        the scan geometry.
    filename : str, optional
        Full path, including name, of the source file.
    mode : str, optional
        Name of a :data:`MeasurementModes` member used when parameters do not
        resolve the measurement mode.

    Attributes
    ----------
    data : dict
        Reshaped interferogram channels.
    filename : str or None
        Source file path.
    mode : str
        Resolved measurement mode name.
    scantype : str
        Resolved scan geometry name.
    parameters : dict
        Measurement parameter dictionary.
    """

    def __init__(
        self, data, parameters, scantype="Point", filename=None, mode="nanoFTIR"
    ):
        super().__init__(
            data, parameters, scantype=scantype, filename=filename, mode=mode
        )
        self.data = data

    @property
    def data(self):
        """Dictionary containing the measurement channels."""
        return self._data

    @data.setter
    def data(self, value):
        """Data setter to reshape properly"""
        self._data = Tools.reshape_ifg_data(value, self._parameters)

    def add_channel(self, values, channelname):
        """Add a channel and reshape it to the interferogram layout.

        Parameters
        ----------
        values : array_like
            Flat channel values with a length matching the measurement
            dimensions.
        channelname : str
            Name under which to store the channel.

        Raises
        ------
        ValueError
            If `channelname` already exists in :attr:`data`.
        """
        if channelname not in list(self._data.keys()):
            self._data[channelname] = np.reshape(
                values,
                (
                    int(self.parameters["PixelArea"][0]),
                    int(self.parameters["PixelArea"][1]),
                    int(self.parameters["PixelArea"][2] * self.parameters["Averaging"]),
                ),
            )
        else:
            raise ValueError


# TRANSFORMATIONS ------------------------------------------------------------------------------------------------------------------
class Transformation:
    """Base class for interferogram transformations."""

    def transform(self, data):
        """Transform interferogram data.

        Parameters
        ----------
        data : array_like
            Interferogram data to transform.

        Raises
        ------
        NotImplementedError
            Always, unless overridden by a subclass.
        """
        raise NotImplementedError()


class ProcessInterferogram(Transformation):
    """Fourier transform one interpolated interferogram.

    Parameters
    ----------
    apod : bool, optional
        Apply an asymmetric apodization window before the Fourier transform.
    windowtype : str, optional
        Name of the window function in ``scipy.signal.windows``.
    nzeros : int, optional
        Zero-filling factor applied to the interferogram length.
    wlpidx : int, optional
        Index of the white-light position. If omitted, use the largest
        absolute interferogram value.
    """

    def __init__(self, apod=True, windowtype="blackmanharris", nzeros=4, wlpidx=None):
        self.apod = apod
        self.nzeros = nzeros
        self.wlpidx = wlpidx
        self.windowtype = windowtype

    def transform(self, ifg, maxis):
        """Calculate the positive-frequency complex spectrum.

        Parameters
        ----------
        ifg : array_like
            One-dimensional, uniformly sampled interferogram.
        maxis : array_like
            Mirror-position axis corresponding to `ifg`, in metres.

        Returns
        -------
        complex ndarray
            Positive-frequency complex spectrum after centering and FFT.
        ndarray
            Wavenumber axis corresponding to the returned spectrum, in
            inverse centimetres.
        """
        # Find the location index of the WLP
        if self.wlpidx is None:
            self.wlpidx = np.argmax(np.abs(ifg))
        # Create apodization window
        if self.apod:
            w = Tools.asymmetric_window(
                npoints=len(ifg), centerindex=self.wlpidx, windowtype=self.windowtype
            )
        else:
            w = np.ones(np.shape(ifg))
        ifg = ifg - np.mean(ifg)
        # Calculate FFT
        complex_spectrum = fftshift(fft(ifg * w, self.nzeros * len(ifg)))
        # Calculate frequency axis
        stepsizes = np.median(np.diff(maxis * 1e6))
        Fs = 1 / np.mean(stepsizes)
        faxis = (Fs / 2) * np.linspace(-1, 1, len(complex_spectrum)) * 10000 / 2
        return (
            complex_spectrum[int(len(faxis) / 2) :],
            faxis[int(len(faxis) / 2) :],
        )


class InterpolateInterferogram(Transformation):
    """Reinterpolate raw interferograms to a uniform mirror-position grid.

    Parameters
    ----------
    method : {"spline", "linear"}, optional
        Interpolation method. ``"spline"`` uses
        :class:`scipy.interpolate.CubicSpline`; ``"linear"`` uses
        :func:`scipy.interpolate.interp1d`.
    """

    def __init__(self, method="spline"):
        self.method = method

    def transform(self, ifg, maxis):
        """Interpolate one or more interferograms.

        Parameters
        ----------
        ifg : ndarray
            One- or two-dimensional interferogram data. Rows in a 2-D array
            represent separate interferograms.
        maxis : ndarray
            Mirror-position axis or axes corresponding to `ifg`.

        Returns
        -------
        tuple of ndarray
            Interpolated interferograms and their uniform mirror-position
            axis. Inputs with unsupported dimensionality are returned
            unchanged.
        """
        # if np.iscomplex(ifg).any():
        newifg = np.zeros(np.shape(ifg)) * complex(1j)
        # else:
        # newifg = np.zeros(np.shape(ifg))

        newmaxis = np.zeros(np.shape(maxis))

        match self.method:
            case "spline":
                interp_object = CubicSpline
            case "linear":
                interp_object = interp1d

        # in case of processing multiple interferograms in a 2d array we have to take the median
        # sometimes the point locations has large jumps at the beginning (neaspec machine artifact)
        startM = np.min(np.nanmedian(maxis, axis=0))
        stopM = np.max(np.nanmedian(maxis, axis=0))

        if ifg.ndim == 1:
            newcoords = np.linspace(startM, stopM, num=len(maxis))
            if np.iscomplex(ifg).any():
                interpR = interp_object(maxis, np.real(ifg))
                interpI = interp_object(maxis, np.imag(ifg))
                newifgR = interpR(newcoords)
                newifgI = interpI(newcoords)
                newifg = newifgR + newifgI * complex(1j)
            else:
                interpifg = interp_object(maxis, ifg)
                newifg = interpifg(newcoords)
            return newifg, newcoords
        elif ifg.ndim == 2:
            newcoords = np.linspace(startM, stopM, num=np.shape(maxis)[1])
            for i in range(np.shape(ifg)[0]):
                if np.iscomplex(ifg[i][:]).any():
                    interpR = interp_object(maxis[i][:], np.real(ifg[i][:]))
                    interpI = interp_object(maxis[i][:], np.imag(ifg[i][:]))
                    newifgR = interpR(newcoords)
                    newifgI = interpI(newcoords)
                    newifg[i][:] = newifgR + newifgI * complex(1j)
                    newmaxis[i][:] = newcoords
                else:
                    interpifg = interp_object(maxis[i][:], ifg[i][:])
                    newifg[i][:] = interpifg(newcoords)
                    newmaxis[i][:] = newcoords
            return newifg, newmaxis
        else:
            return ifg, maxis


class ProcessSingleChannel(Transformation):
    """Process one optical demodulation channel into a spectrum.

    Parameters
    ----------
    order : int
        Optical demodulation order. Channels ``O{order}A`` and ``O{order}P``
        are read from the interferogram.
    method : {"abs", "real", "imag", "complex", "simple"}, optional
        Representation used to combine amplitude and phase channels.
    apod : bool, optional
        Apply asymmetric apodization before the Fourier transform.
    windowtype : str, optional
        Name of the SciPy window function used for apodization.
    nzeros : int, optional
        Zero-filling factor for the Fourier transform.
    interpmethod : {"spline", "linear"}, optional
        Method used to make the mirror-position grid uniform.
    simpleoutput : bool, optional
        If true, return arrays instead of a :class:`NeaSpectrum`.
    """

    def __init__(
        self,
        order,
        method="complex",
        apod=True,
        windowtype="blackmanharris",
        nzeros=4,
        interpmethod="spline",
        simpleoutput=False,
    ):
        self.order = order
        self.method = method
        self.windowtype = windowtype
        self.apod = apod
        self.nzeros = nzeros
        self.interpmethod = interpmethod
        self.simpleoutput = simpleoutput

    def transform(self, neaifg):  # Load amplitude and phase of the given channel
        """Process the selected channel of `neaifg`.

        Parameters
        ----------
        neaifg : NeaInterferogram
            Interferogram containing amplitude, phase, and mirror-position
            channels for the requested order.

        Returns
        -------
        tuple of ndarray or NeaSpectrum
            If `simpleoutput` is true, return ``(amplitude, phase,
            wavenumber)``. Otherwise return the processed data as a
            :class:`pySNOM.spectra.NeaSpectrum`.
        """
        # Calculate the interferogram to process based on the given method

        channelA = f"O{self.order}A"
        channelP = f"O{self.order}P"

        ifgA = np.reshape(
            neaifg.data[channelA],
            (neaifg.parameters["Averaging"], neaifg.parameters["PixelArea"][2]),
        )
        ifgP = np.reshape(
            neaifg.data[channelP],
            (neaifg.parameters["Averaging"], neaifg.parameters["PixelArea"][2]),
        )
        Maxis = np.reshape(
            neaifg.data["M"],
            (neaifg.parameters["Averaging"], neaifg.parameters["PixelArea"][2]),
        )

        match self.method:
            case "abs":
                IFG = np.abs(ifgA * np.exp(ifgP * complex(1j)))
            case "real":
                IFG = np.real(ifgA * np.exp(ifgP * complex(1j)))
            case "imag":
                IFG = np.imag(ifgA * np.exp(ifgP * complex(1j)))
            case "complex":
                IFG = ifgA * np.exp(ifgP * complex(1j))
            case "simple":
                IFG = ifgA

        #  Interpolate
        IFG, Maxis = InterpolateInterferogram(method=self.interpmethod).transform(
            IFG, Maxis
        )

        # PROCESS IFGs
        # Check if it is multiple interferograms or just a single one
        if len(np.shape(IFG)) == 1:
            complex_spectrum, f = ProcessInterferogram(
                apod=self.apod, windowtype=self.windowtype, nzeros=self.nzeros
            ).transform(IFG, Maxis)
            amp = np.abs(complex_spectrum)
            phi = np.angle(complex_spectrum)
        else:
            # Allocate variables
            spectraAll = complex(1j) * np.zeros(
                (np.shape(IFG)[0], int(self.nzeros * np.shape(IFG)[1] / 2))
            )
            fAll = np.zeros(np.shape(spectraAll))
            # Go trough all
            for i in range(np.shape(IFG)[0]):
                spectraAll[i, :], fAll[i, :] = ProcessInterferogram(
                    apod=self.apod, windowtype=self.windowtype, nzeros=self.nzeros
                ).transform(IFG[i, :], Maxis[i, :])
            # Average the complex spectra
            complex_spectrum = np.mean(spectraAll, axis=0)
            # Extract amplitude and phase from the averaged complex spectrum
            amp = np.abs(complex_spectrum)
            phi = np.angle(complex_spectrum)
            f = np.mean(fAll, axis=0)

        if self.simpleoutput:
            return amp, phi, f
        else:
            spectrum_data = {}
            spectrum_parameters = copy.deepcopy(neaifg.parameters)
            spectrum_parameters["ScanArea"] = [
                neaifg.parameters["ScanArea"][0],
                neaifg.parameters["ScanArea"][1],
                len(amp),
            ]
            spectrum_data[channelA] = amp
            spectrum_data[channelP] = phi
            spectrum_data["Wavenumber"] = f
            spectrum = NeaSpectrum(spectrum_data, spectrum_parameters)

            return spectrum


class ProcessMultiChannels(Transformation):
    """Process optical channels for demodulation orders zero through five.

    Parameters
    ----------
    method : {"abs", "real", "imag", "complex", "simple"}, optional
        Representation used for each amplitude/phase channel pair.
    apod : bool, optional
        Apply asymmetric apodization before each Fourier transform.
    windowtype : str, optional
        Name of the SciPy window function used for apodization.
    nzeros : int, optional
        Zero-filling factor for each Fourier transform.
    interpmethod : {"spline", "linear"}, optional
        Method used to make the mirror-position grid uniform.
    simpleoutput : bool, optional
        Intended to select array output; when false, return a
        :class:`NeaSpectrum`.
    """

    def __init__(
        self,
        method="complex",
        apod=True,
        windowtype="blackmanharris",
        nzeros=4,
        interpmethod="spline",
        simpleoutput=False,
    ):
        self.method = method
        self.apod = apod
        self.windowtype = windowtype
        self.nzeros = nzeros
        self.interpmethod = interpmethod
        self.simpleoutput = simpleoutput

    def transform(self, neaifg):
        """Process all supported optical channels in `neaifg`.

        Parameters
        ----------
        neaifg : NeaInterferogram
            Interferogram containing optical amplitude and phase channels.

        Returns
        -------
        NeaSpectrum
            Spectrum containing amplitude and phase channels and a
            ``"Wavenumber"`` axis.
        """
        spectrum_data = {}
        spectrum_parameters = copy.deepcopy(neaifg.parameters)

        for order in range(6):
            channelA = f"O{order}A"
            channelP = f"O{order}P"

            amp, phi, f = ProcessSingleChannel(
                order,
                method=self.method,
                apod=self.apod,
                windowtype=self.windowtype,
                nzeros=self.nzeros,
                interpmethod=self.interpmethod,
                simpleoutput=True,
            ).transform(neaifg)

            spectrum_data[channelA] = amp
            spectrum_data[channelP] = phi
            spectrum_data["Wavenumber"] = f

        if self.simpleoutput:
            spectrum_data
        else:
            spectrum_parameters["ScanArea"] = [
                neaifg.parameters["ScanArea"][0],
                neaifg.parameters["ScanArea"][1],
                len(amp),
            ]
            spectrum = NeaSpectrum(spectrum_data, spectrum_parameters)
            return spectrum


class ProcessAllPoints(Transformation):
    """Process every spatial point in an interferogram scan.

    Parameters
    ----------
    method : {"abs", "real", "imag", "complex", "simple"}, optional
        Representation used to combine amplitude and phase channels.
    apod : bool, optional
        Apply asymmetric apodization before each Fourier transform.
    windowtype : str, optional
        Name of the SciPy window function used for apodization.
    nzeros : int, optional
        Zero-filling factor for each Fourier transform.
    interpmethod : {"spline", "linear"}, optional
        Method used to make the mirror-position grid uniform.
    """

    def __init__(
        self,
        method="complex",
        apod=True,
        windowtype="blackmanharris",
        nzeros=4,
        interpmethod="spline",
    ):
        self.method = method
        self.apod = apod
        self.windowtype = windowtype
        self.nzeros = nzeros
        self.interpmethod = interpmethod

    def transform(self, neaifg):
        """Process all available optical channels at every scan point.

        Parameters
        ----------
        neaifg : NeaInterferogram
            Point, line-scan, or hyperscan interferogram.

        Returns
        -------
        NeaSpectrum
            Spectrum data with spatial dimensions preserved and processed
            amplitude, phase, and wavenumber channels.
        """
        if (
            neaifg.parameters["PixelArea"][0] == 1
            and neaifg.parameters["PixelArea"][1] == 1
        ):
            spectra = ProcessMultiChannels(
                method=self.method,
                windowtype=self.windowtype,
                nzeros=self.nzeros,
                apod=self.apod,
                interpmethod=self.interpmethod,
                simpleoutput=False,
            ).transform(neaifg)
        else:
            pixel_area = [
                neaifg.parameters["PixelArea"][0],
                neaifg.parameters["PixelArea"][1],
                int(self.nzeros * neaifg.parameters["PixelArea"][2] / 2),
            ]

            ampFullData = np.zeros((pixel_area[0], pixel_area[1], pixel_area[2]))
            phiFullData = np.zeros(np.shape(ampFullData))
            fFullData = np.zeros(np.shape(ampFullData))

            pointifg_data = dict()
            pointifg_params = dict()
            pointifg_params["PixelArea"] = [1, 1, neaifg.parameters["PixelArea"][2]]
            pointifg_params["Scan"] = "Fourier Scan"
            pointifg_params["Averaging"] = neaifg.parameters["Averaging"]

            spectra_params = dict()
            spectra_params["Scan"] = "Fourier Scan"
            spectra_params["PixelArea"] = pixel_area
            spectra = NeaSpectrum({}, spectra_params, scantype=neaifg.scantype)

            allchannels = list(neaifg.data.keys())
            optical_channels = [
                name
                for name in allchannels
                if re.match("O(.?)A", name) or re.match("O(.?)P", name)
            ]
            orders = [int(n) for c in optical_channels for n in re.findall(r"\d", c)]
            orders = np.unique(np.asarray(orders))

            for order in orders:
                channelA = f"O{order}A"
                channelP = f"O{order}P"

                if channelA not in list(neaifg.data.keys()) or channelP not in list(
                    neaifg.data.keys()
                ):
                    print(
                        f"Skipped processing for order: {order}, since A or P is missing!"
                    )
                    continue
                else:
                    for i in range(pixel_area[0]):
                        for k in range(pixel_area[1]):
                            pointifg_data[channelA] = neaifg.data[channelA][i, k, :]
                            pointifg_data[channelP] = neaifg.data[channelP][i, k, :]
                            pointifg_data["M"] = neaifg.data["M"][i, k, :]
                            pointifg = NeaInterferogram(pointifg_data, pointifg_params)
                            (
                                ampFullData[i, k, :],
                                phiFullData[i, k, :],
                                fFullData[i, k, :],
                            ) = ProcessSingleChannel(
                                order,
                                method=self.method,
                                apod=self.apod,
                                windowtype=self.windowtype,
                                nzeros=self.nzeros,
                                interpmethod=self.interpmethod,
                                simpleoutput=True,
                            ).transform(
                                pointifg
                            )

                    spectra.data[channelA] = ampFullData
                    spectra.data[channelP] = phiFullData
                    spectra.data["Wavenumber"] = fFullData

        return spectra


# TOOLS ------------------------------------------------------------------------------------------------------------------
class Tools:
    """Utility methods for interferogram reshaping and analysis."""

    def __init__(self):
        pass

    @staticmethod
    def reshape_ifg_data(data, params):
        """Reshape raw interferogram channels for spatial scans.

        Parameters
        ----------
        data : dict
            Channel names mapped to flat arrays. The dictionary is modified
            in place for multi-point data.
        params : dict
            Measurement parameters containing ``"PixelArea"`` and
            ``"Averaging"``.

        Returns
        -------
        dict
            The input dictionary, with multi-point channels reshaped to
            ``(PixelArea[0], PixelArea[1], PixelArea[2] * Averaging)``.
        """
        if params["PixelArea"][1] != 1 or params["PixelArea"][0] != 1:
            for channel in list(data.keys()):
                data[channel] = np.reshape(
                    data[channel],
                    (
                        int(params["PixelArea"][0]),
                        int(params["PixelArea"][1]),
                        int(params["PixelArea"][2] * params["Averaging"]),
                    ),
                )
            return data
        else:
            return data

    @staticmethod
    def reshape_linescan_interferogram(data, parameters):
        """Reshape line-scan data into spatial points and interferogram depth.

        Parameters
        ----------
        data : array_like
            Raw line-scan values.
        parameters : dict
            Measurement parameters containing ``"PixelArea"``.

        Returns
        -------
        ndarray
            Array with shape ``(PixelArea[0], PixelArea[2])``.
        """
        return np.reshape(
            np.ravel(data), (parameters["PixelArea"][0], parameters["PixelArea"][2])
        )

    @staticmethod
    def asymmetric_window(npoints, centerindex=None, windowtype="blackmanharris"):
        """Construct an asymmetric window centered at a white-light position.

        Parameters
        ----------
        npoints : int
            Number of samples in the output window.
        centerindex : int, optional
            Index of the window center. Defaults to the midpoint.
        windowtype : str, optional
            Name of a window function in ``scipy.signal.windows``.

        Returns
        -------
        ndarray
            Window with `npoints` samples.
        """
        if centerindex is None:
            centerindex = int(len(windowPart2) / 2)

        # Calculate the length of the two sides
        length1 = (centerindex) * 2
        length2 = (npoints - centerindex) * 2

        # Generate the two parts of the window

        windowfunc = getattr(signal.windows, windowtype)
        windowPart1 = windowfunc(length1)
        windowPart2 = windowfunc(length2)

        # Construct the asymetric window from the two sides
        asymWindow1 = windowPart1[0 : int(len(windowPart1) / 2)]
        if npoints % 2 == 0:
            asymWindow2 = windowPart2[int(len(windowPart2) / 2) : int(len(windowPart2))]
        else:
            asymWindow2 = windowPart2[int(len(windowPart2) / 2) : int(len(windowPart2))]

        return np.concatenate((asymWindow1, asymWindow2))

    @staticmethod
    def analyse_steps(maxis):
        """Calculate mean step sizes and their spread for mirror positions.

        Parameters
        ----------
        maxis : ndarray
            Two-dimensional array whose rows contain mirror-position axes.

        Returns
        -------
        tuple of ndarray
            Mean step sizes and standard deviations of the step sizes, each
            with shape ``(number_of_rows, 1)``.
        """
        stepsize = np.zeros((np.shape(maxis)[0], 1))
        stepspread = np.zeros((np.shape(maxis)[0], 1))
        for i in range(np.shape(maxis)[0]):
            stepsize[i] = np.mean(np.diff(maxis[i, :]))
            stepspread[i] = np.std(np.diff(maxis[i, :]))

        return stepsize, stepspread
