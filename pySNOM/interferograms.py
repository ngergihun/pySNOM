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
    def __init__(
        self, data, parameters, scantype="Point", filename=None, mode="nanoFTIR"
    ):
        super().__init__(
            data, parameters, scantype=scantype, filename=filename, mode=mode
        )
        self.data = data

    @property
    def data(self):
        """Property - data (dict with measurement channels)"""
        return self._data

    @data.setter
    def data(self, value):
        """Data setter to reshape properly"""
        self._data = Tools.reshape_ifg_data(value, self._parameters)

    def add_channel(self, values, channelname):
        """Adds a new channel to data dictionary"""
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
    def transform(self, data):
        raise NotImplementedError()


class ProcessInterferogram(Transformation):
    def __init__(self, apod=True, windowtype="blackmanharris", nzeros=4, wlpidx=None):
        self.apod = apod
        self.nzeros = nzeros
        self.wlpidx = wlpidx
        self.windowtype = windowtype

    def transform(self, ifg, maxis):
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
    """Reinterpolates the raw interferograms to a uniform grid."""

    def __init__(self, method="spline"):
        self.method = method

    def transform(self, ifg, maxis):
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
    def __init__(self):
        pass

    @staticmethod
    def reshape_ifg_data(data, params):
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
        return np.reshape(
            np.ravel(data), (parameters["PixelArea"][0], parameters["PixelArea"][2])
        )

    @staticmethod
    def asymmetric_window(npoints, centerindex=None, windowtype="blackmanharris"):
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
        stepsize = np.zeros((np.shape(maxis)[0], 1))
        stepspread = np.zeros((np.shape(maxis)[0], 1))
        for i in range(np.shape(maxis)[0]):
            stepsize[i] = np.mean(np.diff(maxis[i, :]))
            stepspread[i] = np.std(np.diff(maxis[i, :]))

        return stepsize, stepspread
