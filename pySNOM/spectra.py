"""Classes and helpers for handling nanoFTIR/PsHet spectral measurement data.

This module provides the :class:`NeaSpectrum` and :class:`SingleChannelSpectrum`
containers for spectral measurement data and parameters, a family of
:class:`Transformation` subclasses implementing common spectral processing
steps (cutting, scaling, normalizing, phase rotation and levelling), and the
:class:`Tools` helper class used to reshape raw spectral data arrays.
"""

import numpy as np
from enum import Enum
from pySNOM.images import type_from_channelname
from pySNOM.defaults import Defaults

#: Supported spectral measurement modes.
MeasurementModes = Enum(
    "MeasurementModes", ["None", "nanoFTIR", "PsHet", "PTE", "nanoRaman"]
)
#: Kinds of data a spectral channel can hold.
DataTypes = Enum("DataTypes", ["Amplitude", "Phase", "Complex", "Topography"])
#: Whether a channel is an optical or mechanical measurement channel.
ChannelTypes = Enum("ChannelTypes", ["None", "Optical", "Mechanical"])
#: Supported spectral scan geometries.
ScanTypes = Enum("ScanTypes", ["Point", "LineScan", "HyperScan"])
ScanTypes = Enum("ScanTypes", ["Point", "LineScan", "HyperScan"])


class NeaSpectrum:
    """Container for full spectral measurement data and its acquisition parameters.

    Parameters
    ----------
    data : dict
        Dictionary mapping channel names to their raw spectral data arrays.
        The arrays are reshaped in-place (see :meth:`Tools.reshape_spectrum_data`)
        according to `parameters`.
    parameters : dict
        Measurement parameter dictionary as read from the info file (e.g.
        containing ``"PixelArea"`` and ``"Scan"`` entries). When provided and
        truthy, `scantype` and `mode` are derived from it instead of the
        given arguments.
    scantype : str, optional
        Name of a :data:`ScanTypes` member describing the scan geometry,
        used only when `parameters` is empty/``None``.
    filename : str, optional
        Full path (with name) of the file the spectrum was loaded from.
    mode : str, optional
        Name of a :data:`MeasurementModes` member, used only when
        `parameters` is empty/``None``.

    Attributes
    ----------
    filename : str or None
        Full path with name of the source file.
    data : dict
        Reshaped measurement channel data.
    mode : str
        Name of the resolved :data:`MeasurementModes` member.
    scantype : str
        Name of the resolved :data:`ScanTypes` member.
    parameters : dict
        Measurement parameter dictionary.

    Raises
    ------
    ValueError
        If `mode` or `scantype` is not a valid enum member name, or if
        `parameters` is provided but does not contain the expected keys.
    """

    def __init__(
        self,
        data: dict,
        parameters: dict,
        scantype="Point",
        filename=None,
        mode="nanoFTIR",
    ):
        self.filename = filename  # Full path with name
        self._parameters = parameters
        self.data = data

        # set measurement mode
        try:
            self._mode = MeasurementModes[mode]
        except ValueError:
            self._mode = MeasurementModes["nanoFTIR"]
            raise ValueError(mode + "is not a measurement mode!")

        try:
            self._scantype = ScanTypes[scantype]
        except ValueError:
            self._scantype = ScanTypes["Point"]
            raise ValueError(scantype + "is not a measurement mode!")

        if parameters:
            try:
                if parameters["PixelArea"][1] == 1 and parameters["PixelArea"][0] == 1:
                    self._scantype = ScanTypes["Point"]
                elif parameters["PixelArea"][1] == 1 or parameters["PixelArea"][0] == 1:
                    self._scantype = ScanTypes["LineScan"]
                else:
                    self._scantype = ScanTypes["HyperScan"]
                self._mode = MeasurementModes[
                    Defaults().spectral_mode_defs[parameters["Scan"]]
                ]
            except:
                raise ValueError("Parameters dictionary is not valid!")
        else:
            # set measurement mode
            try:
                self._mode = MeasurementModes[mode]
            except ValueError:
                self._mode = MeasurementModes["nanoFTIR"]
                raise ValueError(mode + "is not a measurement mode!")
            # set scan type
            try:
                self._scantype = ScanTypes[scantype]
            except ValueError:
                self._scantype = ScanTypes["Point"]
                raise ValueError(scantype + "is not a measurement mode!")

    @property
    def data(self):
        """Property - data (dict with measurement channels)"""
        return self._data

    @data.setter
    def data(self, value):
        """Data setter to reshape properly"""
        self._data = Tools.reshape_spectrum_data(value, self._parameters)

    @property
    def mode(self):
        """Property - mode (MeasurementMode Enum name)"""
        return self._mode.name

    @property
    def scantype(self):
        """Property - scantype (ScanType Enum name)"""
        return self._scantype.name

    @property
    def parameters(self):
        """Property - scantype (ScanType Enum name)"""
        return self._parameters

    def add_channel(self, values, channelname, zerofilling=1):
        """Add a new channel to the data dictionary.

        The flat `values` array is reshaped to match the pixel area recorded
        in :attr:`parameters`, taking `zerofilling` into account for the
        spectral axis length.

        Parameters
        ----------
        values : array_like
            Flat array of channel values to store, whose length must equal
            the product of the pixel area dimensions (scaled by
            `zerofilling` along the spectral axis).
        channelname : str
            Name under which the channel is stored in :attr:`data`. Must not
            already exist in the data dictionary.
        zerofilling : int, optional
            Zero-filling factor applied to the spectral axis length when
            reshaping `values`.

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
                    int(self.parameters["PixelArea"][2]*zerofilling),
                ),
            )
        else:
            raise ValueError

class SingleChannelSpectrum(NeaSpectrum):
    """A single measurement channel paired with its wavenumber axis.

    Parameters
    ----------
    data : dict
        Dictionary of measurement channel data, as accepted by
        :class:`NeaSpectrum`.
    wndata : array_like
        Wavenumber axis values corresponding to the spectral data.
    filename : str, optional
        Full path (with name) of the file the spectrum was loaded from.
    parameters : dict, optional
        Measurement parameter dictionary, as accepted by :class:`NeaSpectrum`.
    mode : str, optional
        Name of a :data:`MeasurementModes` member.
    channelname : str, optional
        Name of the channel represented by this instance. Determines
        :attr:`channeltype`, :attr:`order` and :attr:`datatype` via
        :func:`pySNOM.images.type_from_channelname`.

    Attributes
    ----------
    wndata : array_like
        Wavenumber axis values corresponding to the spectral data.
    channel : str
        Name of the represented channel.
    channeltype : ChannelTypes
        Optical/mechanical classification of the channel.
    order : int
        Demodulation order extracted from the channel name.
    datatype : DataTypes
        Data kind (amplitude, phase, ...) extracted from the channel name.
    """

    def __init__(
        self,
        data,
        wndata,
        filename=None,
        parameters=None,
        mode="nanoFTIR",
        channelname="O2A",
    ):
        super().__init__(data, filename, parameters, mode)
        self._wndata = wndata
        self.channel = channelname
        self.order = 0
        self.datatype = None

    @property
    def data(self):
        """Property - data (dict with measurement channels)"""
        return self._data

    @property
    def wndata(self):
        """Property - wavenumber data"""
        return self._wndata

    @property
    def channel(self):
        """Property - channel (string)"""
        return self._channel

    @channel.setter
    def channel(self, value):
        self._channel = value
        self.channeltype, self.order, self.datatype = type_from_channelname(value)


# TRANSFORMATIONS ------------------------------------------------------------------------------------------------------------------
class Transformation:
    """Base class for spectral data transformations.

    Subclasses implement :meth:`transform` to process a spectrum (and,
    typically, its wavenumber axis) and return the transformed result.
    """

    def transform(self, data):
        """Transform the given spectral data.

        Parameters
        ----------
        data : array_like
            Spectral data to transform.

        Returns
        -------
        array_like
            The transformed spectral data.

        Raises
        ------
        NotImplementedError
            Always, unless overridden by a subclass.
        """
        raise NotImplementedError()

class Cut(Transformation):
    """Crop a spectrum to the wavenumber range [`wavenumber1`, `wavenumber2`].

    Parameters
    ----------
    wavenumber1 : float, optional
        Lower bound of the wavenumber range to keep.
    wavenumber2 : float, optional
        Upper bound of the wavenumber range to keep.

    Attributes
    ----------
    wn1 : float
        Lower bound of the wavenumber range to keep.
    wn2 : float
        Upper bound of the wavenumber range to keep.
    """

    def __init__(self, wavenumber1=0.0, wavenumber2=1000.0):
        self.wn1 = wavenumber1
        self.wn2 = wavenumber2

    def transform(self, spectrum, wnaxis):
        """Crop `spectrum` and `wnaxis` to the configured wavenumber range.

        Parameters
        ----------
        spectrum : array_like
            Spectral data values.
        wnaxis : array_like
            Wavenumber axis values corresponding to `spectrum`.

        Returns
        -------
        tuple of array_like
            The cropped `(spectrum, wnaxis)` pair, restricted to the indices
            closest to `wn1` and `wn2`.
        """
        wn1idx = np.argmin(abs(wnaxis - self.wn1))
        wn2idx = np.argmin(abs(wnaxis - self.wn2))
        return spectrum[wn1idx:wn2idx], wnaxis[wn1idx:wn2idx]
    
class Scale(Transformation):
    """Scale spectral data by a constant factor, or rotate phase by it.

    Parameters
    ----------
    factor : float, optional
        Multiplicative scale factor. For phase data, interpreted as a
        rotation applied via complex exponentiation.
    datatype : DataTypes, optional
        Kind of data being scaled; :data:`DataTypes.Phase` is handled via
        complex rotation, other types via plain multiplication.

    Attributes
    ----------
    factor : float
        Multiplicative scale factor.
    datatype : DataTypes
        Kind of data being scaled.
    """

    def __init__(self, factor=1.0, datatype=DataTypes.Phase):
        self.factor = factor
        self.datatype = datatype

    def transform(self, spectrum):
        """Scale `spectrum` by :attr:`factor`.

        Parameters
        ----------
        spectrum : array_like
            Spectral data values. For :data:`DataTypes.Phase`, may be real
            (radians) or complex; real values are converted to complex
            before rotation.

        Returns
        -------
        array_like
            The scaled (or phase-rotated) spectrum.
        """
        if self.datatype == DataTypes.Phase:
            if not np.iscomplex(spectrum).any():
                spectrum = np.exp(spectrum * self.factor * complex(1j))
            return np.angle(spectrum)
        else:
            return spectrum * self.factor


class LinearNormalize(Transformation):
    """Subtract (or divide by) a linear baseline defined by two reference points.

    The baseline is the line through `(wavenumber1, spectrum[wn1])` and
    `(wavenumber2, spectrum[wn2])`.

    Parameters
    ----------
    wavenumber1 : float, optional
        First reference wavenumber defining the baseline.
    wavenumber2 : float, optional
        Second reference wavenumber defining the baseline.
    datatype : DataTypes, optional
        Kind of data being normalized; :data:`DataTypes.Amplitude` divides
        by the baseline, other types subtract it.

    Attributes
    ----------
    wn1 : float
        First reference wavenumber defining the baseline.
    wn2 : float
        Second reference wavenumber defining the baseline.
    datatype : DataTypes
        Kind of data being normalized.
    """

    def __init__(self, wavenumber1=0.0, wavenumber2=1000.0, datatype=DataTypes.Phase):
        self.wn1 = wavenumber1
        self.wn2 = wavenumber2
        self.datatype = datatype

    def transform(self, spectrum, wnaxis):
        """Remove the linear baseline from `spectrum`.

        Parameters
        ----------
        spectrum : array_like
            Spectral data values.
        wnaxis : array_like
            Wavenumber axis values corresponding to `spectrum`.

        Returns
        -------
        array_like
            The baseline-corrected spectrum, divided by the baseline for
            :data:`DataTypes.Amplitude` data, otherwise with the baseline
            subtracted.
        """
        wn1idx = np.argmin(abs(wnaxis - self.wn1))
        wn2idx = np.argmin(abs(wnaxis - self.wn2))
        m = (spectrum[wn2idx] - spectrum[wn1idx]) / (wnaxis[wn2idx] - wnaxis[wn1idx])
        C = spectrum[wn1idx] - m * wnaxis[wn1idx]

        if self.datatype == DataTypes.Amplitude:
            return spectrum / (m * wnaxis + C)
        else:
            return spectrum - (m * wnaxis + C)

class ConstantNormalize(Transformation):
    """Subtract (or divide by) a constant reference value.

    The reference value can be a fixed constant or read from the spectrum
    itself at a given wavenumber.

    Parameters
    ----------
    value : float, optional
        Constant reference value, or the reference wavenumber when
        `from_spectrum` is ``True``.
    from_spectrum : bool, optional
        If ``True``, the reference value is taken from `spectrum` at the
        point closest to `value` (interpreted as a wavenumber) rather than
        used directly as the reference value.
    datatype : DataTypes, optional
        Kind of data being normalized; :data:`DataTypes.Amplitude` divides
        by the reference value, other types subtract it.

    Attributes
    ----------
    value : float
        Constant reference value or reference wavenumber.
    from_spectrum : bool
        Whether the reference value is read from the spectrum.
    datatype : DataTypes
        Kind of data being normalized.
    """

    def __init__(self, value=1.0, from_spectrum=False, datatype=DataTypes.Phase):
        self.value = value
        self.from_spectrum = from_spectrum
        self.datatype = datatype

    def transform(self, spectrum, wnaxis):
        """Subtract or divide `spectrum` by the configured reference value.

        Parameters
        ----------
        spectrum : array_like
            Spectral data values.
        wnaxis : array_like
            Wavenumber axis values corresponding to `spectrum`, used only
            when :attr:`from_spectrum` is ``True``.

        Returns
        -------
        array_like
            The normalized spectrum, divided by the reference value for
            :data:`DataTypes.Amplitude` data, otherwise with the reference
            value subtracted.
        """
        if self.from_spectrum:
            data_idx = np.argmin(np.abs(wnaxis-self.value))
            data_value = spectrum[data_idx]
        else:
            data_value = self.value

        if self.datatype == DataTypes.Amplitude:
            return spectrum / data_value
        else:
            return spectrum - data_value

class RotatePhase(Transformation):
    """Rotate the phase of a spectrum (equal to shifting), either linearly with wavenumber or by a constant.

    Parameters
    ----------
    degree : float, optional
        Rotation angle in degrees. When `constant_shift` is ``False``, this
        is the rotation applied at `wn_ref` and scales linearly with
        wavenumber; when ``True``, it is applied uniformly across the axis.
    wn_ref : float, optional
        Reference wavenumber at which `degree` applies, used only when
        `constant_shift` is ``False``.
    constant_shift : bool, optional
        If ``True``, apply a constant phase shift of `degree` instead of a
        wavenumber-dependent rotation.

    Attributes
    ----------
    wn_ref : float
        Reference wavenumber for the wavenumber-dependent rotation.
    degree : float
        Rotation angle in degrees.
    constant_shift : bool
        Whether a constant (as opposed to wavenumber-dependent) shift is applied.
    """

    def __init__(self, degree=0.0, wn_ref=1000.0, constant_shift=False):
        self.wn_ref = wn_ref
        self.degree = degree
        self.constant_shift = constant_shift

    def transform(self, spectrum, wnaxis):
        """Apply the configured phase rotation to `spectrum`.

        Parameters
        ----------
        spectrum : array_like
            Spectral data values. Real values are converted to complex
            before rotation.
        wnaxis : array_like
            Wavenumber axis values corresponding to `spectrum`, used only
            when :attr:`constant_shift` is ``False``.

        Returns
        -------
        array_like
            The phase (in radians) of the rotated spectrum.
        """
        if not np.iscomplex(spectrum).any():
            spectrum = np.exp(spectrum * complex(1j))

        if not self.constant_shift:
            angles = wnaxis * np.deg2rad(self.degree) / self.wn_ref
        else:
            angles = np.deg2rad(self.degree)

        return np.angle(spectrum * np.exp(angles * complex(1j)))


class ShiftPhaseToZero(Transformation):
    """
    Calculates and applies the phase rotation needed to get a flat, levelled phase spectrum.
    Two reference frequencies have to be provided.
    """

    def __init__(self, wavenumber1=1000.0, wavenumber2=2200.0):
        self.wn1 = wavenumber1
        self.wn2 = wavenumber2

    def transform(self, spectrum, wnaxis):
        """Level the phase of `spectrum` using the two configured reference points.

        Parameters
        ----------
        spectrum : array_like
            Phase spectrum values. Real values are converted to complex
            before rotation.
        wnaxis : array_like
            Wavenumber axis values corresponding to `spectrum`.

        Returns
        -------
        array_like
            The rotated phase spectrum (in radians), flat and equal to zero
            at `wavenumber1` and `wavenumber2`.
        """
        if not np.iscomplex(spectrum).any():
            spectrum = np.exp(spectrum * complex(1j))

        wn1idx = np.argmin(abs(wnaxis - self.wn1))
        wn2idx = np.argmin(abs(wnaxis - self.wn2))

        theta1 = np.angle(spectrum[wn1idx])
        wn1 = wnaxis[wn1idx]
        spectrum = spectrum * np.exp(-theta1 * complex(1j))

        theta2 = np.angle(spectrum[wn2idx])
        wn2 = wnaxis[wn2idx]

        angles = (wnaxis - wn1) * theta2 / (wn2 - wn1)
        spectrum = np.angle(spectrum * np.exp(-1 * angles * complex(1j)))

        return spectrum


class NormalizeSpectrum(Transformation):
    """Normalize a spectrum against a reference spectrum.

    Parameters
    ----------
    datatype : DataTypes, optional
        Kind of data being normalized; :data:`DataTypes.Phase` and
        :data:`DataTypes.Topography` are treated as phase-like data and
        normalized via complex division, other types via plain division.
    dounwrap : bool, optional
        If ``True`` and `datatype` is phase-like, unwrap the resulting phase
        spectrum.

    Attributes
    ----------
    datatype : DataTypes
        Kind of data being normalized.
    dounwrap : bool
        Whether to unwrap the resulting phase spectrum.
    """

    def __init__(self, datatype=DataTypes.Phase, dounwrap=False):
        self.datatype = datatype
        self.dounwrap = dounwrap

    def transform(self, spectrum, refspectrum):
        """Normalize `spectrum` against `refspectrum`.

        Parameters
        ----------
        spectrum : array_like
            Spectral data values to normalize.
        refspectrum : array_like
            Reference spectral data values, same shape as `spectrum`.

        Returns
        -------
        array_like
            The normalized spectrum: for phase-like data, the (optionally
            unwrapped) phase difference between `spectrum` and
            `refspectrum`; otherwise the elementwise ratio.
        """
        if self.datatype == DataTypes.Phase or self.datatype == DataTypes.Topography:
            newspectrum = np.angle(
                np.exp(spectrum * complex(1j)) / np.exp(refspectrum * complex(1j))
            )
            if self.dounwrap:
                return np.unwrap(newspectrum)
            else:
                return newspectrum
        else:
            return spectrum / refspectrum


# TOOLS ------------------------------------------------------------------------------------------------------------------
class Tools:
    """Helper functions for working with raw spectral measurement data."""

    @staticmethod
    def reshape_spectrum_data(data, params):
        """Reshape flat channel arrays into point/linescan/hyperscan layouts.

        Compensates for the zero-filling NeaSpec applies to Fourier-scan
        interferograms, and infers the spectral axis length from any
        available axis channel (``"Depth"``, ``"Index"``, ``"Omega"``,
        ``"Wavenumber"`` or ``"Wavelength"``), falling back to the pixel
        area recorded in `params`.

        Parameters
        ----------
        data : dict
            Dictionary mapping channel names to flat arrays of raw values.
            Reshaped in place.
        params : dict
            Measurement parameter dictionary containing at least ``"Scan"``
            and ``"PixelArea"`` entries.

        Returns
        -------
        dict
            The same `data` dictionary, with every channel reshaped to
            ``(spectral_depth,)`` for point spectra, or
            ``(PixelArea[0], PixelArea[1], spectral_depth)`` for line/hyper
            scans.
        """
        # To compensate for the zero-filling that NeaSpec does
        n = 1
        if params["Scan"] == "Fourier Scan":
            n = 2

        allchannels = list(data.keys())
        if "Depth" in allchannels:
            spectral_depth = len(np.unique(data["Depth"]))
        elif "Index" in allchannels:
            spectral_depth = len(np.unique(data["Index"]))
        elif "Omega" in allchannels:
            spectral_depth = len(np.unique(data["Omega"]))
        elif "Wavenumber" in allchannels:
            spectral_depth = len(np.unique(data["Wavenumber"]))
        elif "Wavelength" in allchannels:
            spectral_depth = len(np.unique(data["Wavelength"]))
        else:
            spectral_depth = params["PixelArea"][2] * n

        for channel in allchannels:
            # Point spectrum
            if params["PixelArea"][1] == 1 and params["PixelArea"][0] == 1:
                data[channel] = np.reshape(data[channel], (spectral_depth))

            # Linescan and HyperScan
            else:
                data[channel] = np.reshape(
                    data[channel],
                    (
                        params["PixelArea"][0],
                        params["PixelArea"][1],
                        spectral_depth,
                    ),
                )

        return data
