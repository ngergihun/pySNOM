"""File readers for SNOM measurement data.

This module provides the :class:`Reader` base class and its subclasses for
loading images (Gwyddion, GSF, XYZ image stacks), spectra (NeaSpec, XYZ
file formats), and their associated info/header files, plus a handful of
filename and info-file helper functions used by the readers.
"""

import gwyfile
import gsffile
import numpy as np
import pandas as pd
import os
from pathlib import PurePath
import re


class Reader:
    """Base class for all pySNOM file readers.

    Parameters
    ----------
    fullfilepath : str, optional
        Full path (with name) of the file to read.

    Attributes
    ----------
    filename : str or None
        Full path (with name) of the file to read.
    """

    def __init__(self, fullfilepath=None):
        self.filename = fullfilepath


class GwyReader(Reader):
    """Reader for Gwyddion (``.gwy``) files.

    Parameters
    ----------
    fullfilepath : str, optional
        Full path (with name) of the ``.gwy`` file to read.
    channelname : str, optional
        If given, :meth:`read` returns only this channel instead of all of them.

    Attributes
    ----------
    channelname : str or None
        Name of the single channel to extract, or ``None`` to return all channels.
    """

    def __init__(self, fullfilepath=None, channelname=None):
        super().__init__(fullfilepath)
        self.channelname = channelname

    def read(self):
        """Read the Gwyddion file and return its data field(s).

        Returns
        -------
        dict or GwyDataField
            A dictionary mapping channel names to :class:`gwyfile.objects.GwyDataField`
            objects when :attr:`channelname` is ``None``; otherwise the single
            requested :class:`gwyfile.objects.GwyDataField`.
        """
        # Returns a dictionary of all the channels
        gwyobj = gwyfile.load(self.filename)
        allchannels = gwyfile.util.get_datafields(gwyobj)

        if self.channelname is None:
            return allchannels
        else:
            # Read channels from gwyfile and return only a specific one
            channel = allchannels[self.channelname]
            return channel


class GsfReader(Reader):
    """Reader for Gwyddion Simple Field (``.gsf``) files.

    Parameters
    ----------
    fullfilepath : str, optional
        Full path (with name) of the ``.gsf`` file to read.
    """

    def __init__(self, fullfilepath=None):
        super().__init__(fullfilepath)

    def read(self):
        """Read the GSF file and wrap it in a :class:`gwyfile.objects.GwyDataField`.

        Returns
        -------
        gwyfile.objects.GwyDataField
            Data field built from the GSF data and its real size/offset metadata.
        """
        data, metadata = gsffile.read_gsf(self.filename)
        channel = gwyfile.objects.GwyDataField(
            data,
            xreal=metadata["XReal"],
            yreal=metadata["YReal"],
            xoff=metadata["XOffset"],
            yoff=metadata["YOffset"],
            si_unit_xy=None,
            si_unit_z=None,
            typecodes=None,
        )
        return channel


class NeaHeaderReader(Reader):
    """Reader for the ``#``-commented header block of NeaSpec data files.

    Parameters
    ----------
    fullfilepath : str, optional
        Full path (with name) of the NeaSpec data file whose header is read.
    """

    def __init__(self, fullfilepath=None):
        super().__init__(fullfilepath)

    @staticmethod
    def parseline(linestring, params={}):
        """Parse a single ``#``-commented header line into `params`.

        Recognizes several NeaSpec-specific field names (scanner center
        position, scan/pixel area, averaging, interferometer center
        distance, regulator settings, Q-factor) and stores them with
        appropriately typed values; any other field is parsed as a float
        when possible, otherwise kept as a stripped string.

        Parameters
        ----------
        linestring : str
            A single tab-separated header line, starting with ``"# "``.
        params : dict, optional
            Dictionary of parameters to update in place with the parsed
            field. Mutated and returned.

        Returns
        -------
        dict
            The `params` dictionary, updated with the newly parsed field.
        """
        ct = linestring.split("\t")
        fieldname = ct[0][2:-1]
        fieldname = fieldname.replace(" ", "")

        if "Scanner Center Position" in linestring:
            fieldname = fieldname[:-5]
            params[fieldname] = [float(ct[2]), float(ct[3])]

        elif "Scan Area" in linestring:
            fieldname = fieldname[:-7]
            params[fieldname] = [float(ct[2]), float(ct[3]), float(ct[4])]

        elif "Pixel Area" in linestring:
            fieldname = fieldname[:-7]
            params[fieldname] = [int(ct[2]), int(ct[3]), int(ct[4])]

        elif "Averaging" in linestring:
            params[fieldname] = int(ct[2])

        elif "Interferometer Center/Distance" in linestring:
            fieldname = fieldname.replace("/", "")
            params[fieldname] = [
                float(ct[2].replace(",", "")),
                float(ct[3].replace(",", "")),
            ]

        elif "Regulator" in linestring:
            fieldname = fieldname[:-7]
            params[fieldname] = [float(ct[2]), float(ct[3]), float(ct[4])]

        elif "Q-Factor" in linestring:
            fieldname = fieldname.replace("-", "")
            params[fieldname] = float(ct[2])

        else:
            fieldname = ct[0][2:-1]
            fieldname = fieldname.replace(" ", "")
            val = ct[2]
            val = val.replace(",", "")
            try:
                params[fieldname] = float(val)
            except:
                params[fieldname] = val.strip()

        return params

    def read(self):
        """Read the header block and the channel name row that follows it.

        Returns
        -------
        tuple[list of str, dict]
            A `(channels, params)` pair: the list of channel names found on
            the first non-comment line, and the parameter dictionary parsed
            from the preceding ``#``-commented header lines.
        """
        params = {}
        with open(self.filename, encoding="utf8") as f:
            # Read www.neaspec.com
            line = f.readline()
            count = 1
            while f:
                line = f.readline()
                count = count + 1
                try:
                    if line[0] not in ("#", "\n"):
                        break
                    if line[0] == "#":
                        params = NeaHeaderReader.parseline(line, params)
                except IndexError:
                    break

            channels = line.strip().split("\t")
            channels = [channel.strip() for channel in channels]

        return channels, params


class NeaInfoReader(NeaHeaderReader):
    """Reader that extracts only the parameter dictionary from a NeaSpec info file.

    Parameters
    ----------
    fullfilepath : str, optional
        Full path (with name) of the NeaSpec info file to read.
    """

    def __init__(self, fullfilepath=None):
        super().__init__(fullfilepath)

    def read(self):
        """Read the info file and return its parameter dictionary.

        Returns
        -------
        dict
            Measurement parameters parsed from the info file's header.
        """
        _, infodict = super().read()
        return infodict


class NeaSpectralReader(Reader):
    """Reader for NeaSpec spectral data files (tab-separated, with a header block).

    Parameters
    ----------
    fullfilepath : str, optional
        Full path (with name) of the spectral data file to read.
    output : {"dict", "dataframe"}, optional
        Output format for the channel data returned by :meth:`read`. Any
        value other than ``"dict"`` returns the raw :class:`pandas.DataFrame`.

    Attributes
    ----------
    filename : str or None
        Full path (with name) of the spectral data file to read.
    """

    def __init__(self, fullfilepath=None, output="dict"):
        super().__init__(fullfilepath)
        self._output = output

    def read(self):
        """Read the spectral data file.

        Parses the header via :class:`NeaHeaderReader` to determine the
        channel names and the number of rows to skip, then reads the
        remaining tab-separated data.

        Returns
        -------
        tuple[dict or pandas.DataFrame, dict]
            A `(data, params)` pair: the channel data (as a dictionary of
            NumPy arrays when :attr:`_output` is ``"dict"``, otherwise as a
            :class:`pandas.DataFrame`), and the measurement parameter
            dictionary parsed from the header.
        """
        data = {}

        channels, params = NeaHeaderReader(self.filename).read()
        channels.append("")

        count = len(list(params.keys())) + 2

        data = pd.read_csv(
            self.filename,
            sep="\t",
            skiprows=count,
            encoding="utf-8",
            names=channels,
            lineterminator="\n",
        ).dropna(axis=1, how="all")

        cols_to_keep = [c for c in data.columns if c != ""]
        data = data[cols_to_keep]

        if self._output == "dict":
            data = data.to_dict("list")
            for key in list(data.keys()):
                data[key] = np.asarray(data[key])

        return data, params


class ImageStackReader(Reader):
    """Reads a list of images from the subfolders of the specified folder by loading the files that contain the pattern string in the filename"""

    def __init__(self, folder=None, folder_pattern=""):
        super().__init__(folder)
        self.folder = self.filename
        self.folder_pattern = folder_pattern

    def read(self, pattern):
        """Load and wavelength-sort a stack of GSF images matching `pattern`.

        Locates matching files via :func:`get_filenames`, reads each with
        :class:`GsfReader`, and determines a wavelength/index for sorting
        from the corresponding info file (falling back to the enumeration
        index if the info file is missing or invalid).

        Parameters
        ----------
        pattern : str
            Regular expression matched against filenames within the
            matching subfolders.

        Returns
        -------
        tuple[list, list]
            A `(imagestack, wns)` pair: the list of image data fields
            (as read by :class:`GsfReader`), and the corresponding list of
            wavelengths (or fallback indices), both sorted by wavelength.
        """
        imagestack = []
        wns = []
        filepaths = get_filenames(
            self.folder, pattern, folderpattern=self.folder_pattern
        )

        for i, path in enumerate(filepaths):
            data_reader = GsfReader(path)
            imagestack.append(data_reader.read().data)

            try:
                txtpath = recreate_infofile_name_from_path(path)
                inforeader = NeaInfoReader(txtpath)
                infodict = inforeader.read()
                wn = get_wl_from_infofile(infodict)
                wns.append(wn)
            except:
                wns.append(i)

        idxs = np.argsort(np.asarray(wns))
        imagestack = [imagestack[i] for i in idxs]
        wns = [wns[i] for i in idxs]

        return imagestack, wns


def get_wl_from_infofile(infodict: dict):
    """Extract the measurement wavelength/wavenumber from a NeaSpec info dictionary.

    Uses the ``"TargetWavelength"`` entry when set, falling back to the
    ``"InterferometerCenterDistance"`` entry otherwise. Values below 50
    (assumed to be in micrometres) are converted to wavenumbers (cm\\ :sup:`-1`).

    Parameters
    ----------
    infodict : dict
        Measurement parameter dictionary, as returned by :class:`NeaInfoReader`.

    Returns
    -------
    float or None
        The resolved wavelength/wavenumber, or ``None`` if it could not be
        determined from `infodict`.
    """
    try:
        if infodict["TargetWavelength"] == "":
            wn = infodict["InterferometerCenterDistance"][0]
        else:
            wn = infodict["TargetWavelength"]
            # NeaSpec is not consistent in the units of the wavelength.
            # Sometimes it is in cm-1, sometimes in um.
            if wn < 50.0:
                wn = 10000 / wn
    except:
        wn = None
    return wn


def get_wl_from_filename(filename):
    """Extract a wavenumber value embedded in a filename.

    Looks for a number immediately preceded by ``_`` or ``-`` and followed
    by a ``cm-1``/``cm_1`` unit suffix (e.g. ``"...-1234.5_cm-1.gsf"``).

    Parameters
    ----------
    filename : str
        File name or path to search; only the final path component is used.

    Returns
    -------
    float or None
        The extracted wavenumber, or ``None`` if no match was found.
    """

    wn = re.findall(
        r"(?<=[_-])(\d+(?:\.\d+)?)(?=(?:_?cm[-_]1|-?cm[-_]1))", PurePath(filename).name
    )

    if wn:
        wn = float(wn[0])
    else:
        wn = None

    return wn


def get_filenames(folder: str, pattern: str, folderpattern=""):
    """Find files within matching subfolders of `folder` whose names match `pattern`.

    Parameters
    ----------
    folder : str
        Parent folder whose immediate subfolders are searched.
    pattern : str
        Regular expression that a subfolder's file names must match to be
        included.
    folderpattern : str, optional
        Regular expression that subfolder names must match to be searched
        at all. Defaults to matching every subfolder.

    Returns
    -------
    list of str
        Paths (relative to `folder`) of all matching files, in the order
        they were found.
    """

    filepaths = []

    for subfolder in os.listdir(folder):
        if os.path.isdir(os.path.join(folder, subfolder)) and re.search(
            folderpattern, subfolder
        ):
            for name in os.listdir(os.path.join(folder, subfolder)):
                if re.search(pattern, name):
                    subpath = os.path.join(subfolder, name)
                    filepaths.append(os.path.join(folder, subpath))

    return filepaths


def recreate_infofile_name_from_path(filepath: str):
    """Derive the info-file (``.txt``) path corresponding to a data file path.

    The info file is assumed to be named after the data file's parent
    directory, placed one level up (NeaSpec's standard measurement folder
    layout).

    Parameters
    ----------
    filepath : str
        Path of a data file inside a NeaSpec measurement subfolder.

    Returns
    -------
    str
        Path of the corresponding info file.
    """

    pathparts = list(PurePath(filepath).parts)
    newparts = pathparts[:-1]
    newparts.append(pathparts[-2] + ".txt")

    return str(PurePath(*newparts))


class NeaFileLegacyReader(Reader):
    """Reader for legacy ``.nea`` files from older NeaSpec microscopes.

    Parameters
    ----------
    fullfilepath : str, optional
        Full path (with name) of the legacy ``.nea`` file to read.
    """

    def __init__(self, fullfilepath=None):
        super().__init__(fullfilepath)

    def read(self):
        """Read a legacy ``.nea`` file into per-channel data and scan parameters.

        Parses the tab-separated file's header and metadata columns (Row,
        Column, Run, Channel, ...) to reconstruct each optical/mechanical
        channel as a flat array indexed consistently with the scan's pixel
        area, run count and spectral (omega) depth.

        Returns
        -------
        tuple[dict, dict]
            A `(data, params)` pair: a dictionary mapping channel (and
            metadata) names to flat NumPy arrays, and a parameter
            dictionary containing at least ``"PixelArea"`` and ``"Scan"``.
        """
        data = {}
        params = {}

        with open(self.filename, encoding="utf8") as f:
            h = next(f)  # header
            h = h.strip()
            h = h.split("\t")

            l = next(f)
            l = l.strip()
            l = l.split("\t")

            f.seek(0)
            next(f)
            datacols = np.arange(len(h), len(l))
            C_data = np.loadtxt(f, dtype="float", usecols=datacols)

            f.seek(0)
            next(f)

            metacols = np.arange(0, 4)
            meta = np.loadtxt(
                f,
                dtype={"names": tuple(h), "formats": (int, int, int, "S10")},
                usecols=metacols,
            )
            if "Run" in h:
                runs = np.unique(meta["Run"])
            else:
                runs = [0]

            Max_row = len(np.unique(meta["Row"]))
            Max_col = len(np.unique(meta["Column"]))
            Max_run = len(runs)
            Max_omega = np.shape(C_data)[1]

            N_rows = Max_row * Max_col * Max_run * Max_omega

            indexes = np.unique(meta["Channel"], return_index=True)[1]
            channels = [meta["Channel"][index] for index in sorted(indexes)]
            channels = [channel.decode("utf-8") for channel in channels]

            for name in h:
                if name != "Channel":
                    data[name] = np.array(meta[name])

            for i in range(len(channels)):
                data[channels[i]] = np.ravel(C_data[i * Max_run : (i + 1) * Max_run, :])

        alpha = 0
        beta = 0
        data["Run"] = np.zeros(N_rows)
        data["Column"] = np.zeros(N_rows)
        data["Row"] = np.zeros(N_rows)

        for i in range(0, N_rows, Max_omega * Max_run):
            if beta == Max_row:
                beta = 0
                alpha = alpha + 1
            data["Run"][i : i + Max_omega * Max_run] = np.repeat(
                np.arange(Max_run), Max_omega
            )
            data["Column"][i : i + Max_omega * Max_run] = alpha
            data["Row"][i : i + Max_omega * Max_run] = beta
            beta = beta + 1

            params["PixelArea"] = [
                Max_row,
                Max_col,
                Max_omega,
            ]
            params["Scan"] = "Fourier Scan"

        return data, params


class ImageStackXYZReader(Reader):
    """Reader for XYZ-format image stack files (tab-separated, one column per spectral slice).

    Parameters
    ----------
    fullfilepath : str, optional
        Full path (with name) of the XYZ image stack file to read.
    """

    def __init__(self, fullfilepath=None):
        super().__init__(fullfilepath)

    def read(self):
        """Read an XYZ-format image stack file into a list of 2D images.

        Parses the tab-separated file's header row as the spectral axis
        (e.g. wavenumbers) and its ``Row``/``Column`` metadata columns to
        reshape each spectral slice of the data into a 2D image.

        Returns
        -------
        tuple[list of numpy.ndarray, numpy.ndarray or None]
            A `(image_stack, x)` pair: the list of 2D images (one per
            spectral axis value), and the parsed spectral axis values `x`
            (``None`` if the header could not be parsed as floats).

        Raises
        ------
        ValueError
            If no file path was provided (:attr:`filename` is ``None``).
        """
        if self.filename is None:
            raise ValueError("No folder specified")
        else:
            with open(self.filename, encoding="utf8") as f:
                x = next(f)  # header
                x = x.strip()
                x = x.split("\t")

                f.seek(0)
                next(f)
                datacols = np.arange(2, len(x) + 2)
                C_data = np.loadtxt(f, dtype="float", usecols=datacols)

                f.seek(0)
                next(f)

                metacols = np.arange(0, 2)
                meta = np.loadtxt(
                    f,
                    dtype={"names": ("Row", "Column"), "formats": (float, float)},
                    usecols=metacols,
                )

                Max_row = len(np.unique(meta["Row"]))
                Max_col = len(np.unique(meta["Column"]))
                Max_omega = len(x)

                image_stack = []
                for i in range(Max_omega):
                    image_stack.append(np.reshape(C_data[:, i], (Max_col, Max_row)))

                try:
                    x = np.array([float(xi) for xi in x])
                except ValueError:
                    x = None

                return image_stack, x
