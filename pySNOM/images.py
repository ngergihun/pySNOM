import numpy as np
import copy
from enum import Enum
import pySNOM.defaults as defaults

MeasurementModes = Enum('MeasurementModes', ['None','AFM', 'PsHet', 'WLI', 'PTE', 'TappingAFMIR', 'ContactAFM'])
DataTypes = Enum('DataTypes', ['Amplitude', 'Phase', 'Topography'])
ChannelTypes = Enum('ChannelTypes', ['Optical','Mechanical'])

# Full measurement data containing all the measurement channels
class Measurement:
    def __init__(self, filename = None, data = None, info = None, mode = "None"):
        self.filename = filename  # Full path with name
        # Measurement mode (PTE, PSHet, AFM, NanoFTIR) - Enum MeasurementModes
        self.mode = MeasurementModes[mode]
        # Full data from the gwy files with all the channels
        self._data = data # All channels - Dictionary
        # Other parameters from info txt -  Dictionary
        if info is not None:
            self.setParameters(info)
        else:
            self._info = info

    @property
    def mode(self):
        return self._mode
    @mode.setter
    def mode(self, value: str):
        try:
            self._mode = MeasurementModes[value]
        except ValueError:
            self._mode = MeasurementModes["AFM"]
            raise ValueError(value + 'is not a measurement mode!')

    @property
    def data(self):
        return self._data
    @data.setter
    def data(self, data: dict):
        self._data = data

    # @property
    # def info(self):
    #     re

# METHODS --------------------------------------------------------------------------------------
    # Method to retrieve e specific channel from all the measurement channels
    def get_channel(self, channelname: str):
        channel = self.data[channelname]
        return channel
    
    # Method to set the additional informations about the measurement from the neaspec info txt
    def setParameters(self, infodict: dict):
        self.info = infodict
        self.setMeasurementModeFromParameters()
    
    # Method to set the 
    def setMeasurementModeFromParameters(self):
        if self.parameters == None:
            print('Load the info file first')
        else:
            m = self.parameters["Scan"]
            self.mode = defaults.image_mode_defs[m]

    def getImageChannel(self, channelname: str):
        singleimage = Image()

        singleimage.filename = self.filename
        singleimage.mode = self.mode
        singleimage.parameters = self.parameters
        singleimage.channel = channelname

        channeldata = self.getChannelData(channelname)
        singleimage.setImageParameters(channeldata)
        singleimage.setChannel(channelname=channelname)

        return singleimage


# Single image from a single data channel
class Image(Measurement):
    def __init__(self, channeldata = None) -> None:
        super().__init__()
        # Describing channel and datatype
        self.channel = None # Full channel name
        self.order = None   # Order, nth
        self.datatype = None # Amplitude, Phase, Topography - Enum DataTypes
        # Important measurement parameters from gwyddion fields
        self.xreal = None   # Physical image width
        self.yreal = None   # Physical image height
        self.xoff = None    # Center position X
        self.yoff = None    # Center position Y
        self.xres = None    # Pixel size in X
        self.yres = None    # Pixel size in Y
        # Overwrite the data definition of parent class
        self.data = None # Actual data, this case it is only a single channel

        if channeldata:
            self.setImageParameters(channeldata)

    # Method to set the actual data (numpy array) - overwrite parent class method
    def setData(self, data):
        # Set the data
        self.data = data
        
    def getData(self):
        # Set the data
        return self.data
    
    def setChannel(self, channelname):
        self.channel = channelname
        self.order = int(self.channel[1])

        if self.channel[2] == 'P':
            self.datatype = DataTypes["Phase"]
        elif 'Z' in self.channel:
            self.datatype = DataTypes["Topography"]
        else:
            self.datatype = DataTypes["Amplitude"]

    def setImageParameters(self, singlechannel):
        # Set the basic attributes from gwyddion field
        for key in singlechannel:
            if key in dir(self):
                setattr(self, key, singlechannel[key])
        self.setData(data=singlechannel.data)


class Transformation:

    def transform(self, data):
        raise NotImplementedError()


class LineLevel(Transformation):

    def __init__(self, method='median', datatype=DataTypes.Phase):
        self.method = method
        self.datatype = datatype

    def transform(self, data):
        if self.method == 'median':
            norm = np.median(data, axis=1, keepdims=True)
        elif self.method == 'average':
            norm = np.mean(data, axis=1, keepdims=True)
        elif self.method == 'difference':
            if self.datatype == DataTypes.Amplitude:
                norm = np.median(data[1:] / data[:-1], axis=1, keepdims=True)
            else:
                norm = np.median(data[1:] - data[:-1], axis=1, keepdims=True)
            data = data[:-1]  # difference does not make sense for the last row

        if self.datatype == DataTypes.Amplitude:
            return data / norm
        else:
            return data - norm

class RotatePhase(Transformation):

    def __init__(self, degree=90.0):
        self.degree = degree

    def transform(self, amplitudedata, phasedata):
        # Construct complex dataset
        complexdata = amplitudedata * np.exp(phasedata*complex(1j))
        # Rotate and extract phase
        return np.angle(complexdata*np.exp(np.deg2rad(self.degree)*complex(1j)))

class SelfReference(Transformation):

    def __init__(self, datatype=DataTypes.Phase):
        self.datatype = datatype

    def transform(self, data, referencedata):
        if self.datatype == DataTypes.Amplitude:
            return np.divide(data, referencedata)
        elif self.datatype == DataTypes.Phase:
            return data-referencedata
        else:
            # TODO: We should replace this with a datatype error raise
            print("Self-referencing makes only sense for amplitude or phase data")

class SimpleNormalize(Transformation):

    def __init__(self, method='median', value=1, datatype=DataTypes.Phase):
        self.method = method
        self.value = value
        self.datatype = datatype

    def transform(self, data):
        match self.method:
            case 'median':
                if self.datatype == DataTypes.Amplitude:
                    return data / np.median(data)
                else:
                    return data - np.median(data)
            case 'mean':
                if self.datatype == DataTypes.Amplitude:
                    return data / np.mean(data)
                else:
                    return data - np.mean(data)
            case 'manual':
                if self.datatype == DataTypes.Amplitude:
                    return data / self.value
                else:
                    return data - self.value
                
class BackgroundPolyFit(Transformation):

    def __init__(self, xorder=int(1), yorder=int(1), datatype=DataTypes.Phase):
        self.xorder = xorder
        self.yorder = yorder
        self.datatype = datatype
        
    def transform(self, data):
        Z = copy.deepcopy(data)
        x = list(range(0, Z.shape[0]))
        y = list(range(0, Z.shape[1]))
        X, Y = np.meshgrid(x, y)
        x, y = X.ravel(), Y.ravel()

        def get_basis(x, y, max_order_x=1, max_order_y=1):
            """Return the fit basis polynomials: 1, x, x^2, ..., xy, x^2y, ... etc."""
            basis = []
            for i in range(max_order_y+1):
                # for j in range(max_order_x - i +1):
                for j in range(max_order_x+1):
                    basis.append(x**j * y**i)
            return basis

        basis = get_basis(x, y, self.xorder, self.yorder)
        A = np.vstack(basis).T
        b = Z.ravel()
        c, r, rank, s = np.linalg.lstsq(A, b, rcond=None)

        background = np.sum(c[:, None, None] * np.array(get_basis(X, Y, self.xorder, self.yorder)).reshape(len(basis), *X.shape),axis=0)

        if self.datatype == DataTypes["Amplitude"]:
            return Z/background, background
        else:
            return Z-background, background