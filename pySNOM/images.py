import numpy as np
import copy
from enum import Enum
import pySNOM.defaults as defaults

MeasurementModes = Enum('MeasurementModes', ['None','AFM', 'PsHet', 'WLI', 'PTE', 'TappingAFMIR', 'ContactAFM'])
DataTypes = Enum('DataTypes', ['Amplitude', 'Phase', 'Topography'])
ChannelTypes = Enum('ChannelTypes', ['Optical','Mechanical'])

# Full measurement data containing all the measurement channels
class Data:
    def __init__(self, filename = None, data = None, parameters = None, mode = "None"):
        self.filename = filename  # Full path with name
        # Measurement mode (PTE, PSHet, AFM, NanoFTIR) - Enum MeasurementModes
        self.mode = MeasurementModes[mode]
        # Full data from the gwy files with all the channels
        self.data = data # All channels - Dictionary
        # Other parameters from info txt -  Dictionary
        if parameters is not None:
            self.setParameters(parameters)
        else:
            self.parameters = parameters

    # Method to set the actual data read from the gwyddion or gsf files
    def setData(self, data: dict):
        self.data = data

    # Method to set measurement mode directly
    def setMeasurementMode(self, measmode: str):
        self.mode = MeasurementModes[measmode]

    # Method to retrieve e specific channel from all the measurement channels
    def getChannelData(self, channelname: str):
        channel = self.data[channelname]
        return channel
    
    # Method to set the additional informations about the measurement from the neaspec info txt
    def setParameters(self, infodict: dict):
        self.parameters = infodict
        self.setMeasurementModeFromParameters()
    
    # Method to set the 
    def setMeasurementModeFromParameters(self):
        if self.parameters == None:
            print('Load the info file first')
        else:
            m = self.parameters["Scan"]
            self.setMeasurementMode(defaults.image_mode_defs[m])

    def getImageFromChannel(self, channelname: str):
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
class Image(Data):
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
        self.processor = Process()

        if channeldata:
            self.setImageParameters(channeldata)

    # Method to set the actual data (numpy array) - overwrite parent class method
    def setData(self, data):
        # Set the data
        self.data = data
        self.processor.setData(self.data)
        
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

    def setImageParameters(self, singlechannel: dict):
        # Set the basic attributes from gwyddion field
        for key in singlechannel:
            if key in dir(self):
                setattr(self, key, singlechannel[key])
        self.setData(data=singlechannel.data)

class Process:
    def __init__(self, data = None):
        self.data = data
        if self.data is not None:
            self.output = np.zeros(np.shape(self.data))
        else:
            self.output = None

    def setData(self, data):
        self.data = data
        self.output = np.zeros(np.shape(self.data))

    def getOutput(self):
        return self.output
    
    def getData(self):
        return self.data

    def line_level(self, mtype = 'median', datatype = DataTypes.Phase):
        match mtype:  # TODO: "match" is not python 3.9 and 3.8 compatible - we need to specify this in the requirements.
            case 'median':
                for i in range(self.data.shape[0]):
                    if datatype == DataTypes["Amplitude"]:
                        self.output[i][:] = self.data[i][:]/np.median(self.data[i][:])
                    else:
                        self.output[i][:] = self.data[i][:]-np.median(self.data[i][:])
            case 'average':
                for i in range(self.data.shape[0]):
                    if datatype == DataTypes["Amplitude"]:
                        self.output[i][:] = self.data[i][:]/np.mean(self.data[i][:])
                    else:
                        self.output[i][:] = self.data[i][:]-np.mean(self.data[i][:])
            case 'difference':
                for i in range(self.data.shape[0]-1):
                    if datatype == DataTypes["Amplitude"]:
                        c = np.median(self.data[i+1][:]/self.data[i][:])
                        self.output[i][:] = self.data[i][:]/c
                    else:
                        c = np.median(self.data[i+1][:]-self.data[i][:])
                        self.output[i][:] = self.data[i][:]-c
        return self.output

    def bg_polyfit(self, xorder = int(1), yorder = int(1), datatype = DataTypes.Phase):
        Z = copy.deepcopy(self.data)
        x = list(range(0, self.data.shape[0]))
        y = list(range(0, self.data.shape[1]))
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

        basis = get_basis(x, y, xorder, yorder)
        A = np.vstack(basis).T
        b = Z.ravel()
        c, r, rank, s = np.linalg.lstsq(A, b, rcond=None)

        background = np.sum(c[:, None, None] * np.array(get_basis(X, Y, xorder, yorder)).reshape(len(basis), *X.shape),
                            axis=0)

        if datatype == DataTypes["Amplitude"]:
            self.output = Z/background
        else:
            self.output = Z-background

        return self.output, background

    def rotate_phase(self, amplitudedata = None, degree = 90.0):
        # Construct complex dataset
        complexdata = amplitudedata * np.exp(self.data*complex(1j))
        # Rotate and extract phase
        self.output = np.angle(complexdata*np.exp(np.deg2rad(degree)*complex(1j)))

        return self.output

    def self_reference(self, refdata = None, datatype = DataTypes.Phase):
        if datatype == DataTypes.Amplitude:
            self.output = np.divide(self.data, refdata)
        elif datatype == DataTypes.Phase:
            self.output = self.data-refdata
        else:
            print("Self-referencing makes only sense for amplitude or phase data")

        return self.output

    def normalize_simple(self, method='median', value=1, datatype = DataTypes.Phase):
        self.output = np.zeros(np.shape(self.data))
        match method:
            case 'median':
                if datatype == DataTypes.Amplitude:
                    self.output = self.data / np.median(self.data)
                else:
                    self.output = self.data - np.median(self.data)
            case 'manual':
                if datatype == DataTypes.Amplitude:
                    self.output = self.data / value
                else:
                    self.output = self.data - value
        return self.output