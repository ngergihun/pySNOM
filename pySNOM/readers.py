import gwyfile
import gsffile
import numpy as np

class Reader:
    def __init__(self, fullfilepath=None):
        self.filename = fullfilepath

class GwyReader(Reader):

    def __init__(self, fullfilepath=None, channelname=None):
        super().__init__(fullfilepath)
        self.channelname = channelname

    def read(self):
        # Returns a dictionary of all the channels
        gwyobj = gwyfile.load(self.filename)
        allchannels = gwyfile.util.get_datafields(gwyobj)

        if self.channelname == None:
            return allchannels
        else:
            # Read channels from gwyfile and return only a specific one
            channel = allchannels[self.channelname]
            return channel

class GsfReader(Reader):
    def __init__(self, fullfilepath=None):
        super().__init__(fullfilepath)

    def read(self):
        data, metadata = gsffile.read_gsf(self.filename)
        channel = gwyfile.objects.GwyDataField(data,
                 xreal=metadata["XReal"], yreal=metadata["YReal"], xoff=metadata["XOffset"], yoff=metadata["YOffset"],
                 si_unit_xy=None, si_unit_z=None,
                 typecodes=None)
        return channel

class NeaInfoReader(Reader):
    def __init__(self, fullfilepath=None):
        super().__init__(fullfilepath)

    def read(self):
        # reader tested for neascan version 2.1.10719.0
        fid = open(self.filename, errors='replace')
        infodict = {}

        linestring = ''
        Nlines = 0

        while 'Version:' not in linestring:
            Nlines += 1
            linestring = fid.readline()
            if Nlines > 1:
                ct = linestring.split('\t')
                fieldname = ct[0][2:-1]
                fieldname = fieldname.replace(' ', '')

                if 'Scanner Center Position' in linestring:
                    fieldname = fieldname[:-5]
                    infodict[fieldname] = [float(ct[2]), float(ct[3])]

                elif 'Scan Area' in linestring:
                    fieldname = fieldname[:-7]
                    infodict[fieldname] = [float(ct[2]), float(ct[3]), float(ct[4])]

                elif 'Pixel Area' in linestring:
                    fieldname = fieldname[:-7]
                    infodict[fieldname] = [int(ct[2]), int(ct[3]), int(ct[4])]

                elif 'Interferometer Center/Distance' in linestring:
                    fieldname = fieldname.replace('/', '')
                    infodict[fieldname] = [float(ct[2]), float(ct[3])]

                elif 'Regulator' in linestring:
                    fieldname = fieldname[:-7]
                    infodict[fieldname] = [float(ct[2]), float(ct[3]), float(ct[4])]

                elif 'Q-Factor' in linestring:
                    fieldname = fieldname.replace('-', '')
                    infodict[fieldname] = float(ct[2])

                else:
                    fieldname = ct[0][2:-1]
                    fieldname = fieldname.replace(' ', '')
                    val = ct[2]
                    val = val.replace(',', '')
                    try:
                        infodict[fieldname] = float(val)
                    except:
                        infodict[fieldname] = val.strip()
        fid.close()
        return infodict

class NeaSpectrumReader(Reader):
    def __init__(self, fullfilepath=None):
        super().__init__(fullfilepath)

    def read(self):
        # reader tested for neascan version 2.1.10719.0
        fid = open(self.filename,errors='replace')
        data = {}
        params = {}

        linestring = fid.readline()
        Nlines = 1

        while 'Row' not in linestring:
            Nlines += 1
            linestring = fid.readline()
            if Nlines > 1:
                ct = linestring.split('\t')
                fieldname = ct[0][2:-1]
                fieldname = fieldname.replace(' ', '')

                if 'Scanner Center Position' in linestring:
                    fieldname = fieldname[:-5]
                    params[fieldname] = [float(ct[2]), float(ct[3])]

                elif 'Scan Area' in linestring:
                    fieldname = fieldname[:-7]
                    params[fieldname] = [float(ct[2]), float(ct[3]), float(ct[4])]

                elif 'Pixel Area' in linestring:
                    fieldname = fieldname[:-7]
                    params[fieldname] = [int(ct[2]), int(ct[3]), int(ct[4])]

                elif 'Interferometer Center/Distance' in linestring:
                    fieldname = fieldname.replace('/', '')
                    params[fieldname] = [float(ct[2]), float(ct[3])]

                elif 'Regulator' in linestring:
                    fieldname = fieldname[:-7]
                    params[fieldname] = [float(ct[2]), float(ct[3]), float(ct[4])]

                elif 'Q-Factor' in linestring:
                    fieldname = fieldname.replace('-', '')
                    params[fieldname] = float(ct[2])

                else:
                    fieldname = ct[0][2:-1]
                    fieldname = fieldname.replace(' ', '')
                    val = ct[2]
                    val = val.replace(',','')
                    try:
                        params[fieldname] = float(val)
                    except:
                        params[fieldname] = val.strip()

        channels = linestring.split('\t')
        fid.close()

        if "PTE+" in params['Scan']:
            C_data = np.genfromtxt(self.filename, skip_header=Nlines, encoding='utf-8')
        else:
            C_data = np.genfromtxt(self.filename, skip_header=Nlines)

        for i in range(len(channels)-1):
            data[channels[i]] = C_data[:,i]

        return data, params
    
class NeaInterferogramReader(Reader):
    def __init__(self, fullfilepath=None):
        super().__init__(fullfilepath)

    def read(self):
        # reader tested for neascan version 2.1.10719.0
        fid = open(self.filename,errors='replace')
        data = {}
        params = {}

        linestring = fid.readline()
        Nlines = 1

        while 'Row' not in linestring:
            Nlines += 1
            linestring = fid.readline()
            if Nlines > 1:
                ct = linestring.split('\t')
                fieldname = ct[0][2:-1]
                fieldname = fieldname.replace(' ', '')

                if 'Scanner Center Position' in linestring:
                    fieldname = fieldname[:-5]
                    params[fieldname] = [float(ct[2]), float(ct[3])]

                elif 'Scan Area' in linestring:
                    fieldname = fieldname[:-7]
                    params[fieldname] = [float(ct[2]), float(ct[3]), float(ct[4])]

                elif 'Pixel Area' in linestring:
                    fieldname = fieldname[:-7]
                    params[fieldname] = [int(ct[2]), int(ct[3]), int(ct[4])]

                elif 'Averaging' in linestring:
                    # fieldname = fieldname[:-7]
                    params[fieldname] = int(ct[2])

                elif 'Interferometer Center/Distance' in linestring:
                    fieldname = fieldname.replace('/', '')
                    params[fieldname] = [float(ct[2]), float(ct[3])]

                elif 'Regulator' in linestring:
                    fieldname = fieldname[:-7]
                    params[fieldname] = [float(ct[2]), float(ct[3]), float(ct[4])]

                elif 'Q-Factor' in linestring:
                    fieldname = fieldname.replace('-', '')
                    params[fieldname] = float(ct[2])

                else:
                    fieldname = ct[0][2:-1]
                    fieldname = fieldname.replace(' ', '')
                    val = ct[2]
                    val = val.replace(',','')
                    try:
                        params[fieldname] = float(val)
                    except:
                        params[fieldname] = val.strip()

        channels = linestring.split('\t')
        fid.close()

        C_data = np.genfromtxt(self.filename, skip_header=Nlines)

        for i in range(len(channels)-1):
            data[channels[i]] = C_data[:,i]
        
        return data, params