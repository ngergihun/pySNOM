import gwyfile
import gsffile
import numpy as np

class Reader:
    def __init__(self, fullfilepath):
        self.filename = fullfilepath

    def setFilename(self, filename):
        self.filename = filename

    def read_gwyfile(self):
        # Returns a dictionary of all the channels
        gwyobj = gwyfile.load(self.filename)
        allchannels = gwyfile.util.get_datafields(gwyobj)

        return allchannels

    def read_gwychannel(self, channelname: str):
        # Read channels from gwyfile and return only a specific one
        gwyobj = gwyfile.load(self.filename)
        channels = gwyfile.util.get_datafields(gwyobj)
        channel = channels[channelname]

        return channel
    
    def read_gsffile(self):
        data, metadata = gsffile.read_gsf(self.filename)

        channel = gwyfile.objects.GwyDataField(data,
                 xreal=metadata["XReal"], yreal=metadata["YReal"], xoff=metadata["XOffset"], yoff=metadata["YOffset"],
                 si_unit_xy=None, si_unit_z=None,
                 typecodes=None)

        return channel

    def read_nea_infofile(self):
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