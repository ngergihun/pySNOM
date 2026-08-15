# Getting started

## Installation

pySNOM supports Python 3.10 and newer. Install the package from the repository
root with:

```console
python -m pip install .
```

The runtime dependencies are declared in `requirements.txt` and are installed
as part of the package installation.

## First import

The principal classes are available from the top-level package:

```python
from pySNOM import Image, NeaInterferogram, NeaSpectrum, Reader
```

Reader implementations are available in `pySNOM.readers`, while image,
spectrum, and interferogram processing classes are documented in the API
reference.

## Introduction

pySNOM data pipelines generally follow the same two-step pattern:

1. **Read** raw measurement data from disk with a `pySNOM.readers.Reader`
   subclass, appropriate to the file format (Gwyddion `.gwy`, GSF `.gsf`,
   or NeaSpec tab-separated text files).
2. **Process** the resulting data with a chain of `Transformation` objects
   from `pySNOM.images` (for AFM/near-field images) or `pySNOM.spectra`
   (for point, line, and hyperspectral spectra).

Each `Transformation` implements a `transform(data)` method that takes and
returns a NumPy array, so transformations can be composed and applied in
sequence to build up a processing pipeline.

## Image processing

Read a single channel from a Gwyddion file with `GwyReader`, wrap it in an
`Image`, and apply a chain of transformations such as `LineLevel` and
`BackgroundPolyFit`:

```python
from pySNOM.readers import GwyReader
from pySNOM.images import Image, LineLevel, BackgroundPolyFit, DataTypes

reader = GwyReader(fullfilepath="measurement.gwy", channelname="O3A raw")
gwy_data = reader.read()

image = Image(gwy_data, channelname="O3A raw")

leveled = LineLevel(method="median", datatype=DataTypes.Amplitude)
image.data = leveled.transform(image.data)

flattened = BackgroundPolyFit(xorder=1, yorder=1, datatype=DataTypes.Amplitude)
image.data = flattened.transform(image.data)
```

For stacks of images acquired at different wavenumbers (for example
hyperspectral scans), `pySNOM.images` also provides `AlignImageStack` and
related helpers (`create_nparray_stack`, `sort_image_stack`,
`dict_from_imagestack`) to align and reshape the stack before processing
each frame.

## Spectral processing

Read a NeaSpec spectrum file with `NeaSpectralReader` and wrap the resulting
data and parameters in a `NeaSpectrum`:

```python
from pySNOM.readers import NeaSpectralReader
from pySNOM import spectra

data, params = NeaSpectralReader("spectrum.txt").read()
spectrum = spectra.NeaSpectrum(data, params)

print(spectrum.mode, spectrum.scantype)
```

Spectral transformations in `pySNOM.spectra` follow the same
`transform(data)` interface as image transformations, so channels can be
cut, scaled, and normalized in a pipeline, for example with `Cut`, `Scale`,
`LinearNormalize`, and `ShiftPhaseToZero`.

See the API reference for the full list of readers and transformations
available for images, spectra, and interferograms.

