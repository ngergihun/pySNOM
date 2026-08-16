# Interferograms

`pySNOM.interferograms` provides `NeaInterferogram` and a family of
`ProcessSingleChannel`, `ProcessMultiChannels`, and `ProcessAllPoints`
transformations for turning raw interferogram data into spectra, along with
`InterpolateInterferogram` for resampling onto a regular grid.

```python
from pySNOM.readers import NeaSpectralReader
from pySNOM.interferograms import NeaInterferogram

data, params = NeaSpectralReader("interferogram.txt").read()
ifg = NeaInterferogram(data, params)
```

See the [API reference](../api/interferograms.rst) for the full list of
interferogram processing classes and helpers.
