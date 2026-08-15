# Spectra

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

See the [API reference](../api/spectra.rst) for the full list of spectral
transformations available.
