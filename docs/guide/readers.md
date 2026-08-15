# Readers

`pySNOM.readers` provides `Reader` subclasses for the file formats produced
by common SNOM instruments and analysis tools:

- `GwyReader` for Gwyddion (`.gwy`) files.
- `GsfReader` for Gwyddion Simple Field (`.gsf`) files.
- `NeaInfoReader` for NeaSpec info/header files.
- `NeaSpectralReader` for NeaSpec spectral and interferogram data files.
- `NeaFileLegacyReader` for legacy NeaSpec (`.nea`) files.
- `ImageStackXYZReader` for XYZ image stack files.

```python
from pySNOM.readers import GwyReader

reader = GwyReader(fullfilepath="measurement.gwy", channelname="O3A raw")
gwy_data = reader.read()
```

See the [API reference](../api/readers.rst) for the full list of readers and
their options.
