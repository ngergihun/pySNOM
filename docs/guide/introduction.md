# Introduction

pySNOM data pipelines generally follow the same two-step pattern:

1. **Read** raw measurement data from disk with a `pySNOM.readers.Reader`
   subclass, appropriate to the file format (Gwyddion `.gwy`, GSF `.gsf`,
   or NeaSpec tab-separated text files).
2. **Process** the resulting data with a chain of `Transformation` objects
   from `pySNOM.images` (for AFM/near-field images), `pySNOM.spectra` (for
   point, line, and hyperspectral spectra), or `pySNOM.interferograms` (for
   raw interferogram data).

Each `Transformation` implements a `transform(data)` method that takes and
returns a NumPy array, so transformations can be composed and applied in
sequence to build up a processing pipeline.

See [Readers](readers.md), [Images](images.md), [Spectra](spectra.md), and
[Interferograms](interferograms.md) for worked examples of each stage.
