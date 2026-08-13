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
