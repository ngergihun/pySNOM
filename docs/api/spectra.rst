Spectrum processing (:mod:`pySNOM.spectra`)
=======================================================

.. currentmodule:: pySNOM.spectra

This module contains the main spectrum processing classes and functions. 
The :class:`pySNOM.spectra.NeaSpectrum` class is the main data structure for spectra, 
and the :class:`pySNOM.spectra.Transformation` class is the base class for all spectrum processing transformations. 

Data classes
^^^^^^^^^^^^
The following classes are the main data structures for images and measurements. They help to manage the data and provide a consistent 
interface for processing and analysis but it is not neccessary to use them directly.

.. autosummary::
   :toctree: generated
   :template: class.rst

   NeaSpectrum
   SingleChannelSpectrum

Transformer classes
^^^^^^^^^^^^^^^^^^^
Transformer classes are used to apply transformations to spectral data. They are derived from the :class:`pySNOM.Transformation` base class and implement 
the :meth:`pySNOM.Transformation.transform` method to perform the spectrum corrections. They usually accept arraylike data structures and return the processed data.

.. autosummary::
   :toctree: generated
   :template: class.rst

   Transformation
   Cut
   Scale
   LinearNormalize
   ConstantNormalize
   RotatePhase
   ShiftPhaseToZero
   NormalizeSpectrum

Additional tools
^^^^^^^^^^^^^^^^
In addition to the main data structures and transformers, 
this module also contains some utility classes and functions 
that can be used for spectrum processing pipelines. They are wrapped in the following classes.

.. autosummary::
	:toctree: generated
	:template: class.rst

	Tools