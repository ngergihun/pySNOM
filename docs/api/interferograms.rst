Interferogram processing (:mod:`pySNOM.interferograms`)
=======================================================

.. currentmodule:: pySNOM.interferograms

This module contains the main interferogram processing classes and functions. 
The :class:`pySNOM.interferograms.NeaInterferogram` class is the main data structure for interferograms, 
and the :class:`pySNOM.interferograms.Transformation` class is the base class for all interferogram processing transformations. 
Most of the transformations organized to be high level, simple to use, on the :class:`pySNOM.interferograms.NeaInterferogram` class, 
but the most basic operations can be applied to arraylike data structures as well.

Data classes
^^^^^^^^^^^^
The following classes are the main data structures for images and measurements. They help to manage the data and provide a consistent 
interface for processing and analysis but it is not neccessary to use them directly.

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst

   NeaInterferogram

Transformer classes
^^^^^^^^^^^^^^^^^^^
Transformer classes are used to apply transformations to interferograms. 
They are derived from the :class:`pySNOM.interferograms.Transformation` base class and implement 
the :meth:`pySNOM.interferograms.Transformation.transform` method to perform the interferogram processing.

Simple transformers
""""""""""""""""""""""""""""""""""""""
.. autosummary::
	:toctree: generated
	:template: autosummary/class.rst

	ProcessInterferogram
	InterpolateInterferogram

:class:`NeaInterferogram` transformers
""""""""""""""""""""""""""""""""""""""
.. autosummary::
	:toctree: generated
	:template: autosummary/class.rst

	ProcessSingleChannel
	ProcessMultiChannels
	ProcessAllPoints

Additional tools
""""""""""""""""""""""""""""""""""""""
In addition to the main data structures and transformers, 
this module also contains some utility classes and functions 
that can be used for interferogram processing pipelines. They are wrapped in the following classes.

.. autosummary::
	:toctree: generated
	:template: autosummary/class.rst

	Tools
