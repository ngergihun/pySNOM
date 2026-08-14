File readers (:mod:`pySNOM.readers`)
====================================

.. currentmodule:: pySNOM.readers

This module contains the main file readers for pySNOM. The :class:`pySNOM.readers.Reader` class is the base class for all readers, 
and it provides a consistent interface for reading data from different file formats. The following classes are derived from the 
:class:`pySNOM.readers.Reader` base class and implement the :meth:`pySNOM.readers.Reader.read` method to read data from specific file formats.

Base class
^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst

   Reader

Image readers
^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst

   GwyReader
   GsfReader
   ImageStackReader
   ImageStackXYZReader

Spectrum readers
^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst

   NeaSpectralReader
   NeaFileLegacyReader
   
Info file readers and helpers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst

   NeaHeaderReader
   NeaInfoReader

Utility functions
^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated
   :template: autosummary/member.rst

   get_wl_from_infofile
   get_wl_from_filename
   get_filenames
   recreate_infofile_name_from_path

