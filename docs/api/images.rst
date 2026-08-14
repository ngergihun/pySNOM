Images processing (:mod:`pySNOM.images`)
========================================

.. currentmodule:: pySNOM.images

This module contains the main image processing classes and functions. The :class:`pySNOM.Image` class is the main data structure for images, 
and the :class:`pySNOM.Transformation` class is the base class for all image processing transformations. The :class:`pySNOM.Measurement` 
class is the base class for all measurement data structures.

Data classes
^^^^^^^^^^^^
The following classes are the main data structures for images and measurements. They help to manage the data and provide a consistent 
interface for processing and analysis but it is not neccessary to use them directly.

.. autosummary::
   :toctree: generated
   :template: class.rst

   Measurement
   Image
   GwyImage

Transformer classes
^^^^^^^^^^^^^^^^^^^
Transformer classes are used to apply transformations to images. They are derived from the :class:`pySNOM.Transformation` base class and implement 
the :meth:`pySNOM.Transformation.transform` method to perform the image corrections. Some classes implement a :meth:`calculate` method to extract the correction values.
The extracted values can be used to apply the processing to another image by feeding the values to the :meth:`correct` method of the same class. 
For these tranformers the `transform` method is a combination of `calculate` and `correct`.

.. autosummary::
   :toctree: generated
   :template: class.rst

   Transformation
   MaskedTransformation
   LineLevel
   RotatePhase
   SelfReference
   SimpleNormalize
   BackgroundPolyFit
   MaskedBackgroundPolyFit
   LaplaceFillIn
   ValueFillIn
   RemoveSpikes
   ScarRemoval
   CalculateOpticalFlow
   WrapImage
   CalculateXCorrDrift
   CorrectImageDrift
   AlignImageStack

Other functions
^^^^^^^^^^^^^^^
This module also contains some utility functions for image processing. 
They are not part of any class and can be used directly.

.. autosummary::
   :toctree: generated
   :template: member.rst

   type_from_channelname
   mask_from_booleans
   mask_from_datacondition
   cut_cross_section
   shift_fill
   sort_image_stack
   create_nparray_stack
   dict_from_imagestack
   flatten_stack
