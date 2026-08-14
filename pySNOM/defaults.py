"""Definitions that map manufacturer names to pySNOM names.

The dictionaries in this module translate values found in Neaspec metadata
files into the measurement-mode names used by pySNOM.
"""


class Defaults:
    """Mappings between Neaspec metadata and pySNOM measurement modes.

    Attributes
    ----------
    image_mode_defs : dict of str to str
        Mapping from image ``Scan`` values to pySNOM image-mode names.
    spectral_mode_defs : dict of str to str
        Mapping from spectral ``Scan`` values to pySNOM spectral-mode names.
    """

    def __init__(self) -> None:
        """Create the default manufacturer-to-package mappings.

        Returns
        -------
        None
            This constructor initializes the mapping attributes in place.
        """

        self.image_mode_defs: dict[str, str] = {
            "AFM": "AFM",
            "2D (PsHet)": "PsHet",
            "Whitelight Imaging": "WLI",
            "Photo Thermal Expansion+": "PTE",
            "Tapping AFM-IR+": "TappingAFMIR",
            "Contact Mode 2D": "ContactAFM",
        }

        self.spectral_mode_defs: dict[str, str] = {
            "Fourier Scan": "nanoFTIR",
            "Pointspectroscopy PTE+": "PTE",
            "AFM-Raman/PL Scan (Tapping Mode)": "nanoRaman",
        }


defaults = Defaults()
