from dataset.unreflectdataset import UnReflectAnything_Dataset
# Dataset-specific classes inheriting from base UnReflectAnything_Dataset class
class SCRREAM_Dataset(UnReflectAnything_Dataset):
    """
    SCRREAM dataset implementation for polarization-based reflection removal.

    Inherits all functionality from the base UnReflectAnything_Dataset class.
    This class can be extended with SCRREAM-specific preprocessing,
    data augmentation, or validation logic as needed.

    The SCRREAM dataset contains RGB images with corresponding polarization
    data for training reflection removal models.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize SCRREAM dataset.

        Args:
            **kwargs: All arguments passed to parent UnReflectAnything_Dataset class
        """
        super().__init__(
            root_dir="$DATASET_DIR/SCRREAM/",
            rgb_ext=".png",
            pol_ext=".png",
            polarization_format="single_file_clock",  # "single_file_clock", "separate_files" or "mosaic"
            **kwargs,
        )
        # Add any SCRREAM-specific initialization here


class HOUSECAT6D_Dataset(UnReflectAnything_Dataset):
    """
    HOUSECAT6D dataset implementation for 6D pose estimation with polarization.

    Inherits all functionality from the base UnReflectAnything_Dataset class.
    This class can be extended with HOUSECAT6D-specific preprocessing,
    pose annotation loading, or 6D pose-specific data augmentation.

    The HOUSECAT6D dataset provides RGB and polarization data along with
    6D object pose annotations for training pose estimation models.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize HOUSECAT6D dataset.

        Args:
            **kwargs: All arguments passed to parent UnReflectAnything_Dataset class
        """
        super().__init__(
            root_dir="$DATASET_DIR/HouseCat6D/",
            rgb_ext=".png",
            pol_ext=".png",
            polarization_format="single_file_clock",  # "single_file_clock", "separate_files" or "mosaic"
            **kwargs,
        )
        # Add any HOUSECAT6D-specific initialization here


class POLARGB_Dataset(UnReflectAnything_Dataset):
    """
    PolaRGB dataset implementation for polarization-guided RGB processing.

    Inherits all functionality from the base UnReflectAnything_Dataset class.
    This class can be extended with PolaRGB-specific preprocessing,
    polarization analysis, or RGB enhancement techniques.

    The PolaRGB dataset combines RGB imagery with polarization measurements
    for improved scene understanding and image enhancement tasks.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize PolaRGB dataset.

        Args:
            **kwargs: All arguments passed to parent UnReflectAnything_Dataset class
        """
        super().__init__(
            root_dir="$DATASET_DIR/PolaRGB/",
            rgb_ext=".png",
            pol_ext=".png",
            polarization_format="separate_files",  # "single_file_clock", "separate_files" or "mosaic"
            **kwargs,
        )
        # Add any PolaRGB-specific initialization here


class CROMO_Dataset(UnReflectAnything_Dataset):
    """
    CROMO dataset implementation for polarization-guided RGB processing.

    Inherits all functionality from the base UnReflectAnything_Dataset class.
    This class can be extended with CROMO-specific preprocessing,
    polarization analysis, or RGB enhancement techniques.

    The CROMO dataset combines RGB imagery with polarization measurements
    for improved scene understanding and image enhancement tasks.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize CROMO dataset.

        Args:
            **kwargs: All arguments passed to parent UnReflectAnything_Dataset class
        """
        super().__init__(
            root_dir="$DATASET_DIR/CroMo/",
            rgb_dir_name="rgb/left/data",
            pol_dir_name="polarized/left/data/",
            rgb_ext=".png",
            pol_ext=".png",
            polarization_format="single_file_topdown",
            **kwargs,
        )
        # Add any SCARED-specific initialization here


class SCARED_Dataset(UnReflectAnything_Dataset):
    """
    SCARED dataset implementation for polarization-guided RGB processing.

    Inherits all functionality from the base UnReflectAnything_Dataset class.
    This class can be extended with SCARED-specific preprocessing,
    polarization analysis, or RGB enhancement techniques.

    The SCARED dataset combines RGB imagery with polarization measurements
    for improved scene understanding and image enhancement tasks.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize SCARED dataset.

        Args:
            **kwargs: All arguments passed to parent UnReflectAnything_Dataset class
        """
        super().__init__(
            root_dir="$DATASET_DIR/SCARED/",
            rgb_ext=".png",
            pol_ext=".png",
            **kwargs,
        )
        # Add any SCARED-specific initialization here


class STEREOMIS_TRACKING_Dataset(UnReflectAnything_Dataset):
    """
    STEREOMIS_TRACKING dataset implementation for polarization-guided RGB processing.
    Inherits all functionality from the base UnReflectAnything_Dataset class.
    This class can be extended with STEREOMIS_TRACKING-specific preprocessing,
    polarization analysis, or RGB enhancement techniques.

    The STEREOMIS_TRACKING dataset combines RGB imagery with polarization measurements
    for improved scene understanding and image enhancement tasks.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize SCARED dataset.

        Args:
            **kwargs: All arguments passed to parent UnReflectAnything_Dataset class
        """
        super().__init__(
            root_dir="$DATASET_DIR/StereoMIS_Tracking/",
            rgb_dir_name="video_frames",
            rgb_ext=".png",
            **kwargs,
        )
        # Add any SCARED-specific initialization here


class CHOLEC80_Dataset(UnReflectAnything_Dataset):
    """
    CHOLEC80 dataset implementation for polarization-guided RGB processing.
    Inherits all functionality from the base UnReflectAnything_Dataset class.
    This class can be extended with CHOLEC80-specific preprocessing,
    polarization analysis, or RGB enhancement techniques.

    The CHOLEC80 dataset combines RGB imagery with polarization measurements
    for improved scene understanding and image enhancement tasks.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize SCARED dataset.

        Args:
            **kwargs: All arguments passed to parent UnReflectAnything_Dataset class
        """
        super().__init__(
            root_dir="$DATASET_DIR/CHOLEC80/videos/",
            rgb_dir_name="frames",
            rgb_ext=".png",
            **kwargs,
        )
        # Add any SCARED-specific initialization here


class PSD_Dataset(UnReflectAnything_Dataset):
    """
    PSD dataset implementation for polarization-guided RGB processing.

    Inherits all functionality from the base UnReflectAnything_Dataset class.
    This class can be extended with PSD-specific preprocessing,
    polarization analysis, or RGB enhancement techniques.

    The PSD dataset combines RGB imagery with polarization measurements
    for improved scene understanding and image enhancement tasks.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize PSD dataset.

        Args:
            **kwargs: All arguments passed to parent UnReflectAnything_Dataset class
        """
        super().__init__(
            root_dir="$DATASET_DIR/PSD_Dataset/",
            rgb_ext=".png",
            rgb_dir_name="specular",
            diffuse_dir_name="diffuse",
            **kwargs,
        )
