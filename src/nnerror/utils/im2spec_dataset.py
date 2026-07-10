import numpy as np
import torch
from atomai.utils import get_coord_grid, extract_subimages
from torch.utils.data import ConcatDataset, Dataset

__all__ = [
    "AddGaussianNoise",
    "Error_Dataset",
    "augmented_dataset",
    "im2spec_Dataset",
    "norm_0to1",
    "paired_images_spectra",
    "paired_images_spectra_1",
]


class im2spec_Dataset(Dataset):

    """
    PyTorch dataset for image-to-spectrum training pairs.

    Args:
        images: Image-patch array of shape `(N, H, W)`.
        spectra: Spectra array of shape `(N, S)`.
        transform: Optional image transform applied after conversion to tensor.
        norm: Retained for compatibility; normalization is currently disabled.
    """
    def __init__(self, images, spectra, transform = None, norm = True):
        """Store image patches, spectra, and an optional transform."""
        self.images = images
        self.spectra = spectra
        self.transform = transform
#         self.norm = norm
#         self.normalize = transforms.Normalize(mean=[0.5], std=[0.5]) # Normalize in range[-1 to 1]

    def __len__(self):

        """Return the number of image/spectrum pairs."""
        return len(self.images)

    def __getitem__(self, idx):

        """
        Return one `(image, spectrum)` sample as float tensors.

        Args:
            idx: Sample index.

        Returns:
            Tuple `(image, spectra)`.
        """
        image = torch.tensor(self.images[idx], dtype=torch.float32)

        if self.transform is not None:
            image = self.transform(image)

        spectra = torch.tensor(self.spectra[idx], dtype=torch.float32)

#         if self.norm:
#             image = self.normalize(image)
#             spectra = self.normalize(spectra)

        return image, spectra


def augmented_dataset(images, spectra):

    """
    Create a concatenated dataset with original and augmented samples.

    The returned dataset includes the original samples, a noisy horizontal-flip
    copy, and a noisier vertical-flip copy.

    Args:
        images: Image-patch array of shape `(N, H, W)`.
        spectra: Spectra array of shape `(N, S)`.

    Returns:
        `ConcatDataset` containing three image-to-spectrum datasets.
    """
    import torchvision.transforms as transforms

    dataset1 = im2spec_Dataset(images, spectra)


    # Define the transform1
    transform1 = transforms.Compose([
        AddGaussianNoise(mean=0., std=0.1),
        transforms.RandomHorizontalFlip(p = 1)
    ])

    dataset2 = im2spec_Dataset(images, spectra, transform = transform1)


    # Define the transform2
    transform2 = transforms.Compose([
        AddGaussianNoise(mean=0.0, std=0.5),
        transforms.RandomVerticalFlip(p = 1)
    ])

    dataset3 = im2spec_Dataset(images, spectra, transform = transform2)


    # Combine the datsets
    dataset = ConcatDataset([dataset1, dataset2, dataset3])

    return dataset



class Error_Dataset(Dataset):

    """
    PyTorch dataset for image-patch/error-target pairs.

    Args:
        images: Image-patch array of shape `(N, H, W)`.
        err_vector: Error target array. A 1D array is treated as one target per
            image; a higher-dimensional array is treated as ensemble targets.
        transform: Optional image transform applied after conversion to tensor.
    """
    def __init__(self, images, err_vector, transform = None):
        """Store image patches, error targets, and an optional transform."""
        self.images = images
        self.error = err_vector
        self.transform = transform

    def __len__(self):

        """Return the number of image/error pairs."""
        return len(self.images)

    def __getitem__(self, idx):

        """
        Return one `(image, error)` sample as float tensors.

        Args:
            idx: Sample index.

        Returns:
            Tuple `(image, error)`.
        """
        image = torch.tensor(self.images[idx], dtype=torch.float32)

        if self.transform is not None:
            image = self.transform(image)

        # if error is output of a single error model
        if len(self.error.shape) == 1:
            error_data = self.error[:, np.newaxis]
            error = torch.tensor(error_data[idx], dtype=torch.float32)

        # if error is output of an ensemble model
        else:
            error = torch.tensor(self.error[idx], dtype=torch.float32).unsqueeze(1)

        return image, error


def paired_images_spectra(image, hyperspectra, window_size = 30, coordinate_step = 10, image_norm = False, 
                          spectra_norm = False):


    """
    Pair image patches with spectra sampled on a resized hyperspectral grid.
    Use for BEPS data.

    The hyperspectral cube is resized in each spectral channel to match the
    square grid implied by the extracted patches, then flattened into one
    spectrum per image patch.

    Args:
        image: 2D morphology image.
        hyperspectra: 3D hyperspectral array with spectra along the last axis.
        window_size: Width and height of each extracted image patch in pixels.
        coordinate_step: Pixel spacing between neighboring patch centers.
        image_norm: If True, normalize each patch to `[0, 1]`.
        spectra_norm: If True, normalize each spectrum to `[0, 1]`.

    Returns:
        Tuple `(patches, training_spectra, coordinates)` where `patches` has
        shape `(N, window_size, window_size)`, `training_spectra` has shape
        `(N, S)`, and `coordinates` contains patch-center coordinates.
    """
    import cv2

    coords = get_coord_grid(image, step = coordinate_step, return_dict= False)
    #print('initial coordinates = ',coords[:, 0].shape)


    # Extract patches (or features) and the center coordinates of each patch.
    extracted_features = extract_subimages(image, coordinates = coords, window_size = window_size)
    patches, coordinates, _ = extracted_features
    patches = patches.squeeze()

    if image_norm:
        for i in range(len(patches)):
            patches[i] = norm_0to1(patches[i])

    #total number of pathces that are extracted
    n, _, _ = patches.shape


    n_dim = int(n**0.5)
    points = hyperspectra.shape[-1]

    #Reshape the training spectra to the same training set as the image patches
    training_spectra = np.zeros((n_dim, n_dim, points))

    for i in range(points):

        # Extract the spectra at the center of each patch
        training_spectra[:, :, i] = cv2.resize(hyperspectra[:, :, i], (n_dim, n_dim))

    # Reshape the training spectra so that each row is a spectra
    training_spectra = training_spectra.reshape(n, -1)

    if spectra_norm:
        for i in range(len(training_spectra)):
            training_spectra[i] = norm_0to1(training_spectra[i])


    return patches, training_spectra, coordinates


def paired_images_spectra_1(image, cits_obj, hyperspectra, window_size = 30, coordinate_step = 10, 
                            image_norm = False, spectra_norm = False):


    """
    Pair image patches with the nearest measured CITS spectrum. Use for STM data.

    Patch-center image coordinates are converted to the CITS frame size, then
    matched to the nearest measured CITS point before retrieving spectra.

    Args:
        image: 2D morphology image.
        cits_obj: `CITS_Analysis` object used for frame size and nearest-point
            lookup.
        hyperspectra: 3D hyperspectral array with spectra along the last axis.
        window_size: Width and height of each extracted image patch in pixels.
        coordinate_step: Pixel spacing between neighboring patch centers.
        image_norm: If True, normalize each patch to `[0, 1]`.
        spectra_norm: If True, normalize each retrieved spectrum to `[0, 1]`
            before the final global normalization.

    Returns:
        Tuple `(patches, training_spectra, coordinates)` where `patches` has
        shape `(N, window_size, window_size)`, `training_spectra` has shape
        `(N, S)`, and `coordinates` contains patch-center image coordinates.
    """

    coords = get_coord_grid(image, step = coordinate_step, return_dict= False)
    #print('initial coordinates = ',coords[:, 0].shape)


    # Extract patches (or features) and the center coordinates of each patch.
    extracted_features = extract_subimages(image, coordinates = coords, window_size = window_size)
    patches, coordinates, _ = extracted_features
    patches = patches.squeeze()


    if image_norm:
        for i in range(len(patches)):
            patches[i] = norm_0to1(patches[i])

    #total number of pathces that are extracted
    n, _, _ = patches.shape



    scan_frame = cits_obj.get_frame_size()

    image_pixels = image.shape[0]
    training_spectra = []

    for i in range(len(coordinates)):
        coordinate_point = coordinates[i]*scan_frame/image_pixels
        coord_val, cits_coord = cits_obj.nearest_point(coordinate_point)
        spectra = hyperspectra[cits_coord[0], cits_coord[1], :]

        if spectra_norm:
            spectra = norm_0to1(spectra)
        training_spectra.append(spectra)


    training_spectra = np.asarray(training_spectra)

    #Normalized globally over the label set
    training_spectra = norm_0to1(training_spectra)


    # Reshape the training spectra so that each row is a spectra
    training_spectra = training_spectra.reshape(n, -1)


    return patches, training_spectra, coordinates



def norm_0to1(arr, eps=1e-8):
    """
    Normalize an array to the `[0, 1]` range.

    Args:
        arr: Input array.
        eps: Minimum range before treating the array as constant.

    Returns:
        Normalized NumPy array.
    """
    arr = np.asarray(arr)
    arr_min = arr.min()
    arr_range = arr.max() - arr_min
    if arr_range < eps:
        print(f"Warning: norm_0to1 array range {arr_range} < eps {eps}; returning zeros.")
        return np.zeros_like(arr, dtype=float)
    return (arr - arr_min) / arr_range


class AddGaussianNoise():
    """
    Add Gaussian noise to a tensor.

    Args:
        noise_factor: Multiplicative scale applied to sampled noise.
        mean: Mean of the Gaussian noise distribution.
        std: Standard deviation of the Gaussian noise distribution.
    """

    def __init__(self, noise_factor=0.1, mean=0.0, std=1.0):

        """Store the Gaussian noise parameters."""
        self.noise_factor = noise_factor
        self.mean = mean
        self.std = std


    def __call__(self, tensor):

        """
        Add sampled Gaussian noise to a tensor.

        Args:
            tensor: Input tensor.

        Returns:
            Tensor with additive Gaussian noise.
        """
        return tensor + self.noise_factor * torch.normal(mean=self.mean, std=self.std, size=tensor.size())
