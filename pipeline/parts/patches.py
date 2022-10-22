import numpy as np
import pandas as pd

def create_upper_left(img_shape:np.ndarray, patch_size:int) -> np.ndarray:
    """
    Takes as input the shape of a multi-channel image and
    returns the upper left corner of each patch.
    Inputs:
        img_shape : The shape of the image with first
            coordinate representing the channels and the
            latter coordinates representing space
        patch_size : The desired size for the square
            patches
    """

    n_rows = img_shape[1] // patch_size
    n_cols = img_shape[2] // patch_size

    count = 0
    grid_coords = np.zeros((n_rows*n_cols, 2))
    for i in range(n_rows):
        for j in range(n_cols):
            grid_coords[n_cols * i + j, :] = np.array([patch_size * i, patch_size * j])

    return grid_coords.astype(int)


def make_grid(img_shape:np.ndarray, patch_size:int) -> np.ndarray:
    """
    Takes as input the shape of a multi-channel image and
    returns coordinates for tiles in the image.
    The patches are specified by the four courners in the
    order [0,0], [patch_size, 0], [patch_size, patch_size],
    [0, patch_size]
    Inputs:
        img_shape : The shape of the image with first
            coordinate representing the channels and the
            latter coordinates representing space
        patch_size : The desired size for the square
            patches
    """

    n_rows = img_shape[1] // patch_size
    n_cols = img_shape[2] // patch_size

    patches = pd.DataFrame([], columns=["site", "vertices"])

    count = 0
    grid_coords = np.zeros((n_rows*n_cols, 4, 2))
    for i in range(n_rows):
        for j in range(n_cols):
            grid_coords[n_cols * i + j, 0, :] = np.array([patch_size * i, patch_size * j])
            grid_coords[n_cols * i + j, 1, :] = np.array([patch_size * (i+1), patch_size * j])
            grid_coords[n_cols * i + j, 2, :] = np.array([patch_size * (i+1), patch_size * (j+1)])
            grid_coords[n_cols * i + j, 3, :] = np.array([patch_size * i, patch_size * (j+1)])

    return grid_coords.astype(int)

def convert_coordinates_to_df(grid_coords:np.array, site:str) -> pd.DataFrame:
    """
    Takes an 3d numpy array of shape n_patches x 4 x 2
    which contains the 4 vertices of each patch and turns
    it into a pandas DataFrame with each row containing
    the name of the site and the a 4 x 2 numpy array
    containing the vertices of each patch, as required by
    the estimate_agb function
    Inputs:
        grid_coords : The array containing the vertices.
            Should be of shape n x 4 x 2.
        site : The name of the site
    Returns:
        pd.DataFrame : The dataframe containing a row for
            each patch with the site name and vertices
            as 4 x 2 array.
    """
    n_patches = grid_coords.shape[0]
    patches = pd.DataFrame([], columns=["site", "vertices"])
    patches["vertices"] = pd.Series([grid_coords[i,] for i in range(n_patches)])
    patches["site"] = site

    return patches

def pad(img:np.ndarray, patch_size:int) -> np.ndarray:
    """
    Pads an image so that the pixel width and lengths are
    divisible by the patch_size, so the patches will be of
    even size.
    Inputs:
        img : Array representing a multi-channel image
            with channel data in the first coordinate and
            spatial data in the 2 latter coordinates
        patch_size : The desired patch size
    """
    height, width = img.shape[1:3]

    #computing pixels added during padding
    added_height = ((height // patch_size) + 1) * patch_size - height
    added_width = ((width // patch_size) + 1) * patch_size - width

    img = np.pad(img,((0,0),(0, added_height), (0, added_width)), constant_values=0)

    #sanity check
    assert img.shape[1] // patch_size
    assert img.shape[2] // patch_size

    return img

def remove_out_of_bounds(grid_coords:np.array, img_shape:tuple) -> np.array:
    """
    Removes the patches for which any coordinate is
    outside the image.
    """
    shape = np.array(img_shape[1:])
    n_patches = grid_coords.shape[0]
    upper_bounds = np.tile(shape, n_patches * 4).reshape(n_patches, 4, 2)
    inbounds =  (0 <= grid_coords) & (grid_coords <= upper_bounds)
    inbounds_patchwise = inbounds.all(axis=(1,2))

    return grid_coords[inbounds_patchwise,:,:]

def filter_grid(grid_coords:np.array, img:np.array) -> np.array:
    """
    Takes the coords of the patches and the image and
    filters out empty patches
    Inputs:
        grid_coords : An n x 4 x 2 array with n the number
            of patches.
        img : The image as an array of the form C x H x W
    Return:
        np.array : Same shape as grid_coords, but the empty
            patches are removed.
    """



    raise NotImplementedError()

def test():
    patch_size = 400
    path = "/home/jan/sem1/ai4good/ai4est/dataset/"
    site = "Manuel Macias RGB"

    img = make_image(site)
    save_patches(site, img, patch_size, path)


if __name__ == "__main__":
    test()