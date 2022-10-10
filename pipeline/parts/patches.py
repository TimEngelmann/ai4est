from patchify import patchify
from boundary import make_image
import numpy as np


def make_grid(img_shape : np.ndarray, patch_size : int) -> np.ndarray:
    """
    Takes as input an RGB image and returns coordinates for tiles
    in the image
    """

    n_rows = img_shape[1] // patch_size
    n_cols = img_shape[2] // patch_size

    grid_coords = np.zeros((n_rows, n_cols, 4, 2), dtype=int)

    for i in range(n_rows):
        for j in range(n_cols):
            grid_coords[i, j, 0, :] = np.array([patch_size * (i+1), patch_size * j])
            grid_coords[i, j, 1, :] = np.array([patch_size * i, patch_size * j])
            grid_coords[i, j, 2, :] = np.array([patch_size * i, patch_size * (j+1)])
            grid_coords[i, j, 3, :] = np.array([patch_size * (i+1), patch_size * (j+1)])

    return grid_coords


def pad(img:np.ndarray, patch_size:int) -> np.ndarray:
    height, width = img.shape[1:3]
    added_height = ((height // patch_size) + 1) * patch_size - height
    added_width = ((width // patch_size) + 1) * patch_size - width

    assert added_height >= 0
    assert added_width >= 0

    return np.pad(img,((0,0),(0, added_height), (0, added_width)), constant_values=0)


def save_patches(site:str, patch_size:int, path:str):
    """
    Function which takes a site as input, loads the corresponding image,
    splits it into patches and saves the output in the folder
    specified by path.
    Input:
        site : The site name as found in the field data csv file
        patch_size : The size of the square patches in pixels
        path : The path to the folder where patches will be saved
    """
    coords = make_grid(img.shape, patch_size)
    img = make_image(site)
    padded_img = pad(img, patch_size)

    n_rows = img.shape[1] // patch_size
    n_cols = img.shape[2] // patch_size

    count = 0
    for i in range(n_rows):
        for j in range(n_cols):
            x_min, y_min = coords[i, j, 1, :]
            patch = img[:, x_min:(x_min + patch_size), y_min:(y_min + patch_size)]
            np.save(path + f"{site}_count.np")
            count += 1


def test():
    patch_size = 400
    path = "/home/jan/sem1/ai4good/ai4est/dataset/"
    site = "Manuel Macias RGB"

    img = make_image(site)
    save_patches(site, img, patch_size, path)


if __name__ == "__main__":
    test()