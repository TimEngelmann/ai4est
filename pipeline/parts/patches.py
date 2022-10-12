import numpy as np
import pandas as pd

def make_grid(site:str, img_shape:np.ndarray, patch_size:int) -> np.ndarray:
    """
    Takes as input the shape of a multi-channel image and
    returns coordinates for tiles in the image.
    Inputs:
        site : The name of the site as found in the field
         data of the reforestree dataset
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
    for i in range(n_rows):
        for j in range(n_cols):
            grid_coords = np.zeros((4, 2), dtype=int)
            grid_coords[0, :] = np.array([patch_size * (i+1), patch_size * j])
            grid_coords[1, :] = np.array([patch_size * i, patch_size * j])
            grid_coords[2, :] = np.array([patch_size * i, patch_size * (j+1)])
            grid_coords[3, :] = np.array([patch_size * (i+1), patch_size * (j+1)])

            patches.loc[len(patches.index)] = [site, grid_coords]

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


def save_patches(site:str, patch_size:int, path_to_data:str, path:str):
    """
    Function which takes a site as input, loads the corresponding image,
    splits it into patches and saves the output in the folder
    specified by path.
    Input:
        site : The site name as found in the field data csv file
        patch_size : The size of the square patches in pixels
        path_to_data : The path to the reforstree folder containing
            the raw data, must be of the form ".../reforestree/"
        path : The path to the folder where patches will be saved
            must be of the form ".../folder_for_patches/"
    """
    img = make_image(site, path_to_data)
    coords = make_grid(img.shape, patch_size)
    padded_img = pad(img, patch_size)

    n_rows = img.shape[1] // patch_size
    n_cols = img.shape[2] // patch_size

    count = 0
    for i in range(n_rows):
        for j in range(n_cols):
            x_min, y_min = coords[i, j, 1, :]
            patch = img[:, x_min:(x_min + patch_size), y_min:(y_min + patch_size)]
            np.save(path + f"{site}_{count}", patch)
            count += 1


def test():
    patch_size = 400
    path = "/home/jan/sem1/ai4good/ai4est/dataset/"
    site = "Manuel Macias RGB"

    img = make_image(site)
    save_patches(site, img, patch_size, path)


if __name__ == "__main__":
    test()