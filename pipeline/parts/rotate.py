import numpy as np
import pandas as pd
import imutils

def get_inverse_rotation(angle):
    radians = angle * 2 * np.pi / 360

    rot = np.diag([np.cos(radians), np.cos(radians), 1])
    rot[0,1] =  -np.sin(radians)
    rot[1,0] = np.sin(radians)

    return rot

def get_shift_pos(center):
    shift = np.eye(3)
    shift[0,2] = center[0]
    shift[1,2] = center[1]

    return shift

def get_shift_neg(center):
    shift = np.eye(3)
    shift[0,2] = -center[0]
    shift[1,2] = -center[1]

    return shift

def rotate_grid(grid:np.array, rotation:float, img_shape:tuple) -> np.array:
    """
    Takes the patch coordinates created in the make_grid
    function and transforms them back to the original
    coordinate system.
    Inputs:
        patches : Numpy array as created by make_grid
        rotation : The rotation angle in degrees
        img_shape : The shape of the original image
    Return:
        np.array : The array with the same shape as patches
            containing the transformed coordinates.
    """
    #Turning the coordinates into affine coordinates
    affine_coords = np.zeros((grid.shape[0],grid.shape[1], 3))
    affine_coords[:, :, :2] = grid
    affine_coords[:, :, 2] = np.ones((grid.shape[0], grid.shape[1]))

    center = ((np.array(img_shape) - 1) / 2).astype(int) #center of image

    rotation_matrix = get_inverse_rotation(rotation) #affine transformation rotating around center
    pos_shift = get_shift_pos(center) #shifting coordinate system from center to top left
    neg_shift = get_shift_neg(center) #shifting coordinate system from top left to center

    transformation = pos_shift @ rotation_matrix @ neg_shift #combining transformations
    original_affine = np.einsum("kl, ijl -> ijk", transformation, affine_coords) #applying transformation

    return original_affine[:,:,:2].astype(int)


def get_edge_points(img_shape:tuple) -> np.array:
    """
    Function which returns the edge points of a
    multi-channel image, stored as a 3d array with the
    first coordinate being the channels
    Input:
        img_shape : The shape of the image given as by the
            numpy.shape function
    Return:
        4 x 2 numpy array containing the edge points of the
            image.
    """
    height, width = img_shape[1:]
    edge_points = np.zeros((4, 2))
    edge_points[1,0] = height
    edge_points[2,:] = np.array([height, width])
    edge_points[3, 1] = width

    return edge_points

def rotate_img(img, angle):
    center = (np.array(img.shape[1:]) - 1) / 2
    return np.moveaxis(imutils.rotate(np.moveaxis(img, 0, -1), angle, center=center), -1, 0)

def rotate_distribution(d, angle):
    center = (np.array(d.shape) - 1) / 2
    return imutils.rotate(d, angle, center)



if __name__ == "__main__":
    test()