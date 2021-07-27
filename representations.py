"""
Returns a representation vector array for each image pixel
"""
import numpy as np
import cv2 as cv
import utils
from config import REPRESENTATION_CONFIG, SHOW_OFF_MODE, SHOW_RESULTS

WAITKEY_TIME = 1
if SHOW_OFF_MODE:
    WAITKEY_TIME = 0

def get_image_representations(l_channel, r_channel):
    """
    Returns a representation vector array for each image pixel

    Parameters:
    l_channel (uint8 np.array): Left channel imange
    r_channel (uint8 np.array): Right channel imange

    Returns:
    l_representations, r_representations = (float cv2 image np.array):
    """

    mode = REPRESENTATION_CONFIG["mode"]
    window_size = REPRESENTATION_CONFIG["window_size"]
    y_axis, x_axis = l_channel.shape
    half_window_size = int((window_size-1)/2)

    # Make border in order to fit window - O(1)
    l_channel_border = cv.copyMakeBorder(l_channel,
                                         half_window_size, half_window_size,
                                         half_window_size, half_window_size,
                                         cv.BORDER_REFLECT_101)

    r_channel_border = cv.copyMakeBorder(r_channel,
                                         half_window_size, half_window_size,
                                         half_window_size, half_window_size,
                                         cv.BORDER_REFLECT_101)

    # Stack channels to work faster with numpy - O(1)
    image = np.dstack((l_channel_border, r_channel_border))

    # Convert image to be in the range (0-1) - O(1)
    dtype_info = np.iinfo(image.dtype)
    image = utils.change_range(image.astype("float"), dtype_info.min, dtype_info.max, 0, 1)

    # Creates np.array to store representations - O(1)
    representation_size = len(get_block_representation(image[:window_size, :window_size], mode)[0])
    l_representations = np.empty([y_axis, x_axis, representation_size])
    r_representations = np.empty([y_axis, x_axis, representation_size])

    # For each pixel p - O(x.y.w.w) -> O(N.w²)
    for y in range(y_axis):
        utils.progress_bar(y, y_axis, text="Representation Computation")
        for x in range(x_axis):

            # Get the representations for the block around p
            block = image[y:y+window_size, x:x+window_size]
            l_rep, r_rep = get_block_representation(block, mode)

            # Store to array
            l_representations[y, x] = l_rep
            r_representations[y, x] = r_rep

    if SHOW_RESULTS:
        cv.imshow("Representations (Click for details)", l_channel)
        cv.setMouseCallback("Representations (Click for details)", utils.preview_array,
                            param=(l_representations, r_representations, "Representação", ""))
        cv.waitKey(WAITKEY_TIME)

    return l_representations, r_representations


def get_block_representation(block, mode):
    """
    Returns the block representation for a given mode
    """

    if mode == "FullBlock":

        l_representations, r_representations = utils.split_block_channels(block)

        return l_representations, r_representations

    elif mode == "NormalizedFullBlock":

        l_representations, r_representations = utils.split_block_channels(block)

        l_representations = utils.normalize(l_representations, 0, 1)
        r_representations = utils.normalize(r_representations, 0, 1)

        return l_representations, r_representations

    elif mode == "GaussianFullBlock":

        l_representations, r_representations = utils.split_block_channels(block)

        window_size = block.shape[0]
        gauss = utils.get_gaussian_block(window_size, percent=True)

        l_representations = gauss.flatten()*l_representations
        r_representations = gauss.flatten()*r_representations

        return l_representations, r_representations

    elif mode == "NormalizedGaussianFullBlock":

        l_representations, r_representations = utils.split_block_channels(block)

        window_size = block.shape[0]
        gauss = utils.get_gaussian_block(window_size, percent=True)

        l_representations = utils.normalize(gauss.flatten()*l_representations)
        r_representations = utils.normalize(gauss.flatten()*r_representations)

        return l_representations, r_representations

    elif mode == "Differential":

        vertical = block[1:, :, :] - block[:-1, :, :]
        horizontal = block[:, 1:, :] - block[:, :-1, :]

        l_representations = np.concatenate((horizontal[:, :, 0].flatten(),
                                            vertical[:, :, 0].flatten()))
        r_representations = np.concatenate((horizontal[:, :, 1].flatten(),
                                            vertical[:, :, 1].flatten()))

        return l_representations, r_representations

    elif mode == "DifferentialCrop":

        # O(2.w.(w-1)) -> O(w²)
        vertical = block[1:, :, :] - block[:-1, :, :]
        horizontal = block[:, 1:, :] - block[:, :-1, :]

        l_representations = np.concatenate((horizontal[:, :, 0].flatten(),
                                            vertical[:, :, 0].flatten()))

        r_representations = np.concatenate((horizontal[:, :, 1].flatten(),
                                            vertical[:, :, 1].flatten()))

        l_representations = np.where(l_representations < 0,
                                     l_representations + 1, l_representations)

        r_representations = np.where(r_representations < 0,
                                     r_representations + 1, r_representations)

        return l_representations, r_representations

    elif mode == "NormalizedDifferential":

        vertical = block[1:, :, :] - block[:-1, :, :]
        horizontal = block[:, 1:, :] - block[:, :-1, :]

        l_representations = np.concatenate((horizontal[:, :, 0].flatten(),
                                            vertical[:, :, 0].flatten()))

        r_representations = np.concatenate((horizontal[:, :, 1].flatten(),
                                            vertical[:, :, 1].flatten()))

        l_representations = utils.normalize(l_representations, 0, 1)
        r_representations = utils.normalize(r_representations, 0, 1)

        return l_representations, r_representations

    elif mode == "AbsoluteDifferential":

        vertical = block[1:, :, :] - block[:-1, :, :]
        horizontal = block[:, 1:, :] - block[:, :-1, :]

        l_representations = np.concatenate((horizontal[:, :, 0].flatten(),
                                            vertical[:, :, 0].flatten()))

        r_representations = np.concatenate((horizontal[:, :, 1].flatten(),
                                            vertical[:, :, 1].flatten()))

        l_representations = np.absolute(l_representations)
        r_representations = np.absolute(r_representations)

        return l_representations, r_representations

    elif mode == "ModuleDifferential":

        vertical = block[2:, :, :] - block[:-2, :, :]
        horizontal = block[:, 2:, :] - block[:, :-2, :]

        vertical = vertical[:, 1:-1, :]
        horizontal = horizontal[1:-1, :, :]

        module = np.sqrt(vertical*vertical + horizontal*horizontal)

        l_representations = module[:, :, 0].flatten()
        r_representations = module[:, :, 1].flatten()

        return l_representations, r_representations

    elif mode == "AngleDifferential":

        vertical = block[2:, :, :] - block[:-2, :, :]
        horizontal = block[:, 2:, :] - block[:, :-2, :]

        vertical = vertical[:, 1:-1, :]
        horizontal = horizontal[1:-1, :, :]

        phi = np.arctan2(vertical, horizontal)

        l_representations = phi[:, :, 0].flatten()
        r_representations = phi[:, :, 1].flatten()

        return l_representations, r_representations

    elif mode == "NormalizedAbsoluteDifferential":

        vertical = block[1:, :, :] - block[:-1, :, :]
        horizontal = block[:, 1:, :] - block[:, :-1, :]

        l_representations = np.concatenate((horizontal[:, :, 0].flatten(),
                                            vertical[:, :, 0].flatten()))

        r_representations = np.concatenate((horizontal[:, :, 1].flatten(),
                                            vertical[:, :, 1].flatten()))

        l_representations = utils.normalize(np.absolute(l_representations))
        r_representations = utils.normalize(np.absolute(r_representations))

        return l_representations, r_representations

    elif mode == "GaussianDifferential":

        window_size = block.shape[0]
        gauss = utils.get_gaussian_block(window_size, percent=True)

        block[:, :, 0] = (gauss*block[:, :, 0]).astype('float')
        block[:, :, 1] = (gauss*block[:, :, 1]).astype('float')

        vertical = block[1:, :, :] - block[:-1, :, :]
        horizontal = block[:, 1:, :] - block[:, :-1, :]

        l_representations = np.concatenate((horizontal[:, :, 0].flatten(),
                                            vertical[:, :, 0].flatten()))
        r_representations = np.concatenate((horizontal[:, :, 1].flatten(),
                                            vertical[:, :, 1].flatten()))

        return l_representations, r_representations
