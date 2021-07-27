"""
Recovers a stereo pair from a given anaglyph
"""
import cv2 as cv

import cv2 as cv
import numpy as np
from utils import Cl, cprint, check_and_create_dir
import reverse
from config import *

import utils
from utils import Cl, cprint
from split_anaglyph_channels import get_anaglyph_channels
from representations import get_image_representations
from datacost import get_data_costs
from aggregation import get_aggregation
from disparityMap import get_disparity_map
from reciprocity import get_reciprocity
from recover_from_disparity import recover
from refinement import get_refinement
from colorization import colorize

def reverse(anaglyph, min_disp=1, max_disp=None):
    """
    Recovers a stereo pair from a given anaglyph
    """

    # Used to benchmark
    my_timer = utils.MyTimer()
    time_records = {}
    pickle_data = {}

    # If max_disp is not given, set it to 1/5 of image width
    if max_disp is None:
        max_disp = int(anaglyph.shape[1]/5)

    cprint("    STEP x TIME", "Partial / Total", Cl.WARNING, Cl.WARNING)

    # STEP 1 - Opens anaglyph and splits its channels
    # Time complexity - O(1)
    l_channel, r_channel, _, _ = get_anaglyph_channels(anaglyph)
    time_records["load"] = my_timer.get_actual_time("    01 - Anaglyph")
    cv.destroyAllWindows()

    # STEP 2 - Get representations
    # Time complexity - O(N.w²) - N=Number of pixels, w=Window size
    l_representations, r_representations \
        = get_image_representations(l_channel, r_channel)
    time_records["representation"] = my_timer.get_actual_time("    02 - Representations")
    cv.destroyAllWindows()

    # STEP 3 - Get data costs volume
    # Time complexity - O(L.N.w²) - N=Number of pixels, w=Window size, L=Number of disparities
    l_costs, r_costs \
        = get_data_costs(l_channel, r_channel, \
        l_representations, r_representations, min_disp, max_disp)
    time_records["dataCost"] = my_timer.get_actual_time("    03 - Data Costs")
    cv.destroyAllWindows()

    # STEP 4 - Aggregate
    # Time complexity - O(L.N) - N=Number of pixels, L=Number of disparities
    l_aggregated_costs, r_aggregated_costs \
        = get_aggregation(l_costs, r_costs, min_disp, max_disp)
    time_records["aggregation"] = my_timer.get_actual_time("    04 - Aggregation")
    cv.destroyAllWindows()

    # STEP 5 - Computes Initial Disparity map
    # Time complexity - O(1)
    l_disparity_map, r_disparity_map \
        = get_disparity_map(l_aggregated_costs, r_aggregated_costs)
    time_records["disparitypMap"] = my_timer.get_actual_time("    05 - Disparity Maps")
    cv.destroyAllWindows()

    # STEP 6 - Reciprocity Mask from Disparity Map
    # Time complexity - O(N) - N=Number of pixels
    l_valid_disparity_map, r_valid_disparity_map, l_reciprocity_mask, r_reciprocity_mask \
        = get_reciprocity(l_disparity_map, r_disparity_map)
    time_records["reciprocity"] = my_timer.get_actual_time("    06 - Reciprocity")
    cv.destroyAllWindows()

    # STEP 7 - Disparity map refinement
    # Time complexity - O(N) - N=Number of pixels
    l_refined_disparity_map, r_refined_disparity_map, \
        l_refined_reciprocity_mask, r_refined_reciprocity_mask \
        = get_refinement(l_valid_disparity_map, r_valid_disparity_map,
                         l_reciprocity_mask, r_reciprocity_mask)
    time_records["refinement"] = my_timer.get_actual_time("    07 - Refinement")
    cv.destroyAllWindows()

    # STEP 8 - Recover Images from disparity maps
    # Time complexity - O(N) - N=Number of pixels
    l_recovered, r_recovered \
        = recover(anaglyph, l_refined_disparity_map, r_refined_disparity_map)
    time_records["recover"] = my_timer.get_actual_time("    08 - Recovered Image")
    cv.destroyAllWindows()

      # STEP 9 - Colorization
    # Time complexity - O(I) - I=Number of invalid pixels
    l_colorized, r_colorized \
        = colorize(anaglyph, l_recovered, r_recovered, \
                   l_refined_reciprocity_mask, r_refined_reciprocity_mask)
    time_records["colorize"] = my_timer.get_actual_time("    09 - Colorization")
    cv.destroyAllWindows()

    # Save outputs to pickle file for reusing
    pickle_data["anaglyph"] = anaglyph
    pickle_data["valid_disparity_map"] = (l_valid_disparity_map, r_valid_disparity_map)
    pickle_data["reciprocity"] = (l_reciprocity_mask, r_reciprocity_mask)
    pickle_data["refined_disparity_map"] = (l_refined_disparity_map, r_refined_disparity_map)
    pickle_data["refined_reciprocity"] = (l_refined_reciprocity_mask, r_refined_reciprocity_mask)
    pickle_data["recovered"] = (l_recovered, r_recovered)
    pickle_data["colorized"] = (l_colorized, r_colorized)

    return l_colorized, r_colorized, pickle_data, time_records



if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description="Computes the Peak Signal-to-noise Ratio")
    parser.add_argument("base_path", help="Anaglyph input path",
                        type=str)
    parser.add_argument("filename", help="Anagliph filename (without extension). Only bmp files are accepted.",
                        type=str)     

    args = parser.parse_args()

    base_path = args.base_path
    filename = args.filename

    # Loads anaglyph
    anaglyph = cv.imread("{}{}.bmp".format(base_path, filename))

    # Prints details
    cprint("    Recovering Anaglyph", " {}".format(filename), Cl.HEADER)
    cprint("    shape ", "{} - {} pixels"
            .format(anaglyph.shape, anaglyph.shape[0]*anaglyph.shape[1]), Cl.OKBLUE)
    print("     ")
    cprint("    anaglyph_config", ANAGLYPH_CONFIG, Cl.OKBLUE)
    cprint("    representation_config", REPRESENTATION_CONFIG, Cl.OKBLUE)
    cprint("    aggregation_config", AGGREGATION_CONFIG, Cl.OKBLUE)
    cprint("    reciprocity_config", RECIPROCITY_CONFIG, Cl.OKBLUE)
    cprint("    refinement_config", REFINEMENT_CONFIG, Cl.OKBLUE)
    cprint("    colorization_config", COLORIZATION_CONFIG, Cl.OKBLUE)
    print("     ")

    # Reverse!
    l_colorized, r_colorized, pickle_data, time_records = \
        reverse(anaglyph)
    colorized = np.hstack((l_colorized, r_colorized))

    SAVE_PATH = "{}/results".format(base_path)
    check_and_create_dir(SAVE_PATH)

    cv.imwrite("{}/L_{}.bmp".format(SAVE_PATH, filename), l_colorized)
    cv.imwrite("{}/R_{}.bmp".format(SAVE_PATH, filename), r_colorized)
    cv.imwrite("{}/{}.bmp".format(SAVE_PATH, filename), colorized)

    # cv.imshow("FINAL", colorized)
    # cv.waitKey(0)

    print("\n\n\n")
