import numpy as np
import cv2 as cv
from scipy.ndimage import gaussian_filter
import utils
from config import AGGREGATION_CONFIG, SHOW_RESULTS, SHOW_OFF_MODE

WAITKEY_TIME = 1
if SHOW_OFF_MODE:
    WAITKEY_TIME = 0

def get_aggregation(l_costs, r_costs, min_disp, max_disp):
    """
    Performs aggregation via gaussian blur over each data cost image
    # window size = 2*int(truncate*sigma + 0.5) + 1

    Args:
        l_costs (float np.array): Left side data costs array
        r_costs (float np.array): Right side data costs array
        min_disp (int): Minimum disparity value
        max_disp (int): Maximum disparity value

    Returns:
        l_aggregated_costs, r_aggregated_costs (float np.array): Tuple containing the aggregated
        data costs
    """

    sigma = AGGREGATION_CONFIG["sigma"]
    truncate = AGGREGATION_CONFIG["truncate"]

    # Variables to store aggregated costs - O(1)
    l_aggregated_costs = np.ones(l_costs.shape)*float('inf')
    r_aggregated_costs = np.ones(r_costs.shape)*float('inf')

    # At each disparity level - O(L.N)
    for disp in range(min_disp, max_disp+1):

        # Aggregates image cost - O(N)
        l_aggregated_costs[disp] = gaussian_filter(l_costs[disp], sigma=sigma, truncate=truncate)
        r_aggregated_costs[disp] = gaussian_filter(r_costs[disp], sigma=sigma, truncate=truncate)

        if SHOW_RESULTS:
            cv.imshow("Partial Disparity Map",
                      utils.stack_1_by_2(np.argmin(l_aggregated_costs, 0),
                                         np.argmin(r_aggregated_costs, 0),
                                         convert_image=True))
            utils.show_image_stack("Aggregated Data Costs",
                                   utils.stack_1_by_2(l_aggregated_costs, r_aggregated_costs),
                                   min_val=1, max_val=max_disp, initial_pos=disp)
            utils.show_image_stack("Data Costs",
                                   utils.stack_1_by_2(l_costs, r_costs),
                                   min_val=1, max_val=max_disp, initial_pos=disp)
            cv.waitKey(2)

    # if SHOW_RESULTS:
    #     cv.imshow("Aggregated Costs (Click for details)", l_channel)
    #     cv.setMouseCallback("Aggregated  Costs (Click for details)", utils.preview_array,
    #                         param=(l_aggregated_costs, r_aggregated_costs, "SAD", "Disparidade"))
    #     cv.waitKey(WAITKEY_TIME)

    return l_aggregated_costs, r_aggregated_costs
