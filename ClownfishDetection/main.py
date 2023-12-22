import cv2
import glob
import numpy as np
import os.path as osp
from argparse import ArgumentParser
from utils.compute_iou import compute_ious

LIGHT_ORANGE = [1, 190, 150]
DARK_ORANGE = [30, 255, 255]
LIGHT_WHITE = [60, 0, 200]
DARK_WHITE = [145, 150, 255]


def segment_fish(img):
    """
    This method should compute masks for given image
    Params:
        img (np.ndarray): input image in BGR format
    Returns:
        mask (np.ndarray): fish mask. should contain bool values
    """
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the first color filter
    lower_orange = np.array(LIGHT_ORANGE, dtype=np.uint8)
    upper_orange = np.array(DARK_ORANGE, dtype=np.uint8)

    # Define the lower and upper bounds for the second color filter
    lower_white = np.array(LIGHT_WHITE, dtype=np.uint8)
    upper_white = np.array(DARK_WHITE, dtype=np.uint8)

    # Create two masks using the color ranges
    orange_mask = cv2.inRange(hsv_image, lower_orange, upper_orange)
    white_mask = cv2.inRange(hsv_image, lower_white, upper_white)

    # Combine the two masks using a bitwise OR operation
    combined_mask = cv2.bitwise_or(orange_mask, white_mask)

    # return np.array(mask)
    return combined_mask


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--is_train", action="store_true")
    args = parser.parse_args()
    stage = 'train' if args.is_train else 'test'

    data_root = osp.join("dataset", stage, "imgs")
    img_paths = glob.glob(osp.join(data_root, "*.jpg"))
    len(img_paths)

    masks = dict()
    for path in img_paths:
        img = cv2.imread(path)
        mask = segment_fish(img)
        masks[osp.basename(path)] = mask

    print(compute_ious(masks, osp.join("dataset", stage, "masks")))
