import time
import os
import numpy as np
import shutil
import cv2 as cv

CENTROID_DIST = 75

SOURCE_DIR = r'E:\py\VA_BEER_CAN_NUMBERS_NN\pictures'
NEW_DIR = os.getcwd() + r'\extracted'


os.mkdir(NEW_DIR)


def scale_and_get_roi_from_circular_area(image):
    # Downscale it
    scale_percent = 0.5
    width = int(image.shape[1] * scale_percent)
    height = int(image.shape[0] * scale_percent)
    output = cv.resize(image.copy(), (width, height))

    # Find circle / middle point
    circles = cv.HoughCircles(cv.cvtColor(output.copy(), cv.COLOR_BGR2GRAY), cv.HOUGH_GRADIENT, 1, 50)
    circles = np.uint16(np.around(circles))[0][0]

    original_result = output.copy()[circles[1] - CENTROID_DIST:circles[1] + CENTROID_DIST,
                      circles[0] - CENTROID_DIST:circles[0] + CENTROID_DIST]

    return original_result


for i in os.listdir(SOURCE_DIR):
    if i[:2] not in os.listdir(NEW_DIR):
        new_dir = NEW_DIR + "\\" + i[:2]
        os.mkdir(new_dir)

    for j in os.listdir(SOURCE_DIR + "\\" + i):
        img = cv.imread(SOURCE_DIR + "\\" + i + "\\" + j, cv.IMREAD_COLOR)
        img = scale_and_get_roi_from_circular_area(img)
        cv.imwrite(NEW_DIR + "\\" + i[:2] + "\\" + f"{i[:2]}_{time.time()}.bmp", img=img)

# # Creating Train / Val / Test folders (One time use)
classes_dir = [i for i in os.listdir(NEW_DIR)]  # total labels

val_ratio = 0.15
test_ratio = 0.15

for cls in classes_dir:
    os.makedirs(NEW_DIR + '../../data/train/' + cls)
    os.makedirs(NEW_DIR + '../../data/val/' + cls)
    os.makedirs(NEW_DIR + '../../data/testing/' + cls)

    # Creating partitions of the data after shuffeling
    src = NEW_DIR + '\\' + cls  # Folder to copy images from

    allFileNames = os.listdir(src)
    np.random.shuffle(allFileNames)
    train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                              [int(len(allFileNames) * (1 - (val_ratio + test_ratio))),
                                                               int(len(allFileNames) * (1 - test_ratio))])

    train_FileNames = [src + '/' + name for name in train_FileNames.tolist()]
    val_FileNames = [src + '/' + name for name in val_FileNames.tolist()]
    test_FileNames = [src + '/' + name for name in test_FileNames.tolist()]

    print('Class', cls)
    print('Total images: ', len(allFileNames))
    print('Training: ', len(train_FileNames))
    print('Validation: ', len(val_FileNames))
    print('Testing: ', len(test_FileNames))

    # Copy-pasting images
    for name in train_FileNames:
        shutil.copy(name, NEW_DIR + '../../data/train/' + cls)

    for name in val_FileNames:
        shutil.copy(name, NEW_DIR + '../../data/val/' + cls)

    for name in test_FileNames:
        shutil.copy(name, NEW_DIR + '../../data/testing/' + cls)
