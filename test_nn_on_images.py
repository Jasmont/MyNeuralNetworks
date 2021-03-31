import os
import time
import cv2 as cv
import tensorflow as tf
import numpy as np

CENTROID_DIST = 75


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


model_path = r'E:\py\VA_BEER_CAN_NUMBERS_NN\inceptionresnetv2_03-26-2021_16-06-59\inceptionresnetv2_model.h5'

model = tf.keras.models.load_model(model_path)
os.mkdir('\\'.join(model_path.split('\\')[:-1]) + '\\wrongs')
labels = {
    0: 11,
    1: 12,
    2: 13,
    3: 14,
    4: 15,
    5: 17,
    6: 18,
    7: 19
}

current_dir = os.getcwd() + '\\pictures'

for r, d, f in os.walk(current_dir):
    for file in f:
        if file.endswith('.bmp'):
            img = cv.imread(r + '\\' + file, cv.IMREAD_COLOR)
            test_img = scale_and_get_roi_from_circular_area(img.copy()) / 255.
            prediction = model.predict(np.asarray([test_img]))
            prediction = labels[int(np.argmax(prediction))]
            if int(r[-4:-2]) != prediction:
                cv.imwrite('\\'.join(model_path.split('\\')[:-1]) + '\\wrongs\\' + str(r[-4:-2]) + '_' + str(
                    time.time()) + '.bmp', img)
                print('Found one!')
