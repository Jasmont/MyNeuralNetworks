import cv2 as cv
import tensorflow as tf
import numpy as np

import time


def scale_and_get_roi_from_circular_area(image):
    # Downscale it
    scale_percent = 0.5
    width = int(image.shape[1] * scale_percent)
    height = int(image.shape[0] * scale_percent)
    output = cv.resize(image.copy(), (width, height))

    # Find circle / middle point
    circles = cv.HoughCircles(cv.cvtColor(output.copy(), cv.COLOR_BGR2GRAY), cv.HOUGH_GRADIENT, 1, 50)
    circles = np.uint16(np.around(circles))[0][0]

    original_result = output.copy()[circles[1] - 75:circles[1] + 75,
                                    circles[0] - 75:circles[0] + 75]

    return original_result


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

guessed = {
    True: 0,
    False: 0
}


video_path = r'E:\py\VA_BEER_CAN_NUMBERS_NN\testing\test_videos\15_2_Basler_acA1920-40um.avi'
model_path = r'E:\py\VA_BEER_CAN_NUMBERS_NN\inceptionresnetv2_03-26-2021_16-06-59\inceptionresnetv2_model.h5'

model = tf.keras.models.load_model(model_path)
cap = cv.VideoCapture(video_path)

GUESSING = int(video_path.split('\\')[-1][:2])

out = cv.VideoWriter('\\'.join(model_path.split('\\')[:-1]) + '\\output_' + video_path.split('\\')[-1][:4] + '.avi',
                     cv.VideoWriter_fourcc(*'XVID'), 30.0, (1920, 1200))

while True:
    frame = cap.read()[1]

    if frame is None or cv.waitKey(1) == 27:
        break

    acquired_frame = time.time()

    img = scale_and_get_roi_from_circular_area(frame) / 255.

    image_to_predict = np.asarray(img.copy())
    prediction = model.predict(np.asarray([image_to_predict]))
    prediction = labels[int(np.argmax(prediction))]

    guessed[prediction == GUESSING] += 1

    took_ms = (time.time() - acquired_frame) * 1000

    cv.putText(frame, f'Predicted: {prediction}', (15, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv.putText(frame, f'Took time: {took_ms: .0f} ms', (15, 70), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv.putText(frame, f'Guessed: {guessed}', (15, 110), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    out.write(frame)

cap.release()
out.release()
