import os
from os import mkdir, chdir, getcwd
from warnings import warn

import onnx
# import onnxmltools
import tensorflow as tf
from tensorflow.keras.models import Sequential
from typing import Optional, Any
from matplotlib import pyplot
from datetime import datetime


# https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/
# Plotting learning curves / loss
def __learning_curve(hist):
    pyplot.title('Loss')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Cross Entropy')
    pyplot.plot(hist.history['loss'], label='train')
    pyplot.plot(hist.history['val_loss'], label='val')
    pyplot.legend()
    pyplot.savefig('loss.png')


# Plotting accuracy
def __model_accuracy(hist):
    pyplot.title("Model Accuracy")
    pyplot.xlabel("Epoch")
    pyplot.ylabel("Accuracy")
    pyplot.plot(hist.history["accuracy"], label='train')
    pyplot.plot(hist.history["val_accuracy"], label='val')
    pyplot.legend()
    pyplot.savefig('acc.png')


# def __convert_keras_to_onnx(k_model: (Sequential, 'Модель tf.keras для конвертации'),
#                             onnx_name: (str, 'Требуется имя для модели ONNX')):
#     onnx_model = onnxmltools.convert_keras(k_model, target_opset=7)
#     onnx.checker.check_model(onnx_model)
#     onnxmltools.save_model(onnx_model, f'{onnx_name}.onnx')


def save_results(main_name: str,
                 model: Any,
                 history: Optional[Any],
                 loss_acc: list):
    """
    Цель функции - сохранить данные созданной модели для дальнейшего анализа её эффективносити,
    например, по отношению к другим моделям.

    :param main_name: имя папки, любое, необходимо для создания папки для вывода.
    :param model: натренерованная модель. Сохраняется для повторного использования.
    :param history: история, для визуализации графиков acc/loss.
    :param loss_acc: финальные данные acc/loss. Сохраняются в отдельный файл.
    :return: None
    """

    try:
        dir_name = main_name + f'_{datetime.now().strftime("%m-%d-%Y_%H-%M-%S")}'

        mkdir(dir_name)
        chdir(getcwd() + f'\\{dir_name}')

        # Save model
        model.save(f'{main_name}_model.h5')

        # Save tf model
        tf.saved_model.save(model, os.getcwd() + '\\tf_model')

        # Draw/save plots
        __learning_curve(history)
        pyplot.clf()
        __model_accuracy(history)
        pyplot.clf()

        # Save loss/acc results
        with open(main_name + '_loss_acc_results.txt', 'w', encoding='utf8') as f:
            f.write(f'accuracy: {loss_acc[1]: .4f}\nloss: {loss_acc[0]: .2f}\n')

        # Generate onnx
        # os.system(f'python -m tf2onnx.convert --saved-model tf_model --output {main_name}_onnx_model.onnx --opset 6')

    except FileExistsError:
        warn("Creation of the directory failed!")
