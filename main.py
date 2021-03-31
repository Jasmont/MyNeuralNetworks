import tensorflow as tf
from util import save_results

TRAIN_DIR = r'data\train'
VALIDATION_DIR = r'data\validation'
TEST_DIR = r'data\test'
TARGET_SIZE = (150, 150)
BATCH_SIZE = 128

train_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255,
                                                                       rotation_range=270,
                                                                       width_shift_range=[-25, 25],
                                                                       height_shift_range=[-25, 25],
                                                                       brightness_range=[0.3, 1.2],
                                                                       fill_mode='nearest')

validation_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
test_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

train_generator = train_data_generator.flow_from_directory(directory=TRAIN_DIR,
                                                           target_size=TARGET_SIZE,
                                                           batch_size=BATCH_SIZE,
                                                           color_mode='rgb',
                                                           shuffle=True,
                                                           class_mode='categorical',
                                                           seed=42,
                                                           # save_to_dir=r'data\augs', save_prefix='aug', save_format='png',
                                                           )

# for inputs, outputs in train_generator:
# pass

validation_generator = validation_data_generator.flow_from_directory(directory=VALIDATION_DIR,
                                                                     target_size=TARGET_SIZE,
                                                                     batch_size=BATCH_SIZE,
                                                                     color_mode='rgb',
                                                                     shuffle=True,
                                                                     class_mode='categorical',
                                                                     seed=42)

test_generator = test_data_generator.flow_from_directory(directory=TEST_DIR,
                                                         target_size=TARGET_SIZE,
                                                         batch_size=BATCH_SIZE,
                                                         color_mode='rgb',
                                                         shuffle=True,
                                                         class_mode='categorical',
                                                         seed=42)


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.99 and logs.get('val_accuracy') > 0.99:
            print("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training = True


# base_model = tf.keras.applications.MobileNetV2(include_top=False,
#                                                input_shape=train_generator.image_shape)

# base_model = tf.keras.applications.VGG19(include_top=False,
#                                          input_shape=train_generator.image_shape)

base_model = tf.keras.applications.ResNet152V2(include_top=False,
                                               input_shape=train_generator.image_shape)

x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation=tf.keras.activations.relu)(x)
x = tf.keras.layers.Dense(1024, activation=tf.keras.activations.relu)(x)
x = tf.keras.layers.Dense(512, activation=tf.keras.activations.relu)(x)
# x = tf.keras.layers.Dense(512, activation=tf.keras.activations.relu)(x)
# x = tf.keras.layers.Dense(256, activation=tf.keras.activations.relu)(x)
# x = tf.keras.layers.Dense(256, activation=tf.keras.activations.relu)(x)
output = tf.keras.layers.Dense(train_generator.num_classes, activation=tf.keras.activations.softmax)(x)

model = tf.keras.Model(inputs=base_model.input, outputs=output)

for layer in model.layers:
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, epsilon=1e-4),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.n // train_generator.batch_size,
    epochs=250,
    validation_data=validation_generator,
    validation_steps=validation_generator.n // validation_generator.batch_size,
    callbacks=[myCallback()]
)

test_generator.reset()
pred = model.evaluate_generator(test_generator,
                                steps=test_generator.n // test_generator.batch_size,
                                verbose=1)

save_results('inceptionresnetv2', history=history, model=model, loss_acc=pred)
