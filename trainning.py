import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os
import yaml
import matplotlib.pyplot as plt

if __name__ == "__main__":
  img_height = 300
  img_width = 200
  batch_size = 32
  epochs = 50
  data_dir = 'dataset/dataset_keggle'
  class_mapping_file_path = 'model/class_mapping_keggle.yaml'

  train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
  )

  train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
  )

  validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
  )

  class_indices = train_generator.class_indices
  with open(class_mapping_file_path, 'w') as f:
    yaml.dump(class_indices, f)

  base_model = tf.keras.applications.MobileNetV2(
      input_shape=(img_height, img_width, 3),
      include_top=False,
      weights='imagenet'
  )
  base_model.trainable = False

  model = models.Sequential([
      base_model,
      layers.GlobalAveragePooling2D(),
      layers.Dense(128, activation='relu'),
      layers.Dropout(0.5),
      layers.Dense(train_generator.num_classes, activation='softmax')
  ])


  model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

  history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
  )

  model.save('model/cnn_model_keggle.h5')

  plt.plot(history.history['accuracy'], label='Training Accuracy')
  plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')
  plt.show()
