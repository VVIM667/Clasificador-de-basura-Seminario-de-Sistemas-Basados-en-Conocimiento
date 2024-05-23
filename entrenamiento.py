import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import pickle

# Ruta del directorio que contiene las imágenes redimensionadas
redimensionadas_dir = 'imgEntrenamiento'

# Obtener la lista de categorías (nombres de las subcarpetas)
categorias = os.listdir(redimensionadas_dir)

# Número de categorías
num_categorias = len(categorias)

# Tamaño de las imágenes de entrada
tamaño_imagenes = (224, 224)

# Crear un generador de datos para cargar las imágenes
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Cargar las imágenes usando el generador de datos
datos_entrenamiento = datagen.flow_from_directory(
    redimensionadas_dir,
    target_size=tamaño_imagenes,
    batch_size=32,  # Tamaño del lote
    class_mode='categorical'  # Modo de clasificación
)

# Cargar el modelo base preentrenado (MobileNetV2)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Congelar las capas del modelo base
for layer in base_model.layers:
    layer.trainable = False

# Agregar capas adicionales al modelo
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_categorias, activation='softmax')(x)  # Capa de salida con activación softmax para clasificación múltiple

# Construir el modelo final
model = Model(inputs=base_model.input, outputs=predictions)

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo y guardar el historial del entrenamiento
history = model.fit(datos_entrenamiento, epochs=10)  # Especificar el número de épocas de entrenamiento

# Guardar el modelo entrenado
model.save('modelo_entrenado.h5')

# Guardar el historial del entrenamiento
with open('historial_entrenamiento.pkl', 'wb') as f:
    pickle.dump(history.history, f)
