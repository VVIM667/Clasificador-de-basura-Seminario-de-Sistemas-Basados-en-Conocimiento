import os
import cv2
import numpy as np

def load_and_preprocess_image(file_path, target_size=(224, 224)):
    image = cv2.imread(file_path)
    if image is None:
        return None
    image = cv2.resize(image, target_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0  # Convertir a float32 antes de normalizar
    return image

# Directorio del dataset
data_dir = 'TrashType_Image_Dataset'
categories = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Ruta de la carpeta para las imágenes redimensionadas
redimensionadas_dir = 'imgEntrenamiento'

# Crear la carpeta si no existe
if not os.path.exists(redimensionadas_dir):
    os.makedirs(redimensionadas_dir)

# Cargar y preprocesar las imágenes
for category in categories:
    category_path = os.path.join(data_dir, category)
    class_num = categories.index(category)
    category_redimensionadas_dir = os.path.join(redimensionadas_dir, category)
    # Crear la subcarpeta para la categoría si no existe
    if not os.path.exists(category_redimensionadas_dir):
        os.makedirs(category_redimensionadas_dir)
    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)
        image = load_and_preprocess_image(img_path)
        if image is not None:
            # Guardar la imagen redimensionada en la subcarpeta correspondiente
            file_name = f'{category}_{img_name}'
            file_path = os.path.join(category_redimensionadas_dir, file_name)
            cv2.imwrite(file_path, cv2.cvtColor(image * 255, cv2.COLOR_RGB2BGR))
            print(f"Imagen guardada: {file_path}")
