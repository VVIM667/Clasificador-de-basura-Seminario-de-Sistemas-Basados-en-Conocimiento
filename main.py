import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Cargar el modelo entrenado
model = load_model('modelo_entrenado.h5')
categorias = ['carton', 'vidrio', 'metal', 'papel', 'plastico', 'basura']

# Umbral de confianza para la predicción (ajústalo según tus necesidades)
confidence_threshold = 0.5

# Función para preprocesar la imagen antes de hacer predicciones
def preprocess_image(image):
    resized_image = cv2.resize(image, (224, 224))
    processed_image = resized_image.astype(np.float32) / 255.0
    return processed_image

# Función para hacer predicciones en una imagen con umbral de confianza
def predict_image(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(np.expand_dims(processed_image, axis=0), verbose=0)
    predicted_class_index = np.argmax(prediction)
    confidence = prediction[0][predicted_class_index]
    class_label = categorias[predicted_class_index]
    return class_label, confidence

# Función para capturar imágenes de la cámara y hacer predicciones en tiempo real con umbral de confianza
def predict_camera():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret:
            result, confidence = predict_image(frame)
            # Aplicar umbral de confianza
            if confidence > confidence_threshold:
                cv2.putText(frame, f"{result} ({confidence:.2f})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Llamar a la función para iniciar las predicciones en tiempo real desde la cámara con umbral de confianza
predict_camera()
