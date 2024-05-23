import pickle
import matplotlib.pyplot as plt

# Cargar el historial del entrenamiento
with open('historial_entrenamiento.pkl', 'rb') as f:
    history = pickle.load(f)

# Visualizar las gráficas del entrenamiento
# Precisión
plt.plot(history['accuracy'])
plt.title('Precisión del modelo durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend(['Entrenamiento'], loc='upper left')
plt.show()

# Pérdida
plt.plot(history['loss'])
plt.title('Pérdida del modelo durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend(['Entrenamiento'], loc='upper left')
plt.show()
