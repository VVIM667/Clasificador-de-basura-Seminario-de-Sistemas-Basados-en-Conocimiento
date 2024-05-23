# proyectoSSBC
Materiales:
Para la implementacion de este modelo se necesito varios conjuntos de datasets de imagenes de residuos
los cuales fueron puestos en subcarpetas en la carpeta de TrashType_Image_Dataset
Entre los cuales incluí los siguientes datasets:

Trashnet
The dataset spans six classes: glass, paper, cardboard, plastic, metal, and trash. Currently, the dataset consists of 2527 images:

501 glass
594 paper
403 cardboard
482 plastic
410 metal
137 trash

Waste Classification Data v2
A variation about the Waste Classification data: extended by the new category "N" - Nonrecyclable added.

Over 25k images already divided into training data - 22564 + 2508 (N) images and test data - 2513 images + new 397 from category nonrecyclable. Three main categories: Organic (O) and recyclable (R), and nonrecyclable (N). TRAIN folder contains 2508 images in the "N" directory. The TEST folder contains 397 images in the "N" directory.

Download: Directly from kaggle https://www.kaggle.com/sapal6/waste-classification-data-v2

Lenguaje de programacion:
Python

Con esto hice una mezcla de ambos datasets y aproximadamente entrene el modelo con mas de 500 imagenes por categoria.

1. roceso de redimenzionar las imagenes

Se pasa la ubicacion de la carpeta que contiene las imagenes originales, con sus categorias 
correspondientes, despues se redimencionan 224x224 para evitar discrepacias entre las imagenes que tomara el modelo como aprendizaje
y finalmente se guardan en una carpeta llamada imgEntrenamiento

2. Proceso de entrenamiento:
Se importan las librerias necesarias para trabajar con este proyecto como lo es tensorflow
se pasa como variable la carpeta donde estan las carpetas, se establece el numero de categorias y el tamaño que tienen las imagenes
(aqui agarra el codigo de entrenamiento.py y metelo a chatgpt para que lo explique mejor que yo)
se ejecuta y se empieza a entrenar por epocas (10)
y se guarda el modelo (.h5) ya entrenado logrando una precisicion del 58% (depende de cada ejecucion mejorar la precision o empeorarla)


3. Graficas
se describe y se muestra en este archivo el comportamiento del entrenamiento y la perdida del mismo, utilizando el historial_entrenamiento.pk1 generado en el archivo anterior

4. Por ultimo se prueba el rendimiento y la precision del mismo logrando asi obtener las imagenes que estan en la carpeta "Capturas de pantalla", anexo a esto las imagenes de las graficas del comportamiento:



