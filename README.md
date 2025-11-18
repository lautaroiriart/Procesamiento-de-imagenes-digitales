ALPR Mercosur – Proyecto Final

Este proyecto es el trabajo final de la materia Procesamiento Digital de Imágenes. La idea general fue armar un sistema completo que pueda reconocer patentes del formato Mercosur a partir de imágenes, usando dos enfoques distintos:

Un modelo de redes neuronales convolucionales entrenado por mí, utilizando un dataset propio.

Un OCR tradicional (Tesseract) como método de comparación.

Además, se desarrolló una interfaz web en Django que permite subir una imagen, ajustar algunos parámetros de preprocesamiento y ver los resultados de ambos métodos lado a lado.

Funcionalidad general

El flujo es el siguiente:

El usuario sube una imagen de una patente desde la web.

Se puede modificar brillo y contraste para mejorar la lectura.

La imagen se envía a una API que procesa todo.

El backend ejecuta:

el modelo CNN entrenado,

el OCR clásico,

y devuelve los dos resultados (texto detectado + nivel de confianza).

En pantalla se muestran ambos resultados junto con una vista previa de la imagen.

Estructura del proyecto

A grandes rasgos:

alpr/
Contiene la app principal de Django. Ahí está todo lo relacionado a vistas, templates, API y el código del modelo.

alpr/ml/
Módulo donde está el preprocesamiento, carga del dataset, encoding de caracteres, arquitectura de la red y la lógica de inferencia.

alpr/management/commands/train_ocr.py
Script para entrenar el modelo desde la línea de comandos.

docs/
Contiene el dataset usado para entrenar:

docs/images/ → imágenes reales de patentes

mapping_ocr_normal.xlsx → archivo con el nombre de cada imagen y la patente correspondiente

models/
Se genera automáticamente cuando se entrena el modelo. Guarda el archivo .h5 con los pesos entrenados.

media/
Directorio donde se guarda temporalmente la imagen que sube el usuario antes de procesarla.

Instalación del entorno

Crear el entorno virtual:

python -m venv .venv


Activarlo:

.\.venv\Scripts\activate


Instalar dependencias:

pip install -r requirements.txt


Instalar Tesseract OCR (fuera de Python).
Descarga recomendada:
https://github.com/UB-Mannheim/tesseract/wiki

Una vez instalado, asegurarse de configurar correctamente la ruta en inference.py si es necesario.

Aplicar migraciones:

python manage.py migrate

Entrenamiento del modelo

El dataset consiste en aproximadamente 850 imágenes de patentes recortadas manualmente y un archivo Excel que relaciona cada archivo con su patente real. Antes del entrenamiento, el sistema limpia las etiquetas (quita espacios, pasa a mayúsculas y deja solo caracteres válidos).

Para entrenar el modelo:

python manage.py train_ocr


Esto:

carga el dataset,

arma las etiquetas codificadas,

entrena la red con early stopping,

y guarda el modelo final en models/plate_ocr_model.h5.

Ejecución

Para iniciar el servidor:

python manage.py runserver


La aplicación web se encuentra en:

http://127.0.0.1:8000/alpr/upload/


Desde ahí se puede subir una imagen y probar ambos métodos de lectura.

API utilizada internamente

El frontend se comunica con:

POST /alpr/api/predict/


- En el body se envía:

  la imagen,

  el brillo elegido,

  el contraste elegido.

- La respuesta es un JSON con dos secciones:

  resultado del modelo propio,

  resultado del OCR clásico.

Cada uno devuelve el texto detectado y un nivel de confianza.

Sobre el rendimiento

El modelo propio no está entrenado con un dataset muy grande, por lo que su precisión es limitada, especialmente en imágenes nuevas o de estilos distintos. Aun así, sirve para demostrar la arquitectura completa (dataset → preprocesamiento → entrenamiento → inferencia → API → web).

Tesseract, por el contrario, funciona mejor en imágenes limpias y con buena iluminación, pero no necesariamente maneja bien casos más complicados.

El objetivo del proyecto no es obtener un sistema de lectura perfecto, sino mostrar e integrar ambas aproximaciones, entender sus diferencias y montar una solución funcional de punta a punta.

Posibles mejoras

Algunas cosas que podrían incorporarse a futuro:

Aumentar el dataset o aplicar técnicas de data augmentation.

Implementar modelos más avanzados (CRNN, LSTM o transformers para secuencias).

Mejorar el recorte del área negra de la patente antes de procesarla.

Aplicar reglas específicas de formato (por ejemplo LLLNNN, LLNNLL, etc.).

Comparar más métricas entre ambos métodos.

Estado del proyecto

La aplicación funciona de punta a punta:

sube imágenes,

ajusta parámetros,

procesa con ambos métodos,

muestra resultados comparativos,

y permite demostrar los distintos enfoques para OCR.