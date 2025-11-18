# Trabajo Final – Sistema de Reconocimiento de Patentes Mercosur (ALPR)

Este proyecto corresponde al trabajo final de la materia Procesamiento Digital de Imágenes.  
El objetivo fue desarrollar un sistema funcional para el reconocimiento automático de patentes del formato Mercosur mediante dos enfoques distintos:

1. Un modelo de redes neuronales convolucionales (CNN), entrenado con un dataset propio.
2. Un método de OCR tradicional (Tesseract), utilizado como punto de comparación.

El trabajo incluye un backend en Django, un pipeline de procesamiento de imágenes, un modelo entrenado específicamente y una interfaz web que permite probar el sistema de forma interactiva.

---

## 1. Descripción general del sistema

El sistema permite cargar una imagen desde la interfaz web y ajustar parámetros simples de preprocesamiento (brillo y contraste).  
Luego, la imagen es enviada a una API que ejecuta dos métodos de lectura independientes:

- el modelo propio entrenado,
- y el OCR clásico.

La respuesta incluye el texto detectado por cada método y un nivel de confianza. También se muestra una vista previa de la imagen procesada.

El objetivo no es obtener resultados perfectos, sino demostrar un flujo de trabajo completo combinando técnicas de procesamiento digital de imágenes y reconocimiento de caracteres.

---

## 2. Arquitectura y estructura del proyecto

La organización principal del proyecto es la siguiente:

### 2.1 Directorios y roles

- **alpr/**  
  Aplicación principal de Django. Incluye:
  - vistas,
  - templates HTML,
  - API REST,
  - integración con los modelos de OCR.

- **alpr/ml/**  
  Contiene todo lo relacionado con el procesamiento de imágenes y OCR:
  - preprocesamiento y normalización,
  - carga del dataset,
  - encoding y decodificación de caracteres,
  - arquitectura de la CNN,
  - funciones de inferencia (modelo propio + OCR clásico).

- **alpr/management/commands/train_ocr.py**  
  Script para entrenar la red neuronal desde la línea de comandos.

- **docs/**  
  Directorio con el dataset utilizado para el entrenamiento:
  - `docs/images/` → imágenes de patentes recortadas manualmente,
  - `mapping_ocr_normal.xlsx` → archivo que relaciona cada imagen con la patente real.

- **models/**  
  Se genera automáticamente después del entrenamiento. Guarda los archivos `.h5` con los pesos del modelo.

- **media/**  
  Directorio donde se guardan temporalmente las imágenes subidas desde la interfaz web.

---

## 3. Instalación del entorno

A continuación se detallan los pasos para que cualquier persona pueda ejecutar el proyecto correctamente.

### 3.1 Crear entorno virtual

```bash
python -m venv .venv
```

### 3.2 Activar entorno virtual (Windows)
```bash
.\.venv\Scripts\activate
```

### 3.3 Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3.4 Instalar Tesseract OCR

Tesseract debe instalarse como aplicación del sistema (fuera del entorno Python).
Se recomienda utilizar:

https://github.com/UB-Mannheim/tesseract/wiki

En caso de ser necesario, la ruta del ejecutable puede configurarse en:

alpr/ml/inference.py

### 3.5 Aplicar migraciones de Django
```bash
python manage.py migrate
```

Con esto el entorno queda preparado para entrenar el modelo y ejecutar la aplicación.

## 4. Entrenamiento del modelo

El dataset se encuentra en el directorio docs/ y está compuesto por aproximadamente 850 imágenes junto con un archivo Excel que vincula cada imagen con su etiqueta correspondiente.

Antes del entrenamiento:

- las imágenes se convierten a escala de grises,

- se redimensionan,

- y las etiquetas se limpian (mayúsculas, sin espacios, solo caracteres válidos).

Para entrenar el modelo:
```bash
python manage.py train_ocr
```

Durante este proceso se aplican técnicas como early stopping y validación cruzada básica.
Al finalizar, se guarda el modelo en:
```bash
models/plate_ocr_model.h5
```

## 5. Ejecución de la aplicación web

Una vez instalado el entorno y entrenado el modelo (o utilizando uno ya generado), se puede iniciar el servidor:
```bash
python manage.py runserver
```

La interfaz web está disponible en:

http://127.0.0.1:8000/alpr/upload/

Desde la interfaz se puede:

- subir una imagen de patente,

- ajustar brillo y contraste,

- procesarla mediante los dos métodos de lectura (CNN y OCR clásico),

- visualizar ambos resultados comparados,

- ver la imagen preprocesada.

## 6. API interna del sistema

El frontend se comunica con el backend a través del endpoint:

POST /alpr/api/predict/

Parámetros:

- image: archivo cargado por el usuario,

- brightness: valor numérico para ajuste,

- contrast: valor numérico para ajuste.

```bash
Respuesta:
{
  "custom_model": {
    "plate_text": "ABC123",
    "confidence": 0.87
  },
  "external_ocr": {
    "plate_text": "ABC123",
    "confidence": 0.92
  }
}
```

El endpoint también puede utilizarse para pruebas externas o integración con otros sistemas.

## 7. Consideraciones sobre el rendimiento

El modelo propio fue entrenado con un dataset de tamaño reducido, lo que limita su capacidad de generalización, especialmente frente a imágenes con variaciones de iluminación, enfoque o estilos no presentes en el dataset original.

A pesar de estas limitaciones, el modelo cumple su función dentro del marco del trabajo práctico: demostrar una implementación completa del flujo de procesamiento de imágenes y reconocimiento óptico de caracteres.

El OCR tradicional (Tesseract) ofrece un rendimiento más estable en imágenes limpias y bien contrastadas, pero también presenta dificultades en presencia de ruido o recortes imperfectos.

El valor principal del sistema radica en la integración de ambos enfoques y la posibilidad de compararlos de manera directa.

## 8. Posibles mejoras futuras

Algunas líneas de mejora identificadas son:

- Aumento del dataset y mayor diversidad de ejemplos.

- Aplicación de técnicas de data augmentation.

- Implementación de arquitecturas específicas para OCR (CRNN, LSTM, transformers).

- Preprocesamiento más robusto (binarización adaptativa, detección automática del área del texto).

- Validaciones basadas en los formatos oficiales de patentes Mercosur.

- Evaluaciones cuantitativas detalladas del desempeño de ambos métodos.

## 9. Estado actual del proyecto

El sistema está completamente operativo y permite:

- cargar imágenes desde la web,

- aplicar preprocesamiento,

- procesarlas con dos métodos diferentes,

- comparar resultados,

- y recorrer todo el pipeline implementado.

El proyecto cumple los objetivos planteados y sirve como base para posibles mejoras o extensiones futuras.