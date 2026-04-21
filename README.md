# Sistema de Medicio de Botellas en Imagenes RAW
### Licorera de Caldas —  Vision por Computador
### Topicos avanzados de visison Artitificial


---

## 1. Contexto y motivacion

Las lineas de produccion de bebidas necesitan verificar que cada botella cumpla con especificaciones geometricas precisas: altura total, diametro del cuerpo, ancho del cuello, relacion entre secciones, entre otras. Una botella fuera de tolerancia puede causar problemas de sellado, llenado incorrecto, o rechazo en el mercado.

Actualmente este control se hace de forma manual o con equipos industriales costosos. El objetivo de este proyecto es demostrar que es posible extraer medidas geometricas confiables directamente desde fotografias digitales tomadas con una camara Canon (archivos CR3).

El presente  proyecto surge como **demo tecnica** para evaluar la viabilidad del enfoque antes de escalar a un sistema de inspeccion en linea.

---

## 2. Objetivo del proyecto

Construir un sistema modular que:

- Lea imagenes en formato RAW de camara Canon (`.CR3`) .
- Detecte la botella en la imagen y extraiga su contorno.
- Mida automaticamente las dimensiones geometricas relevantes en pixeles.
- Compare dos estrategias de procesamiento (baseline vs mejorado) con metricas objetivas.
- Genere evidencia visual y numerica de los resultados.

---

## 3. compenentes importantes

```
LICORERA DE CALDAS/
│
├── config/
│   └── raw_config.json          # Parametros de lectura: formato, resize, extensiones
│
├── fotos_raw/                   # Imagenes de entrada (CR3 de Canon)
│   ├── IMG_0290.CR3
│   ├── IMG_0291.CR3
│   ├── IMG_0292.CR3
│   ├── IMG_0293.CR3
│   └── IMG_0294.CR3
│
│
├── outputs/                   
│   ├── dashboard_medidas.png    # Dashboard visual con todas las metricas
│   ├── comparacion_medidas.png  # Grafica de barras baseline vs mejorado
│   ├── calidad_borde.png        # Densidad, continuidad y tiempos por imagen
│   ├── overlay_IMG_029*.png     # Contornos superpuestos imagen por imagen
│   └── resumen_medidas.csv      # Tabla completa de todas las medidas
│
└─── src/
    ├── __init__.py
    ├── raw_loader.py            # Carga imagenes CR3 con rawpy
    ├── preprocessing.py         # Filtros de preprocesamiento
    ├── edge_detection.py        # Deteccion de bordes y extraccion de mascara
    ├── contour_measurement.py   # Medicion geometrica de la botella
    ├── evaluation.py            # Pipelines completos, metricas y graficas
    └── utils.py                 # Visualizacion y guardado de resultados


```

---

## 4. Funcionamiento

### 4.1 Lectura de imagenes RAW (CR3)

**Archivo:** `src/raw_loader.py`

Las imagenes son archivos `.CR3`, que es el formato RAW propietario de Canon (Canon RAW version 3). A diferencia de un JPEG, un archivo RAW contiene los datos crudos del sensor de la camara, con mucho mayor rango dinamico y sin compresion con perdida. Esto permite recuperar detalle en zonas oscuras o brillantes que en un JPEG ya se habrian perdido.

El problema es que los archivos CR3 no son imagenes convencionales: tienen una estructura binaria propia y requieren una biblioteca especializada para decodificarlos. Para esto se usa **rawpy**, un wrapper de Python sobre la biblioteca **LibRaw** que soporta practicamente todos los formatos RAW del mercado.

El proceso de lectura es:

```
Archivo .CR3
    ↓ rawpy.imread()
Datos del sensor (Bayer pattern, 14-bit)
    ↓ raw.postprocess(use_camera_wb=True, output_bps=8)
Imagen RGB uint8 (con balance de blancos de camara aplicado)
    ↓ cv2.resize() si max_dimension > 2000px
Imagen redimensionada a max 2000px por lado
    ↓ cv2.cvtColor(img, COLOR_RGB2GRAY)
Imagen en escala de grises (H, W) uint8
```

El redimensionamiento a 2000px maximo es una decision de rendimiento: las imagenes originales son de **6984 x 4660 pixeles** (32 megapixeles). Procesar esa resolucion completa para detectar contornos es innecesario y muy lento. A 2000px se conserva suficiente detalle geometrico para las medidas.

La funcion principal es `load_images_from_folder(folder, config)` que:
1. Busca todos los archivos con extensiones configuradas (CR3, CR2, RAW, etc.)
2. Carga cada uno e informa del resultado
3. Retorna una lista de dicts con `name`, `path` e `image` (array numpy)

La configuracion en `config/raw_config.json` controla el comportamiento:

```json
{
  "source_format": "CR3",
  "use_rawpy": true,
  "rawpy_params": {
    "use_camera_wb": true,
    "no_auto_bright": false,
    "output_bps": 8
  },
  "resize_for_processing": {
    "enabled": true,
    "max_dimension": 2000
  },
  "image_folder": "fotos_raw",
  "extensions": [".CR3", ".cr3", ".CR2", ".cr2", ".raw", ".RAW"]
}
```

---

### 4.2 Pipeline baseline

**Archivos:** `src/preprocessing.py`, `src/edge_detection.py`

El pipeline baseline representa la aproximacion clasica y directa al problema. Cada paso tiene una razon de ser:

#### Paso 1 — Suavizado Gaussiano

```python
blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
```

El sensor de la camara introduce ruido de alta frecuencia incluso en buenas condiciones de iluminacion. Canny es muy sensible a ese ruido: un pixel ruidoso produce un falso borde. El filtro Gaussiano promedia cada pixel con sus vecinos (kernel 5x5, sigma=1.0), eliminando variaciones pequenas pero preservando los bordes reales de la botella, que son transiciones bruscas de intensidad mucho mas anchas que el ruido puntual.

#### Paso 2 — Deteccion de bordes con Canny

```python
median = np.median(preprocessed)
low  = int(max(0,   (1.0 - 0.33) * median))
high = int(min(255, (1.0 + 0.33) * median))
edges = cv2.Canny(preprocessed, low, high)
```

Canny es el detector de bordes mas usado en vision por computador porque:
- Detecta bordes reales con alta probabilidad (baja tasa de falsos positivos).
- Localiza el borde en su posicion exacta (no lo engrosa).
- Usa dos umbrales (histeresis): el umbral alto acepta bordes fuertes y el bajo extiende bordes conectados a ellos, eliminando bordes debiles aislados.

Los umbrales se calculan automaticamente como fraccion de la mediana de la imagen. Esto hace el sistema adaptable a diferentes niveles de exposicion sin ajuste manual: si la imagen es mas oscura (mediana baja), los umbrales bajan proporcionalmente.

#### Paso 3 — Cierre morfologico

```python
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
```

El cierre morfologico es una dilatacion seguida de una erosion. Su efecto practico es **cerrar gaps pequenos en lineas de borde**. La botella de vidrio tiene zonas transparentes y reflejos que interrumpen el borde continuo. Sin este paso, el contorno de la botella aparece fragmentado y no se puede rellenar correctamente.

#### Paso 4 — Seleccion del contorno y relleno de mascara

```python
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
best = _best_bottle_contour(contours, img_area)
cv2.drawContours(mask, [best], -1, 255, thickness=cv2.FILLED)
```

De todos los contornos encontrados se selecciona el mas "botella-like" con la funcion `_best_bottle_contour`:
- Se descartan contornos menores al 0.3% del area de la imagen (ruido).
- Se separan contornos verticales (alto >= ancho) de horizontales.
- Entre los verticales, se elige el de mayor altura absoluta.
- Si no hay verticales, se usa el de mayor area como fallback.

Esta heuristica funciona porque las botellas son objetos verticales. Un contorno de fondo o sombra raramente es mas alto que ancho.

---

### 4.3 Pipeline mejorado

**Archivos:** `src/preprocessing.py`, `src/edge_detection.py`

El pipeline mejorado parte de las mismas tecnicas pero aplica mejoras especificamente disenadas para fotografias de botellas de vidrio, que presentan dos desafios principales:

1. **Reflejos especulares (hot-spots):** el vidrio refleja la luz directamente hacia la camara, creando zonas saturadas de blanco que "queman" el detalle del borde.
2. **Transparencia:** el vidrio deja ver el fondo a traves de el, lo que hace que el detector de bordes encuentre tanto el borde de la botella como el fondo visible tras ella.

#### Paso 1 — CLAHE (Contrast Limited Adaptive Histogram Equalization)

```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
enhanced = clahe.apply(gray)
```

El histograma de una imagen de botella tiende a estar comprimido: el cuerpo de la botella tiene tonos similares entre si, y el fondo tambien. La diferencia entre ambos puede ser sutil. La ecualizacion de histograma global empeoraria esto al amplificar el ruido globalmente.

CLAHE divide la imagen en tiles de 16x16 pixeles y ecualiza el histograma de cada tile independientemente, con un limite de amplificacion (clipLimit=2.0) para no magnificar el ruido. El resultado es que las transiciones locales de intensidad, como el borde de la botella, se vuelven mas visibles incluso en zonas de iluminacion no uniforme.

#### Paso 2 — Supresion de reflejos

```python
p99 = np.percentile(enhanced, 99)
enhanced = np.clip(enhanced, 0, int(p99))
enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
```

Los pixeles en el percentil 99 y superior son casi con certeza reflejos especulares, no informacion util del borde. Al saturarlos al valor del percentil 99 y renormalizar, se elimina el efecto de "quemado" sin afectar el resto de la imagen.

#### Paso 3 — Suavizado Gaussiano adaptado

```python
smoothed = cv2.GaussianBlur(enhanced, (7, 7), 2.0)
```

Kernel mas grande (7x7) y sigma mayor (2.0) que el baseline. Despues de CLAHE, el ruido de alta frecuencia es mas visible porque el contraste local aumento. Un suavizado mas fuerte lo suprime antes de que Canny lo confunda con bordes reales.

#### Paso 4 — Canny con gradiente L2

```python
edges = cv2.Canny(preprocessed, low, high, L2gradient=True)
```

La diferencia con el baseline es `L2gradient=True`. Por defecto Canny usa la norma L1 para calcular la magnitud del gradiente (|Gx| + |Gy|), que es una aproximacion rapida. Con L2 usa la formula exacta sqrt(Gx² + Gy²), que es mas precisa geometricamente, especialmente en bordes diagonales como los de la botella en angulo.

#### Paso 5 — Cierre morfologico y refinamiento de mascara

```python
# En improved_mask():
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
```

Sobre la mascara (no sobre el mapa de bordes) se aplica:
- **Cierre 7x7 x2 iteraciones:** para rellenar huecos internos causados por la transparencia del vidrio. Un kernel mas grande que el baseline porque los huecos en el vidrio son mas grandes que las discontinuidades del borde.
- **Apertura 3x3 x1 iteracion:** para eliminar pequenas protuberancias externas que no pertenecen a la botella.

#### Paso 6 — Validacion con keypoints (SIFT / ORB)

```python
detector = cv2.SIFT_create(nfeatures=500)  # o ORB si SIFT no disponible
keypoints = detector.detect(gray, None)
```

Se detectan puntos de interes en la imagen usando SIFT (Scale-Invariant Feature Transform) o ORB como alternativa. Estos detectores encuentran esquinas y regiones con estructura local rica. En una botella, los puntos mas ricos en textura suelen estar en la etiqueta, la tapa y el relieve del vidrio, todos dentro del contorno de la botella.

En la implementacion actual esta validacion es **diagnostica**: no modifica la mascara sino que sirve para verificar que los keypoints encontrados estén mayoritariamente dentro de la region detectada. Si en una version futura se quiere refinar la mascara activamente, la base esta puesta.

---

### 4.4 Medicion geometrica

**Archivo:** `src/contour_measurement.py`

Una vez obtenida la mascara binaria de la botella, se extraen las siguientes medidas:

| Medida | Como se calcula | Que representa |
|--------|----------------|----------------|
| `height_px` | Alto del bounding box | Altura total de la botella |
| `width_max_px` | Ancho del bounding box | Diametro maximo |
| `width_25_px` | Ancho de la mascara en la fila al 25% de la altura | Diametro en zona alta |
| `width_50_px` | Ancho de la mascara en la fila al 50% de la altura | Diametro en la cintura |
| `width_75_px` | Ancho de la mascara en la fila al 75% de la altura | Diametro en zona baja |
| `width_neck_px` | Promedio de anchos en el 0-10% superior | Ancho del cuello |
| `width_base_px` | Promedio de anchos en el 85-100% inferior | Ancho de la base |
| `ratio_w_h` | width_max / height | Esbeltez de la botella |
| `ratio_neck_base` | width_neck / width_base | Relacion cuello-base (tipifica el modelo) |
| `area_px` | cv2.contourArea() | Area total en pixeles cuadrados |
| `perimeter_px` | cv2.arcLength() | Perimetro del contorno |

Los anchos relativos se calculan escaneando la fila correspondiente de la mascara binaria y tomando la distancia entre el primer y ultimo pixel activo. Esto da el ancho real del cuerpo de la botella en esa altura, incluso si el contorno no es perfectamente simetrico.

La relacion cuello/base es especialmente util para clasificar modelos: una botella tipo vino tiene cuello muy estrecho relative a la base (ratio < 0.5), mientras que una botella de aguardiente puede tener una proporcion diferente segun el diseno.

**Nota sobre unidades:** todas las medidas estan en pixeles. Para convertirlas a milimetros se necesita una referencia metrica en la escena (ver seccion de siguientes pasos).

---

### 4.5 Evaluacion comparativa

**Archivo:** `src/evaluation.py`

Ademas de las medidas geometricas, el sistema calcula metricas de calidad del proceso:

#### Metricas de calidad de borde (sin referencia)

Cuando no se tiene una mascara de referencia ("ground truth"), se usan metricas proxy:

- **Densidad de borde:** proporcion de pixeles que son borde sobre el total de la imagen. Un valor muy bajo indica que Canny no encontro suficientes bordes; muy alto indica ruido excesivo.

- **Continuidad de borde:** fraccion de pixeles de borde que tienen al menos un vecino que tambien es borde (en ventana 3x3). Un borde real de botella es continuo; el ruido produce bordes aislados con baja continuidad.

#### IoU entre mascaras (Intersection over Union)

```
IoU = Area(mascara_baseline ∩ mascara_mejorado) / Area(mascara_baseline ∪ mascara_mejorado)
```

Mide cuanto se solapan las mascaras de ambos pipelines. Un IoU cercano a 1.0 significa que ambos detectaron practicamente la misma region. Un IoU bajo indica que los pipelines discrepan sobre donde esta la botella.

#### Tiempo de procesamiento

Se mide el tiempo de ejecucion de cada pipeline completo (preprocesado + bordes + mascara + medicion) usando `time.time()`. Esto permite cuantificar el costo computacional del pipeline mejorado frente al baseline.

---

## 5. Resultados obtenidos

Las 5 imagenes CR3 procesadas (redimensionadas a ~1999x1334 px desde los 6984x4660 originales):

| Imagen | Altura base | Altura mejorado | Ancho base | Ancho mejorado | IoU |
|--------|------------|----------------|-----------|---------------|-----|
| IMG_0290 | 370 px | 209 px | 285 px | 264 px | 0.76 |
| IMG_0291 | 1120 px | 245 px | 364 px | 264 px | 0.55 |
| IMG_0292 | 755 px | 630 px | 71 px | 48 px | 0.83 |
| IMG_0293 | 1030 px | 175 px | 297 px | 288 px | 0.56 |
| IMG_0294 | 1140 px | 728 px | 258 px | 60 px | 0.00 |

| Metrica | Baseline | Mejorado |
|---------|----------|----------|
| Tiempo promedio | ~0.013 s | ~0.32 s |
| Continuidad de borde | 1.00 | 1.00 |
| Densidad de borde | 0.041 | 0.027 |

**Interpretacion:** El baseline es ~25x mas rapido. Ambos logran continuidad de borde perfecta (1.0). La densidad de borde del mejorado es menor, lo que indica que es mas selectivo: detecta menos bordes pero mas significativos (menos ruido). La variacion entre imagenes en las medidas geometricas refleja que las fotografias fueron tomadas en condiciones y encuadres distintos.

**Outputs generados:**

- `outputs/dashboard_medidas.png` — Panel visual completo con 12 graficas comparativas
- `outputs/comparacion_medidas.png` — Barras de medidas clave
- `outputs/calidad_borde.png` — Densidad, continuidad y tiempos
- `outputs/overlay_IMG_029*.png` — Contorno baseline (azul) vs mejorado (naranja) sobre la imagen original, por cada imagen
- `outputs/resumen_medidas.csv` — Tabla completa exportable
- `outputs/resumen_medidas.html` — Tabla en HTML para visualizacion web

---

## 6. Decisiones tecnicas y por que se tomaron

### Por que rawpy y no PIL o cv2.imread directamente?

`PIL` y `cv2.imread` no soportan archivos CR3 de Canon. Solo pueden leer formatos de imagen estandar (JPEG, PNG, TIFF, BMP). Los archivos RAW de camara tienen una estructura binaria propia que requiere LibRaw para decodificar el patron Bayer del sensor y aplicar el pipeline de demosaicing. `rawpy` es el binding de Python mas completo y activo para LibRaw.

### Por que escala de grises y no color?

La deteccion de bordes clasica (Canny) opera sobre imagenes de un canal. El color no agrega informacion util para encontrar el contorno geometrico de una botella de vidrio transparente: el borde existe en todos los canales por igual. Procesar en gris reduce el uso de memoria en un factor 3 y acelera todos los pasos posteriores. Si en el futuro se necesita distinguir por color (ej. tapa roja vs azul), se puede agregar clasificacion por color como paso adicional sobre la region ya detectada.

### Por que redimensionar a 2000px y no a otro valor?

Es un balance entre precision y velocidad. A 2000px:
- Los bordes de la botella tienen suficientes pixeles para que Canny los detecte con precision.
- El kernel morfologico de 5x5 representa una distancia fisica razonable.
- El tiempo de procesamiento por imagen es menor a 0.5 segundos en hardware convencional.

A 4000px el tiempo se cuadruplicaria sin mejora significativa en las medidas relativas.

### Por que CLAHE con tileGridSize=(16,16) y clipLimit=2.0?

Con tiles grandes (ej. 8x8 o menor numero de tiles) el efecto de CLAHE es mas global y se acerca al histograma global, perdiendo el beneficio local. Con tiles muy pequenos (32x32 o mas) amplifica demasiado el ruido de cada tile. 16x16 tiles para una imagen de ~1300px es un buen equilibrio.

`clipLimit=2.0` es conservador. Valores altos (4-8) producen mayor contraste pero tambien amplifican mas el ruido. Para botellas de vidrio donde el ruido compite con los bordes reales, el valor bajo es mas seguro.

### Por que no usar redes neuronales (deep learning)?

Para una demo de viabilidad tecnica, las redes neuronales requieren:
1. Dataset etiquetado de botellas (no disponible en este proyecto).
2. Entrenamiento con GPU (infraestructura no establecida).
3. Tiempo de desarrollo mayor.

Las tecnicas clasicas permiten tener resultados en horas, son completamente interpretables (cada paso tiene un efecto claro) y son suficientes para demostrar el concepto. Una vez validado el enfoque, se puede iterar hacia deep learning (ver siguientes pasos).

### Por que la funcion `_best_bottle_contour` prefiere contornos verticales?

Las botellas son objetos verticales por naturaleza. Un contorno de fondo, sombra o artefacto rara vez es mas alto que ancho. Al priorizar contornos donde altura >= ancho, se reduce drasticamente la probabilidad de seleccionar el contorno equivocado en imagenes con fondos complejos, sin necesidad de parametros adicionales.

---

## 7. Limitaciones actuales

1. **Sin calibracion metrica.** Todas las medidas son en pixeles. Para obtener milimetros se necesita colocar una referencia de longitud conocida en la escena (ej. una regla) y calcular el factor pixels/mm.

2. **Sensibilidad al fondo.** Si el fondo tiene objetos con forma similar a una botella (ej. columnas, marcos), el sistema puede confundirse. Un fondo neutro y uniforme mejora considerablemente los resultados.

3. **Sin calibracion de camara.** La distorsion de lente del Canon no esta compensada. En un sistema de produccion habria que calibrar la camara con un patron de damero (cv2.calibrateCamera) para corregir la distorsion antes de medir.

4. **Posicion de la botella no fija.** El sistema busca la botella en toda la imagen. Si la botella siempre estuviera en la misma posicion (conveyor belt), se podria definir una region de interes fija que reduce errores y tiempo de computo.

5. **Sin clasificacion por modelo.** El sistema mide pero no clasifica. No determina si la botella es del modelo correcto o si esta dentro de tolerancia.

---



## 8. Siguientes pasos recomendados

### Corto plazo (mejoras al sistema actual)

1. **Calibracion metrica:** agregar una referencia fisica conocida en cada fotografia (ej. una moneda o una regla) y calcular automaticamente el factor de conversion px → mm. Esto convierte las medidas de relativas a absolutas.

2. **Region de interes fija:** si las botellas siempre se fotografian en la misma posicion, definir un rectangulo de busqueda en `raw_config.json` para acelerar el proceso y reducir falsos positivos.

3. **Tolerancias por modelo:** agregar un archivo `config/modelos_botella.json` con las especificaciones nominales y tolerancias de cada modelo. El sistema podria marcar automaticamente botellas fuera de especificacion.

4. **Deteccion multi-botella:** adaptar el sistema para detectar y medir varias botellas por imagen, util para fotografia de lotes.

## Autores

- **Edward Fabian Goyeneche Velandia**  
  - egoyeneche@unal.edu.co

- **Samuel Esteban Reyes Nazate**  
  - sreyesn@unal.edu.co  


