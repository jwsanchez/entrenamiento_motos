# Sistema de Clasificación de Motocicletas mediante Aprendizaje de Maquina

Sistema de predicción de cilindraje y categorización de motocicletas basado en características técnicas.
---

## Descripción

Este proyecto clasifica motocicletas colombianas en tres categorías según su cilindraje (Bajo: 90-200cc, Medio: 201-600cc, Alto: 601cc+) utilizando técnicas de aprendizaje supervisado. A partir de seis características medibles (peso, potencia, torque, capacidad del tanque, precio y velocidad máxima), el sistema estima el cilindraje con precisión y deriva automáticamente su categoría.

---

## Resumen

Los sistemas expertos tradicionales requieren que expertos humanos codifiquen reglas manualmente para cada caso. Este proyecto demuestra cómo el aprendizaje automático puede descubrir patrones complejos directamente de los datos, eliminando la necesidad de reglas explícitas y adaptándose fácilmente a nuevas información.

**Comparación de enfoques:**

Reglas manuales (tradicional):
```python
if peso < 150 and potencia < 15:
    return "Bajo cilindraje"
```

Aprendizaje automático (implementado):
```python
modelo.fit(datos_entrenamiento)
prediccion = modelo.predict(moto_nueva)
```

El segundo enfoque captura relaciones no lineales entre variables y se actualiza simplemente re-entrenando con nuevos datos.

---

## Arquitectura

El sistema utiliza un ensemble de dos algoritmos:

**Random Forest (500 árboles)**: Combina múltiples árboles de decisión para predicciones robustas. Cada árbol aprende diferentes aspectos de los datos y el promedio reduce el error.

**Gradient Boosting (300 iteraciones)**: Construye árboles secuencialmente donde cada uno corrige los errores del anterior. Especialmente efectivo para capturar patrones sutiles.

**Predicción final**: Promedio de ambos modelos, aprovechando las fortalezas de cada uno.

El flujo de trabajo es directo:
1. Normalización de características
2. Predicción de cilindraje por ambos modelos
3. Promedio de predicciones
4. Derivación de categoría según rangos definidos

---

## Dataset

Motocicletas del mercado colombiano, balanceadas en 100 ejemplos por categoría. Incluye desde motos de trabajo básicas (Hero Splendor, Honda Wave) hasta modelos premium (BMW R1250GS, Ducati Panigale). Las marcas representadas son Honda, Yamaha, Suzuki, Kawasaki, KTM, BMW, Bajaj, TVS, Ducati, Triumph, Harley-Davidson, Indian, Royal Enfield, entre otras.

**Variables utilizadas:**
- Peso (kg): 95-423 kg
- Potencia (HP): 7.5-231 HP
- Torque (Nm): 7.8-221 Nm
- Capacidad tanque (L): 9-27 L
- Precio (millones COP): 3.8-155
- Velocidad máxima (km/h): 85-299 km/h

Las variables categóricas (marca, modelo) se excluyen intencionalmente para que el modelo generalice mejor a motos nuevas de cualquier fabricante.

---

## Comportamiento

La matriz de confusión muestra que la mayoría de errores ocurren cerca de los límites de categoría (200cc y 600cc), lo cual es esperado dado que una moto de 195cc y otra de 205cc son técnicamente similares aunque estén en categorías diferentes.

El modelo identifica potencia y velocidad máxima como los predictores más importantes, seguidos por precio y torque.

---

## Instalación

**Requisitos:**
- Python 3.8+

**Dependencias:**
```bash
pip install numpy pandas scikit-learn
```

O instalar desde el archivo de requisitos:
```bash
pip install -r requirements.txt
```

**Contenido de requirements.txt:**
```
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
```

---

## Uso

**Entrenar el modelo:**
```bash
python entrenar.py
```

Esto genera tres archivos .pkl con los modelos entrenados y el transformador de normalización. El proceso toma 0-1 minuto dependiendo del hardware.

**Clasificar motocicletas:**
```bash
python clasificar_moto.py
```

El sistema solicita las seis características y devuelve el cilindraje estimado, categoría y nivel de confianza.

**Ejemplo de sesión:**
```
Peso (kg): 185
Potencia (HP): 74
Torque (Nm): 68
Tanque (litros): 14
Precio (millones COP): 39
Velocidad máxima (km/h): 195

Cilindraje estimado: 689 cc
Categoría: MEDIO CILINDRAJE (201-600cc)
Confianza: 87.3%
Margen de error: +/-16 cc
```

---

## Limitaciones

**Limitaciones actuales:**
- Optimizado para el mercado colombiano (precios en COP)
- Categorías fijas de cilindraje (no detecta automáticamente rangos óptimos)
- No distingue entre tipos de uso (deportiva, touring, trabajo)

---

## Referencias

Documentación de scikit-learn: https://scikit-learn.org/stable/

Especificaciones técnicas de fabricantes: Honda Colombia, Yamaha Colombia, Suzuki, Kawasaki, KTM, BMW Motorrad, entre otros.

---
