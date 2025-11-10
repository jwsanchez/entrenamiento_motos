"""
ENTRENAMIENTO DEL MODELO DE CLASIFICACIÓN DE MOTOCICLETAS

Este módulo entrena un modelo de regresión para estimar el cilindraje
y deriva la categoría automáticamente de ese valor.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
import pickle

print("="*70)
print("SISTEMA DE CLASIFICACIÓN DE MOTOCICLETAS")
print("Entrenamiento de Modelo de Machine Learning")
print("="*70)

print("\n[1/5] Cargando dataset...")
df = pd.read_csv('dataset_motos_colombia.csv')
print(f"Dataset cargado: {len(df)} motocicletas")

print("\n[2/5] Preparando datos...")
features_numericas = ['Peso', 'Potencia', 'Torque', 'Capacidad_tanque',
                      'Precio', 'Velocidad_maxima']

X = df[features_numericas].values
y_cilindraje = df['Cilindraje'].values
y_categoria = df['Categoria'].values

X_train, X_test, y_train_cil, y_test_cil, y_train_cat, y_test_cat = train_test_split(
    X, y_cilindraje, y_categoria, test_size=0.2, random_state=42, stratify=y_categoria
)

print(f"Datos de entrenamiento: {len(X_train)} motos")
print(f"Datos de prueba: {len(X_test)} motos")

print("\n[3/5] Normalizando datos...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Datos normalizados")

print("\n[4/5] Entrenando modelo de regresión para estimar cilindraje...")

modelo_regresion = RandomForestRegressor(
    n_estimators=150,
    random_state=42,
    max_depth=15,
    min_samples_split=3,
    min_samples_leaf=2
)

modelo_regresion.fit(X_train_scaled, y_train_cil)

cilindraje_pred = modelo_regresion.predict(X_test_scaled)
mae = mean_absolute_error(y_test_cil, cilindraje_pred)
r2 = r2_score(y_test_cil, cilindraje_pred)

print(f"Modelo entrenado")
print(f"Error medio absoluto: {mae:.1f} cc")
print(f"R2 Score: {r2:.3f}")

# Evaluar accuracy de clasificación derivada del cilindraje
def cilindraje_a_categoria(cilindraje):
    """Convierte cilindraje en categoría según rangos definidos"""
    if cilindraje <= 200:
        return 0
    elif cilindraje <= 600:
        return 1
    else:
        return 2

categorias_pred = np.array([cilindraje_a_categoria(cc) for cc in cilindraje_pred])
accuracy_clasificacion = accuracy_score(y_test_cat, categorias_pred)

print(f"\nPrecisión de clasificación (derivada del cilindraje): {accuracy_clasificacion*100:.2f}%")

# Mostrar matriz de confusión
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test_cat, categorias_pred)
print("\nMatriz de confusión:")
print(cm)

print("\nReporte de clasificación:")
print(classification_report(y_test_cat, categorias_pred,
                          target_names=['Bajo', 'Medio', 'Alto'],
                          digits=3))

print("\n[5/5] Guardando modelo...")

with open('modelo_cilindraje.pkl', 'wb') as f:
    pickle.dump(modelo_regresion, f)

with open('scaler_motos.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('features_nombres.pkl', 'wb') as f:
    pickle.dump(features_numericas, f)

info_modelo = {
    'algoritmo': 'Random Forest Regressor',
    'mae': mae,
    'r2': r2,
    'accuracy_clasificacion': accuracy_clasificacion,
    'features': features_numericas
}

with open('info_modelo.pkl', 'wb') as f:
    pickle.dump(info_modelo, f)

print(f"Modelo guardado: modelo_cilindraje.pkl")
print(f"Scaler guardado: scaler_motos.pkl")
print(f"Información guardada: info_modelo.pkl")

print("\n" + "="*70)
print("RESUMEN DEL ENTRENAMIENTO")
print("="*70)

print(f"\nModelo de regresión:")
print(f"  Algoritmo: Random Forest Regressor")
print(f"  Error promedio: +/-{mae:.1f} cc")
print(f"  R2 Score: {r2:.3f}")

print(f"\nPrecisión de clasificación (derivada):")
print(f"  Accuracy: {accuracy_clasificacion*100:.2f}%")

print(f"\nSistema listo para hacer predicciones")
print(f"Ejecute 'clasificar_moto.py' para clasificar motos nuevas")
print("="*70)

