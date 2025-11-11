import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report
import pickle

df = pd.read_csv('dataset_motos_colombia.csv')

features_numericas = ['Peso', 'Potencia', 'Torque', 'Capacidad_tanque',
                      'Precio', 'Velocidad_maxima']

X = df[features_numericas].values
y_cilindraje = df['Cilindraje'].values
y_categoria = df['Categoria'].values

X_train, X_test, y_train_cil, y_test_cil, y_train_cat, y_test_cat = train_test_split(
    X, y_cilindraje, y_categoria, test_size=0.15, random_state=42, stratify=y_categoria
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modelo 1: Random Forest
rf_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=30,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

# Modelo 2: Gradient Boosting
gb_model = GradientBoostingRegressor(
    n_estimators=300,
    max_depth=10,
    learning_rate=0.05,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

# Entrenar ambos modelos
rf_model.fit(X_train_scaled, y_train_cil)
gb_model.fit(X_train_scaled, y_train_cil)

# Predicciones
rf_pred = rf_model.predict(X_test_scaled)
gb_pred = gb_model.predict(X_test_scaled)

# Ensemble: Promedio de ambos modelos
ensemble_pred = (rf_pred + gb_pred) / 2


# Evaluar cada modelo
def evaluar_modelo(y_true, y_pred, nombre):
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    def cilindraje_a_categoria(cilindraje):
        if cilindraje <= 200:
            return 0
        elif cilindraje <= 600:
            return 1
        else:
            return 2

    categorias_pred = np.array([cilindraje_a_categoria(cc) for cc in y_pred])
    accuracy = accuracy_score(y_test_cat, categorias_pred)

    print(f"\n{nombre}:")
    print(f"  MAE: {mae:.1f} cc")
    print(f"  R²: {r2:.3f}")
    print(f"  Accuracy: {accuracy * 100:.2f}%")

    return mae, r2, accuracy


print("\nResultados por modelo:")
mae_rf, r2_rf, acc_rf = evaluar_modelo(y_test_cil, rf_pred, "Random Forest")
mae_gb, r2_gb, acc_gb = evaluar_modelo(y_test_cil, gb_pred, "Gradient Boosting")
mae_ens, r2_ens, acc_ens = evaluar_modelo(y_test_cil, ensemble_pred, "Ensemble")

# Seleccionar el mejor modelo
modelos = {
    'Random Forest': (rf_model, mae_rf, r2_rf, acc_rf),
    'Gradient Boosting': (gb_model, mae_gb, r2_gb, acc_gb),
    'Ensemble': ('ensemble', mae_ens, r2_ens, acc_ens)
}

mejor_nombre = max(modelos.items(), key=lambda x: x[1][3])[0]
print(f"\nMejor modelo: {mejor_nombre}")

# Guardar el mejor modelo individual o ensemble
if mejor_nombre == 'Ensemble':
    # Guardar ambos modelos para hacer ensemble
    with open('modelo_cilindraje.pkl', 'wb') as f:
        pickle.dump({'rf': rf_model, 'gb': gb_model, 'tipo': 'ensemble'}, f)
    modelo_final = 'ensemble'
    mae_final = mae_ens
    r2_final = r2_ens
    acc_final = acc_ens
else:
    modelo_seleccionado = modelos[mejor_nombre][0]
    with open('modelo_cilindraje.pkl', 'wb') as f:
        pickle.dump(modelo_seleccionado, f)
    modelo_final = mejor_nombre
    mae_final = modelos[mejor_nombre][1]
    r2_final = modelos[mejor_nombre][2]
    acc_final = modelos[mejor_nombre][3]

with open('scaler_motos.pkl', 'wb') as f:
    pickle.dump(scaler, f)

info_modelo = {
    'algoritmo': modelo_final,
    'mae': mae_final,
    'r2': r2_final,
    'accuracy_clasificacion': acc_final,
    'features': features_numericas
}

with open('info_modelo.pkl', 'wb') as f:
    pickle.dump(info_modelo, f)


def cilindraje_a_categoria(cilindraje):
    if cilindraje <= 200:
        return 0
    elif cilindraje <= 600:
        return 1
    else:
        return 2


if mejor_nombre == 'Ensemble':
    categorias_pred = np.array([cilindraje_a_categoria(cc) for cc in ensemble_pred])
elif mejor_nombre == 'Random Forest':
    categorias_pred = np.array([cilindraje_a_categoria(cc) for cc in rf_pred])
else:
    categorias_pred = np.array([cilindraje_a_categoria(cc) for cc in gb_pred])

print("\n" + "=" * 50)
print(classification_report(y_test_cat, categorias_pred,
                            target_names=['Bajo', 'Medio', 'Alto'], digits=3))
