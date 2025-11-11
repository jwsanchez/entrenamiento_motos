import pickle
import numpy as np

try:
    with open('modelo_cilindraje.pkl', 'rb') as f:
        modelo_data = pickle.load(f)

    with open('scaler_motos.pkl', 'rb') as f:
        scaler = pickle.load(f)

    with open('info_modelo.pkl', 'rb') as f:
        info = pickle.load(f)

    # Verificar si es ensemble o modelo único
    if isinstance(modelo_data, dict) and modelo_data.get('tipo') == 'ensemble':
        es_ensemble = True
        modelo_rf = modelo_data['rf']
        modelo_gb = modelo_data['gb']
    else:
        es_ensemble = False
        modelo = modelo_data

    print("="*70)
    print("SISTEMA DE CLASIFICACIÓN DE MOTOCICLETAS")
    print("="*70)
    print(f"\nModelo: {info['algoritmo']}")
    print(f"Error promedio: +/-{info['mae']:.0f} cc")
    print(f"Precisión: {info['accuracy_clasificacion']*100:.2f}%")

except FileNotFoundError:
    print("ERROR: Ejecute primero 'entrenar_modelo.py'")
    exit()

def obtener_numero(mensaje, min_val=0, max_val=None):
    while True:
        try:
            valor = float(input(mensaje))
            if valor < min_val:
                print(f"  Error: Mínimo {min_val}")
                continue
            if max_val is not None and valor > max_val:
                print(f"  Error: Máximo {max_val}")
                continue
            return valor
        except ValueError:
            print("  Error: Número inválido")

def cilindraje_a_categoria(cilindraje):
    if cilindraje <= 200:
        return 0
    elif cilindraje <= 600:
        return 1
    else:
        return 2

def calcular_confianza(cilindraje):
    if cilindraje <= 200:
        distancia_limite = min(200 - cilindraje, cilindraje - 90)
        confianza = min(100, 50 + (distancia_limite / 110) * 50)
    elif cilindraje <= 600:
        distancia_limite = min(cilindraje - 200, 600 - cilindraje)
        confianza = min(100, 50 + (distancia_limite / 400) * 50)
    else:
        distancia_limite = cilindraje - 600
        confianza = min(100, 50 + (min(distancia_limite, 700) / 700) * 50)
    return confianza

def clasificar_moto():
    print("\n" + "="*70)
    print("INGRESE CARACTERÍSTICAS")
    print("="*70)

    peso = obtener_numero("Peso (kg): ", 50, 450)
    potencia = obtener_numero("Potencia (HP): ", 5, 300)
    torque = obtener_numero("Torque (Nm): ", 5, 250)
    capacidad_tanque = obtener_numero("Tanque (litros): ", 5, 30)
    precio = obtener_numero("Precio (millones COP): ", 2, 200)
    velocidad_maxima = obtener_numero("Velocidad máxima (km/h): ", 70, 350)

    caracteristicas = np.array([[peso, potencia, torque, capacidad_tanque,
                                 precio, velocidad_maxima]])

    caracteristicas_normalizadas = scaler.transform(caracteristicas)

    if es_ensemble:
        pred_rf = modelo_rf.predict(caracteristicas_normalizadas)[0]
        pred_gb = modelo_gb.predict(caracteristicas_normalizadas)[0]
        cilindraje_estimado = (pred_rf + pred_gb) / 2
    else:
        cilindraje_estimado = modelo.predict(caracteristicas_normalizadas)[0]

    prediccion = cilindraje_a_categoria(cilindraje_estimado)
    confianza = calcular_confianza(cilindraje_estimado)

    categorias = {
        0: 'BAJO CILINDRAJE (90-200cc)',
        1: 'MEDIO CILINDRAJE (201-600cc)',
        2: 'ALTO CILINDRAJE (601cc+)'
    }

    print("\n" + "="*70)
    print("RESULTADO")
    print("="*70)
    print(f"\nCilindraje estimado: {cilindraje_estimado:.0f} cc")
    print(f"Categoría: {categorias[prediccion]}")
    print(f"Confianza: {confianza:.1f}%")
    print(f"Margen de error: +/-{info['mae']:.0f} cc")

    if confianza < 70:
        print(f"\nNota: La moto está cerca del límite entre categorías")

    print("="*70)

def menu_principal():
    while True:
        print("\n" + "="*70)
        print("MENÚ")
        print("="*70)
        print("\n1. Clasificar motocicleta")
        print("2. Ver información del modelo")
        print("3. Salir")

        opcion = input("\nOpción (1-3): ").strip()

        if opcion == '1':
            clasificar_moto()
        elif opcion == '2':
            mostrar_info_modelo()
        elif opcion == '3':
            print("\nGracias por usar el sistema")
            print("="*70)
            break
        else:
            print("\nOpción inválida")

def mostrar_info_modelo():
    print("\n" + "="*70)
    print("INFORMACIÓN DEL MODELO")
    print("="*70)
    print(f"\nAlgoritmo: {info['algoritmo']}")
    print(f"Error medio: +/-{info['mae']:.0f} cc")
    print(f"R² Score: {info['r2']:.3f}")
    print(f"Precisión: {info['accuracy_clasificacion']*100:.2f}%")
    print(f"\nCaracterísticas:")
    for i, feature in enumerate(info['features'], 1):
        print(f"  {i}. {feature}")
    print(f"\nDataset: 300 motocicletas colombianas")
    print("="*70)

if __name__ == "__main__":
    menu_principal()