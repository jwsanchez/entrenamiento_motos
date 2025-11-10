"""
SISTEMA DE CLASIFICACIÓN DE MOTOCICLETAS

Este módulo permite clasificar motocicletas según sus características técnicas,
estimando el cilindraje exacto en CC y derivando la categoría correspondiente.
"""

import pickle
import numpy as np

try:
    with open('modelo_cilindraje.pkl', 'rb') as f:
        modelo_cilindraje = pickle.load(f)

    with open('scaler_motos.pkl', 'rb') as f:
        scaler = pickle.load(f)

    with open('info_modelo.pkl', 'rb') as f:
        info = pickle.load(f)

    print("=" * 70)
    print("SISTEMA DE CLASIFICACIÓN DE MOTOCICLETAS")
    print("=" * 70)
    print(f"\nModelo cargado: {info['algoritmo']}")
    print(f"Error promedio: +/-{info['mae']:.1f} cc")
    print(f"Precisión de clasificación: {info['accuracy_clasificacion'] * 100:.2f}%")

except FileNotFoundError:
    print("ERROR: No se encontraron los archivos del modelo.")
    print("Por favor, ejecute primero 'entrenar_modelo.py'")
    exit()


def obtener_numero(mensaje, min_val=0, max_val=None):
    """
    Solicita y valida entrada numérica del usuario.

    Args:
        mensaje (str): Mensaje a mostrar
        min_val (float): Valor mínimo permitido
        max_val (float): Valor máximo permitido

    Returns:
        float: Valor validado ingresado por el usuario
    """
    while True:
        try:
            valor = float(input(mensaje))
            if valor < min_val:
                print(f"  Error: El valor debe ser mayor o igual a {min_val}")
                continue
            if max_val is not None and valor > max_val:
                print(f"  Error: El valor debe ser menor o igual a {max_val}")
                continue
            return valor
        except ValueError:
            print("  Error: Por favor ingrese un número válido")


def cilindraje_a_categoria(cilindraje):
    """
    Convierte un valor de cilindraje en su categoría correspondiente.

    Args:
        cilindraje (float): Cilindraje en CC

    Returns:
        int: Categoría (0=Bajo, 1=Medio, 2=Alto)
    """
    if cilindraje <= 200:
        return 0
    elif cilindraje <= 600:
        return 1
    else:
        return 2


def clasificar_moto():
    """
    Solicita características de una motocicleta y realiza la clasificación.
    """
    print("\n" + "=" * 70)
    print("INGRESE LAS CARACTERÍSTICAS DE LA MOTOCICLETA")
    print("=" * 70)
    print("\nRangos de referencia:")
    print("  Peso: 100-270 kg")
    print("  Potencia: 7-200 HP")
    print("  Torque: 8-150 Nm")
    print("  Tanque: 9-25 litros")
    print("  Precio: 4-110 millones COP")
    print("  Velocidad máxima: 85-300 km/h")
    print("")

    peso = obtener_numero("1. Peso (kg): ", min_val=50, max_val=400)
    potencia = obtener_numero("2. Potencia (HP): ", min_val=5, max_val=300)
    torque = obtener_numero("3. Torque (Nm): ", min_val=5, max_val=200)
    capacidad_tanque = obtener_numero("4. Capacidad del tanque (litros): ", min_val=5, max_val=30)
    precio = obtener_numero("5. Precio (millones COP): ", min_val=2, max_val=200)
    velocidad_maxima = obtener_numero("6. Velocidad máxima (km/h): ", min_val=70, max_val=350)

    caracteristicas = np.array([[peso, potencia, torque, capacidad_tanque,
                                 precio, velocidad_maxima]])

    caracteristicas_normalizadas = scaler.transform(caracteristicas)

    cilindraje_estimado = modelo_cilindraje.predict(caracteristicas_normalizadas)[0]
    prediccion = cilindraje_a_categoria(cilindraje_estimado)

    categorias = {
        0: 'BAJO CILINDRAJE (90-200cc)',
        1: 'MEDIO CILINDRAJE (201-600cc)',
        2: 'ALTO CILINDRAJE (601cc+)'
    }

    print("\n" + "=" * 70)
    print("RESULTADO DE LA CLASIFICACIÓN")
    print("=" * 70)

    print("\nCaracterísticas ingresadas:")
    print(f"  Peso: {peso} kg")
    print(f"  Potencia: {potencia} HP")
    print(f"  Torque: {torque} Nm")
    print(f"  Capacidad tanque: {capacidad_tanque} litros")
    print(f"  Precio: ${precio} millones COP")
    print(f"  Velocidad máxima: {velocidad_maxima} km/h")

    print("\n" + "-" * 70)
    print(f"CILINDRAJE ESTIMADO: {cilindraje_estimado:.0f} cc")
    print(f"CATEGORÍA PREDICHA: {categorias[prediccion]}")
    print("-" * 70)

    # Calcular nivel de confianza basado en qué tan lejos está de los límites
    if prediccion == 0:
        distancia_al_limite = 200 - cilindraje_estimado
        confianza = min(100, 50 + (distancia_al_limite / 200) * 50)
    elif prediccion == 1:
        distancia_inferior = cilindraje_estimado - 200
        distancia_superior = 600 - cilindraje_estimado
        distancia_minima = min(distancia_inferior, distancia_superior)
        confianza = min(100, 50 + (distancia_minima / 400) * 50)
    else:
        distancia_al_limite = cilindraje_estimado - 600
        confianza = min(100, 50 + (distancia_al_limite / 600) * 50)

    print(f"\nNivel de confianza: {confianza:.1f}%")
    print(f"Margen de error: +/-{info['mae']:.0f} cc")

    print("\n" + "=" * 70)
    print("INFORMACIÓN ADICIONAL")
    print("=" * 70)

    if prediccion == 0:
        print(f"""
Esta motocicleta se clasifica como BAJO CILINDRAJE.
Cilindraje estimado: {cilindraje_estimado:.0f} cc

Características típicas:
  Cilindraje: 90-200cc
  Uso ideal: Ciudad, domicilios, trabajo
  Ventajas: Económica, bajo consumo, fácil manejo
  Ejemplos: Honda CB125F, Yamaha Crypton, Bajaj Boxer
        """)
    elif prediccion == 1:
        print(f"""
Esta motocicleta se clasifica como MEDIO CILINDRAJE.
Cilindraje estimado: {cilindraje_estimado:.0f} cc

Características típicas:
  Cilindraje: 201-600cc
  Uso ideal: Ciudad y carretera, viajes medios
  Ventajas: Balance entre potencia y consumo
  Ejemplos: Yamaha R3, KTM Duke 390, Kawasaki Ninja 400
        """)
    else:
        print(f"""
Esta motocicleta se clasifica como ALTO CILINDRAJE.
Cilindraje estimado: {cilindraje_estimado:.0f} cc

Características típicas:
  Cilindraje: 601cc en adelante
  Uso ideal: Carretera, touring, alta velocidad
  Ventajas: Máxima potencia y prestaciones
  Ejemplos: BMW R1250GS, Kawasaki Ninja 650, Ducati Monster
        """)

    print("=" * 70)


def menu_principal():
    """
    Menú principal del sistema de clasificación.
    """
    while True:
        print("\n" + "=" * 70)
        print("MENÚ PRINCIPAL")
        print("=" * 70)
        print("\n1. Clasificar una nueva motocicleta")
        print("2. Ver información del modelo")
        print("3. Salir")

        opcion = input("\nSeleccione una opción (1-3): ").strip()

        if opcion == '1':
            clasificar_moto()
        elif opcion == '2':
            mostrar_info_modelo()
        elif opcion == '3':
            print("\nGracias por usar el sistema de clasificación")
            print("=" * 70)
            break
        else:
            print("\nOpción no válida. Por favor seleccione 1, 2 o 3.")


def mostrar_info_modelo():
    """
    Muestra información detallada sobre el modelo entrenado.
    """
    print("\n" + "=" * 70)
    print("INFORMACIÓN DEL MODELO")
    print("=" * 70)
    print(f"\nAlgoritmo: {info['algoritmo']}")
    print(f"Error medio absoluto: +/-{info['mae']:.1f} cc")
    print(f"R2 Score: {info['r2']:.3f}")
    print(f"Precisión de clasificación: {info['accuracy_clasificacion'] * 100:.2f}%")

    print(f"\nEnfoque de clasificación:")
    print(f"  1. El modelo estima el cilindraje en CC")
    print(f"  2. La categoría se deriva automáticamente del cilindraje:")
    print(f"       - Bajo: 90-200cc")
    print(f"       - Medio: 201-600cc")
    print(f"       - Alto: 601cc+")
    print(f"  3. Esto garantiza coherencia total entre cilindraje y categoría")

    print(f"\nCaracterísticas utilizadas:")
    for i, feature in enumerate(info['features'], 1):
        print(f"  {i}. {feature}")

    print(f"\nDataset de entrenamiento: 90 motocicletas colombianas")
    print("=" * 70)


if __name__ == "__main__":
    menu_principal()