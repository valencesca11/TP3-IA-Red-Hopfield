import numpy as np

class HopfieldTP3:
    """Prototipo de una red de Hopfield para imágenes de 10 x 10 píxeles."""

    FILAS = 10
    COLUMNAS = 10

    def __init__(self, patrones):
        self.patrones = [self.imagen_a_vector(patron) for patron in patrones]

    @staticmethod
    def imagen_a_vector(imagen):
        """Convierte una imagen con '#' y '.' a un vector bipolar."""
        if len(imagen) != HopfieldTP3.FILAS:
            raise ValueError("La imagen debe tener 10 filas.")
        if any(len(fila) != HopfieldTP3.COLUMNAS for fila in imagen):
            raise ValueError("Cada fila debe tener 10 columnas.")

        return np.array(
            [1 if pixel == "#" else -1 for fila in imagen for pixel in fila],
            dtype=float,
        )

    @staticmethod
    def vector_a_imagen(vector):
        """Convierte un vector bipolar en una imagen de caracteres."""
        matriz = vector.reshape(HopfieldTP3.FILAS, HopfieldTP3.COLUMNAS)
        return ["".join("#" if valor == 1 else "." for valor in fila) for fila in matriz]

    @staticmethod
    def mostrar_imagen(titulo, vector):
        print(f"\n{titulo}")
        if isinstance(vector, list):
            vector = HopfieldTP3.imagen_a_vector(vector)
        for fila in HopfieldTP3.vector_a_imagen(vector):
            print(fila)

    def entrenar_hebb(self):
        """Calcula W mediante la regla de Hebb y elimina las autoconexiones."""
        cantidad_neuronas = self.patrones[0].size
        pesos = np.zeros((cantidad_neuronas, cantidad_neuronas))

        for patron in self.patrones:
            pesos += np.outer(patron, patron)

        # Una neurona no debe conectarse consigo misma.
        np.fill_diagonal(pesos, 0)
        return pesos

    def entrenar_pseudoinversa(self):
        """Calcula W utilizando la pseudoinversa de Moore-Penrose."""
        matriz_patrones = np.column_stack(self.patrones)
        pesos = matriz_patrones @ np.linalg.pinv(matriz_patrones)

        # Se simetriza la matriz y se eliminan las autoconexiones.
        pesos = (pesos + pesos.T) / 2
        np.fill_diagonal(pesos, 0)
        return pesos

    @staticmethod
    def agregar_ruido(vector, porcentaje, semilla=10):
        """Invierte una cantidad controlada de píxeles del patrón."""
        resultado = vector.copy()
        cantidad = int(round(len(resultado) * porcentaje))
        generador = np.random.default_rng(semilla)
        posiciones = generador.choice(len(resultado), size=cantidad, replace=False)
        resultado[posiciones] *= -1
        return resultado

    @staticmethod
    def energia(vector, pesos):
        return float(-0.5 * vector.T @ pesos @ vector)

    @staticmethod
    def recuperar(entrada, pesos, max_iteraciones=20):
        """Recupera un patrón mediante actualización asincrónica."""
        estado = entrada.copy()
        energias = [HopfieldTP3.energia(estado, pesos)]

        for iteracion in range(1, max_iteraciones + 1):
            cambios = 0

            # La actualización asincrónica favorece la convergencia del modelo.
            for neurona in range(len(estado)):
                activacion = float(pesos[neurona] @ estado)

                if activacion > 0:
                    nuevo_valor = 1
                elif activacion < 0:
                    nuevo_valor = -1
                else:
                    nuevo_valor = estado[neurona]

                if nuevo_valor != estado[neurona]:
                    estado[neurona] = nuevo_valor
                    cambios += 1

            energias.append(HopfieldTP3.energia(estado, pesos))

            if cambios == 0:
                return estado, iteracion, energias

        return estado, max_iteraciones, energias

    def identificar_patron(self, resultado):
        """Devuelve el patrón almacenado con menor distancia de Hamming."""
        distancias = [int(np.sum(resultado != patron)) for patron in self.patrones]
        indice = int(np.argmin(distancias))
        return indice + 1, distancias

    def ejecutar_prueba(self, nombre, patron, pesos, porcentaje_ruido, semilla):
        if isinstance(patron, list):
            patron = self.imagen_a_vector(patron)
        entrada = self.agregar_ruido(patron, porcentaje_ruido, semilla)
        resultado, iteraciones, energias = self.recuperar(entrada, pesos)
        patron_identificado, distancias = self.identificar_patron(resultado)

        self.mostrar_imagen(f"{nombre} - Entrada con ruido", entrada)
        self.mostrar_imagen(f"{nombre} - Patrón recuperado", resultado)

        print(f"Patrón identificado: {patron_identificado}")
        print(f"Distancias a los patrones almacenados: {distancias}")
        print(f"Iteraciones: {iteraciones}")
        print(f"Energía: {energias[0]:.2f} -> {energias[-1]:.2f}")
        print(f"Energía no creciente: {all(energias[i + 1] <= energias[i] + 1e-9 for i in range(len(energias) - 1))}")

        return {
            "nombre": nombre,
            "patron_identificado": patron_identificado,
            "distancias": distancias,
            "iteraciones": iteraciones,
            "energia_inicial": energias[0],
            "energia_final": energias[-1],
            "energias": energias,
        }


def crear_patron_c():
    return [
        ".#######..",
        "##........",
        "##........",
        "##........",
        "##........",
        "##........",
        "##........",
        ".#######..",
        "..........",
        "..........",
    ]


def crear_patron_escuadra():
    return [
        "##........",
        "##........",
        "##........",
        "##........",
        "##........",
        "##........",
        "##........",
        "########..",
        "..........",
        "..........",
    ]


def ejecutar_modelo(nombre_metodo, pesos, red, patron_c, patron_escuadra):
    print("\n" + "=" * 60)
    print(f"PRUEBAS CON {nombre_metodo.upper()}")
    print("=" * 60)

    red.mostrar_imagen("Patrón 1 almacenado: figura C", patron_c)
    red.mostrar_imagen("Patrón 2 almacenado: escuadra", patron_escuadra)

    resultados = []
    resultados.append(red.ejecutar_prueba("Prueba 1 - C con 15% de ruido", patron_c, pesos, 0.15, 10))
    resultados.append(red.ejecutar_prueba("Prueba 2 - C con 25% de ruido", patron_c, pesos, 0.25, 20))
    resultados.append(red.ejecutar_prueba("Prueba 3 - Escuadra con 20% de ruido", patron_escuadra, pesos, 0.20, 30))
    return resultados


def main():
    patron_c = crear_patron_c()
    patron_escuadra = crear_patron_escuadra()
    red = HopfieldTP3([patron_c, patron_escuadra])

    pesos_hebb = red.entrenar_hebb()
    pesos_pseudoinversa = red.entrenar_pseudoinversa()

    ejecutar_modelo("Hebb", pesos_hebb, red, patron_c, patron_escuadra)
    ejecutar_modelo("Pseudoinversa", pesos_pseudoinversa, red, patron_c, patron_escuadra)


if __name__ == "__main__":
    main()
