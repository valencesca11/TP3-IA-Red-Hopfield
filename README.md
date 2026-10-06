# TP3 - Inteligencia Artificial: Red de Hopfield

Este repositorio contiene el prototipo desarrollado para el Trabajo Práctico N.º 3 de la materia Inteligencia Artificial.

## Descripción

El programa implementa una red de Hopfield destinada a recuperar imágenes de 10 × 10 píxeles a partir de entradas con ruido.

Las imágenes utilizan la siguiente representación:

- `#`: píxel activo.
- `.`: píxel vacío.

Internamente, los píxeles se convierten a valores bipolares:

- `#` → `+1`
- `.` → `-1`

El prototipo utiliza dos patrones de referencia:

- Una figura con forma de “C”.
- Una escuadra como elemento de referencia.

## Métodos implementados

El programa compara dos métodos de entrenamiento:

- Regla de Hebb.
- Matriz pseudoinversa de Moore-Penrose.

También permite agregar ruido a las imágenes, recuperar los patrones y calcular la función de energía de la red.

## Resultados

Se realizaron pruebas con diferentes niveles de ruido:

- Figura “C” con 15 % de ruido.
- Figura “C” con 25 % de ruido.
- Escuadra con 20 % de ruido.

En las pruebas realizadas, los patrones fueron recuperados correctamente en dos iteraciones.

## Requisitos

- Python 3.
- Biblioteca NumPy.

Para instalar NumPy:

```bash
pip install numpy
