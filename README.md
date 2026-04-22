# APLICACION VIBECODEADA, DUDAR DE T

# Modelado y Simulacion con Django

MVP web para ejecutar scripts de modelado/simulacion desde una interfaz con subpaginas, entrada de funciones matematicas y visualizacion de graficos.

Las simulaciones son fijas y estan definidas en el codigo (no hay carga de nuevas simulaciones por parte del usuario).

## Incluye

- Navegacion por subpaginas (`Inicio`, `Simulaciones`, `Acerca de`)
- Registro de simulaciones extensible por `slug`
- Catalogo de simulaciones fijo (definido por el equipo)
- Orden de catalogo configurable en `config/simulation_order.json`
- Formularios dinamicos para funciones y parametros
- Entrada de formulas en estilo LaTeX sin usar delimitadores `$...$`
- Vista previa en tiempo real completamente local para las formulas escritas (MathJax offline)
- Graficos renderizados del lado servidor con `matplotlib`
- Simulaciones de ejemplo:
  - Graficar `f(x)`
  - Metodo de Euler para EDO de primer orden
  - Estimacion de PI por Monte Carlo

## Requisitos

- Python 3.12+
- Dependencias de `requirements.txt`

## Ejecucion local (Windows PowerShell)

```powershell
py -m pip install -r requirements.txt
py manage.py migrate
py manage.py runserver
```

Abrir `http://127.0.0.1:8000/`.

## Ejecutar pruebas

```powershell
py manage.py test
```

## Configurar orden de simulaciones

El orden de la pagina `Simulaciones` se lee desde `config/simulation_order.json`.
Cada entrada debe ser el `slug` de una simulacion registrada.

- Si un `slug` no existe en el registro, se ignora.
- Si falta una simulacion en el archivo, se muestra al final en orden alfabetico por titulo.

## Como agregar una nueva simulacion

1. Crear una funcion `runner` en `simulations/services.py` que reciba `cleaned_data` y devuelva un `dict` con:
   - `title`, `description`, `summary`, `plot`, `table`
2. Registrar un `SimulationSpec` en `register_default_simulations()` dentro de `simulations/services.py`.
3. Definir expresiones (`ExpressionSpec`) y parametros (`ParameterSpec`) requeridos.

Con eso la simulacion aparece automaticamente en la subpagina de listado.

## Sintaxis de funciones

Los campos de funciones aceptan una sintaxis tipo LaTeX sin necesidad de escribir `$...$`.
Ejemplos:

- `\frac{1}{2}x`
- `x^2 + 3x + 1`
- `\sqrt{x}`
- `\sin(x) + \cos(x)`

La vista previa se actualiza mientras escribes y funciona sin conexion a internet.


