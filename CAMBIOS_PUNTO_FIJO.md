# Cambios en el Método de Punto Fijo

## Resumen
Se mejoró la simulación del **método de punto fijo** con:
- ✓ **Parámetros de intervalo** para visualización (a, b)
- ✓ **Gráfico de fase** mostrando g(x), y=x, y las iteraciones como pasos (escalera)
- ✓ **Gráfico de convergencia** mostrando la evolución de x_n por iteración

## Cambios Realizados

### 1. Nueva función auxiliar: `_aux_plot_fixed_point()`
**Ubicación:** `simulations/services.py` líneas 383-459

Genera un gráfico que muestra:
- **Curva g(x)**: Función de iteración (color gris claro)
- **Línea y=x**: Línea de referencia en punteado (color gris oscuro)
- **Pasos de iteración**: Visualización de las iteraciones como una "escalera" 
  - Líneas verticales: suben desde (x_n, x_n) hasta (x_n, g(x_n))
  - Líneas horizontales: avanzan desde (x_n, g(x_n)) hasta (g(x_n), g(x_n))
- **Punto inicial x0**: Marcado en verde

```python
def _aux_plot_fixed_point(g, a: float, b: float, iterations: list, title: str) -> str:
```

### 2. Función mejorada: `_run_fixed_point()`
**Cambios:**
- Agregó parámetros `a` y `b` del diccionario de datos (líneas 469-471)
- Con valores por defecto: `a = x0 - 1.0`, `b = x0 + 1.0`
- Generación de gráfico de fase con `_aux_plot_fixed_point()` (línea 497)
- Generación de gráfico de convergencia (líneas 489-494)
- Retorno de **2 gráficos auxiliares** en `auxiliary_plots` (líneas 514-516)
- Inclusión del intervalo en el resumen (línea 506)

### 3. Actualización del registro: `register_default_simulations()`
**Cambios en SimulationSpec para "punto-fijo":**

```python
parameters=(
    ParameterSpec("x0", "Valor inicial x0", "float", 0.5),
    ParameterSpec("a", "Extremo inferior (visualizacion)", "float", -0.5),     # ← NUEVO
    ParameterSpec("b", "Extremo superior (visualizacion)", "float", 1.5),      # ← NUEVO
    ParameterSpec("tol", "Tolerancia", "float", 1e-6, 1e-12, 1.0),
    ParameterSpec("max_iter", "Max iteraciones", "int", 100, 1, 10000),
)
```

## Resultado en la Interfaz

### Formulario
El usuario ahora ve 5 campos en lugar de 3:
1. **Función g(x)** (expresión)
2. **Valor inicial x0** (parámetro)
3. **Extremo inferior (visualización) a** (parámetro nuevo)
4. **Extremo superior (visualización) b** (parámetro nuevo)
5. **Tolerancia** (parámetro)
6. **Max iteraciones** (parámetro)

### Página de Resultados
Se muestran **3 gráficos**:
1. **Gráfico principal**: Convergencia (iteración vs x_n)
2. **Figuras auxiliares:**
   - "Fase del punto fijo": g(x) con y=x y pasos de iteración
   - "Convergencia": Iteración vs x_n

### Resumen
Se incluye el campo `intervalo: [a, b]` en el resumen de resultados.

## Visualización del Gráfico de Fase

El gráfico muestra exactamente el tipo de diagrama que enviaste:
```
       y
       ↑
       |     y=x (línea punteada)
       |    /
       |   /  g(x) (curva)
       |  /
       |/___→ x
```

Con las iteraciones dibujadas como escaleras:
- Vertical: sube hasta la curva g(x)
- Horizontal: cruza a la línea y=x
- Se repite hasta convergencia (máximo 20 iteraciones visualizadas)

## Pruebas

### Test Ejecutado
```
Entrada: g(x) = cos(x), x0 = 0.5, a = -0.5, b = 1.5, tol = 1e-6, max_iter = 100

Resultado:
✓ Simulación ejecutada correctamente
✓ 34 iteraciones hasta convergencia
✓ Aproximación: 0.739085
✓ Error final: 1e-06
✓ 2 gráficos auxiliares generados
✓ Tabla con primeras 25 iteraciones
```

## Compatibilidad
- ✓ Template ya soporta `auxiliary_plots`
- ✓ No requiere cambios en template HTML
- ✓ Compatible con el resto de simulaciones
- ✓ Django check: `System check identified no issues`
