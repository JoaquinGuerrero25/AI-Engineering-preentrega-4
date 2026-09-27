---
id: functools
title: "functools: funciones de orden superior y caché"
category: programacion-funcional
tags: [functools, lru_cache, cache, partial, wraps, decoradores, memoizacion]
---

# functools: funciones de orden superior y caché

El módulo `functools` reúne herramientas para trabajar con funciones que reciben o devuelven otras funciones: decoradores de caché, aplicación parcial de argumentos, reducción de secuencias y despacho según el tipo del argumento.

## lru_cache: memoización con límite de tamaño

`@functools.lru_cache(maxsize=128, typed=False)` guarda los resultados de una función según sus argumentos. Si la función se vuelve a llamar con los mismos argumentos, devuelve el valor guardado sin recalcularlo. Es ideal para funciones **puras** y costosas.

```python
from functools import lru_cache

@lru_cache(maxsize=256)
def fibonacci(n: int) -> int:
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

fibonacci(80)
print(fibonacci.cache_info())
# CacheInfo(hits=78, misses=81, maxsize=256, currsize=81)
```

Detalles del parámetro **`maxsize`**:

- Indica cuántos resultados distintos se guardan como máximo. Cuando la caché se llena, se descarta la entrada **usada hace más tiempo** (política LRU, *Least Recently Used*).
- Con `maxsize=None` la caché crece sin límite y la política LRU se desactiva, lo que la hace un poco más rápida.
- Conviene que sea una potencia de dos por eficiencia, aunque no es obligatorio.

El parámetro `typed=True` hace que argumentos de tipos distintos se guarden por separado: `f(3)` y `f(3.0)` serían dos entradas diferentes.

### Estadísticas y limpieza

- `funcion.cache_info()` devuelve una tupla con nombre `CacheInfo(hits, misses, maxsize, currsize)`: aciertos, fallos, tamaño máximo y cantidad actual de entradas. Sirve para medir si la caché realmente se está aprovechando.
- `funcion.cache_clear()` vacía la caché y reinicia las estadísticas.

### Restricciones

- Todos los argumentos deben ser **hashables**: una lista o un diccionario como argumento produce `TypeError: unhashable type`. Se pueden convertir a tuplas o `frozenset` antes de llamar.
- No debe usarse con funciones que tienen efectos secundarios o que dependen de datos que cambian (la hora, un archivo, una base de datos): devolvería resultados viejos.
- Aplicado a métodos, la caché mantiene una referencia a `self` y puede impedir que los objetos se liberen de memoria.

## cache (Python 3.9+)

`@functools.cache` es un atajo equivalente a `lru_cache(maxsize=None)`: una caché ilimitada, más simple y liviana. Útil cuando la cantidad de argumentos distintos es acotada.

## cached_property

`@functools.cached_property` convierte un método en un atributo que se calcula la primera vez que se accede y luego se guarda en la instancia. A diferencia de `lru_cache`, la caché vive en cada objeto y se libera con él. Para recalcularlo basta con borrar el atributo (`del objeto.atributo`).

## partial: fijar argumentos

`functools.partial(func, *args, **kwargs)` crea una nueva función con algunos argumentos ya fijados.

```python
from functools import partial

def potencia(base, exponente):
    return base ** exponente

cuadrado = partial(potencia, exponente=2)
print(cuadrado(7))      # 49
```

Es muy práctico para pasar callbacks que necesitan argumentos extra, por ejemplo a `executor.map` o a `sorted(key=...)`.

## wraps: decoradores que conservan metadatos

Al escribir un decorador, la función envoltorio reemplaza a la original y se pierden su nombre, su docstring y su firma. `@functools.wraps(func)` copia esos metadatos (`__name__`, `__doc__`, `__module__`, `__wrapped__`) a la función envoltorio.

```python
from functools import wraps
import time

def cronometrar(func):
    @wraps(func)
    def envoltorio(*args, **kwargs):
        inicio = time.perf_counter()
        resultado = func(*args, **kwargs)
        print(f"{func.__name__} tardó {time.perf_counter() - inicio:.3f} s")
        return resultado
    return envoltorio
```

## reduce

`functools.reduce(funcion, iterable, inicial)` aplica acumulativamente una función de dos argumentos: `reduce(lambda a, b: a * b, [1, 2, 3, 4])` devuelve 24. Para sumas, máximos o uniones suele ser más legible usar `sum`, `max` o un bucle explícito.

## singledispatch: sobrecarga por tipo

`@functools.singledispatch` convierte una función en genérica: se registran implementaciones alternativas según el tipo del **primer** argumento con `@funcion.register`. Para métodos existe `singledispatchmethod`.

```python
from functools import singledispatch

@singledispatch
def describir(valor):
    return f"objeto: {valor!r}"

@describir.register
def _(valor: int):
    return f"entero: {valor}"

@describir.register
def _(valor: list):
    return f"lista de {len(valor)} elementos"
```

## total_ordering

`@functools.total_ordering` completa los métodos de comparación de una clase: basta con definir `__eq__` y uno de `__lt__`, `__le__`, `__gt__` o `__ge__` para obtener los demás.
