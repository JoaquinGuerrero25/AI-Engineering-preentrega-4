---
id: concurrent-futures
title: "threading y concurrent.futures: hilos y procesos"
category: concurrencia
tags: [threading, concurrent.futures, ThreadPoolExecutor, ProcessPoolExecutor, gil, multiprocessing]
---

# threading y concurrent.futures: hilos y procesos

Python ofrece dos modelos clásicos para ejecutar trabajo en paralelo además de `asyncio`: **hilos** (módulo `threading`) y **procesos** (módulo `multiprocessing`). El módulo `concurrent.futures` los unifica bajo una interfaz de alto nivel basada en *executors* y objetos `Future`, y es la opción recomendada para la mayoría de los casos.

## El GIL y la elección entre hilos y procesos

En CPython, el **GIL** (Global Interpreter Lock) impide que dos hilos ejecuten bytecode de Python exactamente al mismo tiempo. Por eso:

- Los **hilos** son útiles para tareas **I/O-bound** (descargas, lectura de archivos, consultas a una base de datos): mientras un hilo espera la red, el GIL se libera y otro hilo avanza.
- Para tareas **CPU-bound** (procesar imágenes, cálculos numéricos puros en Python, compresión) los hilos no aceleran el programa; hay que usar **procesos**, cada uno con su propio intérprete y su propio GIL.

Desde Python 3.13 existe una compilación opcional "free-threaded" que desactiva el GIL, pero la distribución estándar de CPython todavía lo incluye y muchas extensiones en C aún no son compatibles con ese modo.

## threading: hilos de bajo nivel

```python
import threading

def tarea(n):
    print(f"procesando {n}")

hilos = [threading.Thread(target=tarea, args=(i,)) for i in range(3)]
for h in hilos:
    h.start()
for h in hilos:
    h.join()          # espera a que cada hilo termine
```

Cuando varios hilos modifican un mismo dato hay que protegerlo con un `threading.Lock` (usado como `with lock:`) para evitar condiciones de carrera. Un hilo marcado con `daemon=True` no impide que el programa termine.

## ThreadPoolExecutor

`concurrent.futures.ThreadPoolExecutor` administra un **pool de hilos** reutilizables. Se usa como administrador de contexto, lo que garantiza que al salir del bloque se espere a todas las tareas y se liberen los hilos.

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request

def descargar(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return url, len(r.read())

urls = ["https://python.org", "https://docs.python.org", "https://pypi.org"]

with ThreadPoolExecutor(max_workers=8) as executor:
    futuros = [executor.submit(descargar, u) for u in urls]
    for futuro in as_completed(futuros):
        url, tamanio = futuro.result()
        print(url, tamanio)
```

- `executor.submit(fn, *args)` programa una llamada y devuelve inmediatamente un objeto **`Future`**, que representa un resultado que todavía no está disponible.
- `futuro.result(timeout=None)` bloquea hasta que el resultado esté listo; si la función lanzó una excepción, `result()` la vuelve a lanzar en el hilo principal.
- `as_completed(futuros)` entrega los futuros **a medida que terminan**, en el orden de finalización.
- `executor.map(fn, iterable)` es la alternativa simple: aplica la función a cada elemento y devuelve los resultados **en el orden de entrada**.

Si no se indica `max_workers`, el valor por defecto es `min(32, os.cpu_count() + 4)`.

## ProcessPoolExecutor

`ProcessPoolExecutor` tiene exactamente la misma interfaz, pero cada tarea corre en un **proceso separado**, lo que evita el GIL y aprovecha todos los núcleos para trabajo CPU-bound.

```python
from concurrent.futures import ProcessPoolExecutor

def es_primo(n):
    if n < 2:
        return False
    return all(n % d for d in range(2, int(n ** 0.5) + 1))

if __name__ == "__main__":
    numeros = range(10_000_000, 10_000_200)
    with ProcessPoolExecutor() as executor:
        resultados = list(executor.map(es_primo, numeros, chunksize=20))
    print(sum(resultados))
```

Diferencias importantes respecto de los hilos:

- La función y sus argumentos se envían a otro proceso mediante **pickle**: deben ser serializables y la función debe estar definida a nivel de módulo (no una lambda ni una función anidada).
- En Windows y macOS los procesos se crean con el método *spawn*, que vuelve a importar el módulo principal. Por eso el código que crea el pool **debe** estar protegido con `if __name__ == "__main__":`; de lo contrario se generan procesos en cascada.
- Crear procesos es más costoso que crear hilos, y los datos no se comparten en memoria: cada proceso trabaja con su propia copia.
- El parámetro `chunksize` de `map` agrupa elementos para reducir la sobrecarga de comunicación cuando hay muchas tareas pequeñas.

## Cancelación y manejo de errores

Un `Future` que todavía no empezó puede cancelarse con `futuro.cancel()`. `executor.shutdown(wait=True, cancel_futures=True)` (Python 3.9+) cancela las tareas pendientes. Las excepciones no se pierden: quedan guardadas en el futuro y aparecen al llamar a `result()`, o se consultan con `futuro.exception()`.

## Resumen: ¿qué herramienta elegir?

| Tipo de trabajo | Herramienta recomendada |
|---|---|
| Muchas operaciones de red con bibliotecas asíncronas | `asyncio` |
| I/O con bibliotecas sincrónicas (requests, drivers de BD) | `ThreadPoolExecutor` |
| Cálculo intensivo en Python puro | `ProcessPoolExecutor` |
