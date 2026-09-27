---
id: asyncio
title: "asyncio: programación asíncrona con corrutinas"
category: concurrencia
tags: [asyncio, corrutinas, async, await, event-loop, io-bound]
---

# asyncio: programación asíncrona con corrutinas

El módulo `asyncio` permite escribir código concurrente con la sintaxis `async` / `await`. Está pensado para tareas **limitadas por entrada/salida** (I/O-bound): pedidos HTTP, consultas a bases de datos, lectura de sockets o llamadas a APIs. Mientras una operación espera la respuesta de la red, el bucle de eventos (*event loop*) aprovecha ese tiempo para avanzar con otras tareas. Todo ocurre en un solo hilo, por lo que no hay condiciones de carrera clásicas entre hilos, pero tampoco hay paralelismo real de CPU.

## Corrutinas y el punto de entrada

Una función definida con `async def` es una **función corrutina**. Llamarla no ejecuta su cuerpo: devuelve un objeto corrutina que hay que esperar con `await` o programar en el bucle de eventos.

```python
import asyncio

async def saludar(nombre: str) -> str:
    await asyncio.sleep(1)          # cede el control al event loop durante 1 segundo
    return f"Hola, {nombre}"

async def main():
    mensaje = await saludar("Ada")
    print(mensaje)

asyncio.run(main())
```

`asyncio.run()` crea un event loop nuevo, ejecuta la corrutina principal hasta que termina y cierra el loop. Debe llamarse una sola vez, desde código sincrónico, como punto de entrada del programa. No se puede usar `await` fuera de una función `async` (salvo en el REPL `python -m asyncio`).

Un error frecuente es usar `time.sleep()` dentro de una corrutina: esa llamada **bloquea** todo el event loop y ninguna otra tarea avanza. Dentro de código asíncrono siempre hay que usar `await asyncio.sleep()`.

## Ejecutar varias corrutinas al mismo tiempo

Esperar corrutinas una detrás de otra con `await` las ejecuta en forma secuencial. Para que avancen de manera concurrente hay que convertirlas en **tareas**.

### asyncio.gather

`asyncio.gather(*aws)` ejecuta varias corrutinas o tareas en simultáneo y espera a que **terminen todas**. Devuelve una lista con los resultados en el mismo orden en que se pasaron los argumentos, sin importar cuál terminó primero.

```python
async def descargar(url: str) -> int:
    await asyncio.sleep(0.5)        # simula la espera de la red
    return len(url)

async def main():
    urls = ["https://a.com", "https://b.com", "https://c.com"]
    tamanios = await asyncio.gather(*(descargar(u) for u in urls))
    print(tamanios)                 # tarda ~0.5 s en total, no 1.5 s
```

Por defecto, si una de las corrutinas lanza una excepción, `gather` la propaga de inmediato al que espera. Con `return_exceptions=True` las excepciones se devuelven como un valor más dentro de la lista de resultados y el resto de las tareas sigue su curso.

### asyncio.create_task

`asyncio.create_task(coro)` programa una corrutina para que empiece a ejecutarse "en segundo plano" lo antes posible y devuelve un objeto `Task`. Luego se puede esperar con `await tarea`. Es importante guardar una referencia a la tarea: el event loop sólo mantiene referencias débiles y una tarea sin referencias puede ser recolectada antes de terminar.

### asyncio.TaskGroup (Python 3.11+)

`TaskGroup` es la forma moderna y recomendada de lanzar un grupo de tareas relacionadas. Al salir del bloque `async with` se espera a todas. Si alguna falla, las demás se cancelan y los errores se reportan juntos en un `ExceptionGroup` (que se captura con `except*`).

```python
async def main():
    async with asyncio.TaskGroup() as tg:
        t1 = tg.create_task(descargar("https://a.com"))
        t2 = tg.create_task(descargar("https://b.com"))
    print(t1.result(), t2.result())
```

## Tiempos límite y cancelación

`asyncio.wait_for(aw, timeout)` espera una corrutina durante un máximo de segundos; si se excede, cancela la tarea y lanza `TimeoutError`. Desde Python 3.11 también existe el administrador de contexto `asyncio.timeout(segundos)`, que aplica un límite a todo un bloque de código:

```python
async def main():
    try:
        async with asyncio.timeout(2):
            await descargar("https://lento.com")
    except TimeoutError:
        print("La operación tardó demasiado")
```

Una tarea se cancela con `tarea.cancel()`, lo que inyecta un `asyncio.CancelledError` dentro de la corrutina en su próximo `await`. Se puede capturar para liberar recursos, pero conviene volver a lanzarlo.

## Limitar la concurrencia y compartir datos

Lanzar miles de pedidos simultáneos puede saturar un servidor o superar el límite de solicitudes de una API. `asyncio.Semaphore(n)` permite que como máximo `n` corrutinas entren a la vez en una sección:

```python
sem = asyncio.Semaphore(5)

async def descargar_limitado(url):
    async with sem:                 # como máximo 5 descargas simultáneas
        return await descargar(url)
```

Para patrones productor-consumidor existe `asyncio.Queue`, una cola segura para corrutinas con `await cola.put(item)` y `await cola.get()`. También hay primitivas `asyncio.Lock` y `asyncio.Event`, equivalentes asíncronos de las de `threading`.

## Integrar código bloqueante

Si hay que llamar a una biblioteca sincrónica que bloquea (por ejemplo `requests` o una operación de disco lenta), `asyncio.to_thread(funcion, *args)` (Python 3.9+) la ejecuta en un hilo aparte y devuelve una corrutina que se puede esperar sin congelar el event loop. Para trabajo intensivo de CPU, en cambio, conviene un `ProcessPoolExecutor` mediante `loop.run_in_executor`.

## Cuándo usar asyncio

- **Sí**: muchas operaciones de red concurrentes, servidores web asíncronos (FastAPI, aiohttp), bots y clientes de APIs.
- **No**: cálculos pesados de CPU (usar procesos) o scripts simples con pocas llamadas, donde la complejidad extra no se justifica.
