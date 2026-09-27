---
id: logging
title: "logging: registro de eventos de una aplicación"
category: observabilidad
tags: [logging, logger, handlers, formatter, niveles, RotatingFileHandler]
---

# logging: registro de eventos de una aplicación

El módulo `logging` es el sistema estándar de Python para registrar lo que ocurre en un programa. A diferencia de `print`, permite clasificar los mensajes por **nivel de severidad**, enviarlos a distintos destinos (consola, archivos, servicios externos) y cambiar el formato o el nivel sin tocar el código que genera los mensajes.

## Niveles

| Nivel | Valor | Uso típico |
|---|---|---|
| `DEBUG` | 10 | Detalle para diagnosticar problemas |
| `INFO` | 20 | Confirmación de que todo funciona como se espera |
| `WARNING` | 30 | Algo inesperado, pero el programa sigue (nivel por defecto) |
| `ERROR` | 40 | Una operación falló |
| `CRITICAL` | 50 | Error grave; el programa quizás no pueda continuar |

Un logger sólo emite los mensajes cuyo nivel es **mayor o igual** al nivel configurado. Como el nivel por defecto del logger raíz es `WARNING`, los mensajes `info` y `debug` no se ven hasta que se configura otro nivel.

## Configuración rápida con basicConfig

`logging.basicConfig()` configura el logger raíz en una sola llamada. Debe ejecutarse una vez, al inicio del programa, antes de emitir mensajes.

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename="app.log",
    filemode="a",
    encoding="utf-8",
)

logger = logging.getLogger(__name__)
logger.info("Aplicación iniciada")
```

- `filename` envía los mensajes a un archivo en lugar de la consola; `filemode="w"` lo sobrescribe en cada ejecución y `"a"` agrega al final.
- `format` define la plantilla del mensaje con atributos como `%(asctime)s` (fecha y hora), `%(levelname)s`, `%(name)s` (nombre del logger), `%(funcName)s`, `%(lineno)d` y `%(message)s`.
- `datefmt` controla el formato de la fecha de `%(asctime)s`, con la misma sintaxis que `time.strftime`.
- Si el logger raíz ya tiene handlers, `basicConfig` no hace nada; con `force=True` (Python 3.8+) se eliminan y se reconfigura.

## Loggers, handlers y formatters

Para aplicaciones más grandes conviene entender las piezas del sistema:

- **Logger**: el objeto con el que el código emite mensajes. Se obtiene con `logging.getLogger(__name__)`, de modo que cada módulo tiene su propio logger y los nombres forman una jerarquía (`app`, `app.db`, `app.api`). Los mensajes se **propagan** hacia los loggers padres salvo que `propagate = False`.
- **Handler**: decide **a dónde** va cada mensaje. `StreamHandler` escribe en consola, `FileHandler` en un archivo, `RotatingFileHandler` rota el archivo al alcanzar un tamaño y `TimedRotatingFileHandler` lo rota por tiempo (por ejemplo, a medianoche).
- **Formatter**: define **cómo** se ve cada línea.

```python
import logging
from logging.handlers import RotatingFileHandler

logger = logging.getLogger("app")
logger.setLevel(logging.DEBUG)

consola = logging.StreamHandler()
consola.setLevel(logging.INFO)

archivo = RotatingFileHandler("app.log", maxBytes=1_000_000, backupCount=5, encoding="utf-8")
archivo.setLevel(logging.DEBUG)

formato = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
consola.setFormatter(formato)
archivo.setFormatter(formato)

logger.addHandler(consola)
logger.addHandler(archivo)
```

En este ejemplo la consola muestra desde `INFO` y el archivo guarda todo desde `DEBUG`. `RotatingFileHandler` con `maxBytes=1_000_000` y `backupCount=5` mantiene `app.log` más cinco archivos anteriores (`app.log.1` … `app.log.5`).

## Buenas prácticas

- Usar formato **diferido**: `logger.info("Usuario %s creó %d pedidos", usuario, n)` en vez de un f-string. Así el texto sólo se arma si el mensaje realmente se va a emitir.
- Dentro de un bloque `except`, `logger.exception("mensaje")` registra en nivel `ERROR` e incluye automáticamente el *traceback* completo.
- Las bibliotecas no deben llamar a `basicConfig` ni agregar handlers: sólo crean su logger con `getLogger(__name__)` y dejan la configuración a la aplicación.
- No registrar datos sensibles (contraseñas, tokens, números de tarjeta).

## Configuración declarativa con dictConfig

`logging.config.dictConfig(config)` recibe un diccionario (que puede venir de un archivo YAML o JSON) con la definición de formatters, handlers y loggers. Permite cambiar la configuración de logging de producción sin modificar código.
