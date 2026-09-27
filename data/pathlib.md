---
id: pathlib
title: "pathlib: rutas del sistema de archivos orientadas a objetos"
category: sistema-de-archivos
tags: [pathlib, Path, archivos, directorios, glob, os.path]
---

# pathlib: rutas del sistema de archivos orientadas a objetos

El módulo `pathlib` (Python 3.4+) representa las rutas de archivos y directorios como objetos `Path` en lugar de cadenas de texto. Reemplaza la mayoría de las funciones de `os.path`, funciona igual en Windows, Linux y macOS, y hace que el código sea más legible.

## Crear y combinar rutas

```python
from pathlib import Path

base = Path(__file__).resolve().parent     # carpeta del script actual
datos = base / "datos" / "ventas.csv"      # el operador / une segmentos
print(Path.cwd())                          # directorio de trabajo actual
print(Path.home())                         # carpeta del usuario
```

El operador `/` une segmentos de ruta usando el separador correcto del sistema operativo, sin concatenar cadenas a mano. `resolve()` devuelve la ruta absoluta y resuelve enlaces simbólicos y componentes `..`.

## Partes de una ruta

Para `p = Path("/proyectos/informe.final.pdf")`:

- `p.name` → `"informe.final.pdf"` (nombre completo)
- `p.stem` → `"informe.final"` (nombre sin la última extensión)
- `p.suffix` → `".pdf"` y `p.suffixes` → `[".final", ".pdf"]`
- `p.parent` → `Path("/proyectos")`
- `p.with_suffix(".docx")` y `p.with_name("otro.pdf")` devuelven rutas nuevas modificadas.

Los objetos `Path` son **inmutables**: estos métodos no renombran nada en disco, sólo crean otro objeto ruta.

## Consultar el sistema de archivos

- `p.exists()`, `p.is_file()`, `p.is_dir()` informan si la ruta existe y qué es.
- `p.stat().st_size` da el tamaño en bytes y `p.stat().st_mtime` la fecha de modificación.
- `p.iterdir()` recorre el contenido directo de un directorio.
- `p.glob("*.csv")` busca por patrón en un directorio y `p.rglob("*.py")` lo hace **recursivamente** en todos los subdirectorios. Desde Python 3.12, `Path.walk()` recorre el árbol como `os.walk`.

```python
for script in Path("src").rglob("*.py"):
    print(script, script.stat().st_size)
```

## Leer y escribir archivos

`Path` incluye atajos para leer o escribir un archivo completo en una línea:

```python
config = Path("config.txt")
config.write_text("modo=produccion\n", encoding="utf-8")
contenido = config.read_text(encoding="utf-8")
imagen = Path("logo.png").read_bytes()
```

Conviene indicar **siempre** `encoding="utf-8"`: si se omite, Python usa la codificación por defecto del sistema, que en Windows suele ser cp1252 y produce errores con tildes y eñes. Para archivos grandes o escritura incremental se usa `with p.open("a", encoding="utf-8") as f:`.

## Crear, mover y borrar

- `p.mkdir(parents=True, exist_ok=True)` crea el directorio y todos los padres que falten, sin error si ya existe.
- `p.touch()` crea un archivo vacío.
- `p.rename(destino)` mueve o renombra; `p.replace(destino)` además sobrescribe el destino si existe.
- `p.unlink(missing_ok=True)` borra un archivo (el parámetro `missing_ok` existe desde Python 3.8).
- `p.rmdir()` borra un directorio **vacío**; para borrar un árbol completo se usa `shutil.rmtree(p)`.

## Equivalencias con os.path

| os / os.path | pathlib |
|---|---|
| `os.path.join(a, b)` | `Path(a) / b` |
| `os.path.exists(p)` | `Path(p).exists()` |
| `os.path.basename(p)` | `Path(p).name` |
| `os.path.splitext(p)[1]` | `Path(p).suffix` |
| `os.path.dirname(p)` | `Path(p).parent` |
| `os.makedirs(p, exist_ok=True)` | `Path(p).mkdir(parents=True, exist_ok=True)` |
| `glob.glob("**/*.py", recursive=True)` | `Path().rglob("*.py")` |

Casi todas las funciones de la biblioteca estándar aceptan objetos `Path` directamente (`open`, `shutil.copy`, `json.load` con `p.open()`), porque implementan el protocolo `os.PathLike`. Si alguna biblioteca antigua exige una cadena, basta con `str(p)`.
