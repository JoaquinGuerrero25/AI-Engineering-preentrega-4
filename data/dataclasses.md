---
id: dataclasses
title: "dataclasses: clases de datos sin código repetitivo"
category: modelado-de-datos
tags: [dataclasses, dataclass, field, default_factory, frozen, post_init]
---

# dataclasses: clases de datos sin código repetitivo

El decorador `@dataclass` (módulo `dataclasses`, Python 3.7+) genera automáticamente los métodos habituales de una clase que sólo guarda datos: `__init__`, `__repr__` y `__eq__`, a partir de las anotaciones de tipo de sus atributos.

```python
from dataclasses import dataclass

@dataclass
class Producto:
    nombre: str
    precio: float
    stock: int = 0

p = Producto("Teclado", 25_000.0)
print(p)                    # Producto(nombre='Teclado', precio=25000.0, stock=0)
print(p == Producto("Teclado", 25_000.0))   # True
```

Las anotaciones de tipo son obligatorias para que un atributo sea un campo, pero **no se validan** en tiempo de ejecución: `Producto("x", "caro")` se crea sin error. Para validación se usan bibliotecas como Pydantic o se valida manualmente en `__post_init__`.

Los campos con valor por defecto deben ir después de los campos sin valor por defecto; de lo contrario se produce `TypeError: non-default argument follows default argument`.

## Valores por defecto mutables: field(default_factory=...)

No se puede usar directamente una lista, un diccionario o un conjunto como valor por defecto de un campo:

```python
@dataclass
class Carrito:
    items: list = []        # ValueError: mutable default <class 'list'> for field items is not allowed
```

El decorador lo prohíbe porque esa única lista sería **compartida por todas las instancias**. La solución es `field(default_factory=list)`, que llama a la fábrica para crear un objeto nuevo en cada instancia:

```python
from dataclasses import dataclass, field

@dataclass
class Carrito:
    cliente: str
    items: list[str] = field(default_factory=list)
    descuentos: dict[str, float] = field(default_factory=dict)

a = Carrito("Ana")
b = Carrito("Luis")
a.items.append("mouse")
print(b.items)              # [] -> cada carrito tiene su propia lista
```

`default_factory` acepta cualquier callable sin argumentos, por ejemplo `field(default_factory=lambda: ["general"])` o `field(default_factory=datetime.now)`.

## Opciones de field()

`field()` también permite ajustar cada campo por separado:

- `repr=False`: excluye el campo del `__repr__` (útil para contraseñas o datos muy largos).
- `compare=False`: el campo no participa de `__eq__` ni del ordenamiento.
- `init=False`: el campo no aparece en el `__init__`; suele calcularse en `__post_init__`.
- `kw_only=True`: el campo sólo puede pasarse por nombre.

## __post_init__: validación y campos derivados

El método `__post_init__` se ejecuta al final del `__init__` generado. Es el lugar para validar datos o calcular campos derivados:

```python
@dataclass
class Rectangulo:
    ancho: float
    alto: float
    area: float = field(init=False)

    def __post_init__(self):
        if self.ancho <= 0 or self.alto <= 0:
            raise ValueError("Las medidas deben ser positivas")
        self.area = self.ancho * self.alto
```

Los parámetros declarados como `InitVar[...]` se reciben en el `__init__` y se pasan a `__post_init__`, pero no se guardan como campos.

## Parámetros del decorador

- `frozen=True`: instancias **inmutables**. Asignar un atributo lanza `dataclasses.FrozenInstanceError`, y la instancia pasa a ser hashable (se puede usar como clave de diccionario o en un `set`).
- `order=True`: genera `__lt__`, `__le__`, `__gt__` y `__ge__`, comparando los campos en orden como si fueran tuplas.
- `slots=True` (Python 3.10+): usa `__slots__`, lo que reduce el consumo de memoria y acelera el acceso a atributos.
- `kw_only=True` (Python 3.10+): todos los campos deben pasarse por nombre.
- `eq=False`: no genera `__eq__` (se compara por identidad).

## Funciones auxiliares

- `asdict(instancia)` convierte la dataclass (y las dataclasses anidadas) en un diccionario; útil para serializar a JSON.
- `astuple(instancia)` la convierte en tupla.
- `replace(instancia, **cambios)` devuelve una **copia** con algunos campos modificados; es la forma idiomática de "modificar" una dataclass congelada.
- `fields(clase)` devuelve la descripción de los campos, útil para introspección.

## dataclass vs NamedTuple vs Pydantic

`typing.NamedTuple` crea tuplas inmutables con nombre, livianas pero sin métodos `__post_init__`. `@dataclass` es más flexible y mutable por defecto. Pydantic agrega validación y conversión de tipos en tiempo de ejecución, ideal para datos que vienen de afuera (APIs, archivos de configuración).
