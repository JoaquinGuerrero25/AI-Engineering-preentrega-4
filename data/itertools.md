---
id: itertools
title: "itertools: iteradores eficientes"
category: programacion-funcional
tags: [itertools, iteradores, groupby, chain, islice, combinatoria, batched]
---

# itertools: iteradores eficientes

El módulo `itertools` ofrece bloques de construcción para trabajar con iteradores de forma **perezosa** (*lazy*): los elementos se generan de a uno cuando se necesitan, sin construir listas completas en memoria. Esto permite procesar secuencias enormes o incluso infinitas.

## Iteradores infinitos

- `count(inicio=0, paso=1)` genera 0, 1, 2, 3… indefinidamente.
- `cycle(iterable)` repite los elementos de un iterable una y otra vez.
- `repeat(objeto, veces)` repite un mismo objeto; sin `veces`, para siempre.

Siempre deben combinarse con algo que los corte, como `islice`, `zip` o un `break`.

## Recortar y encadenar

```python
from itertools import islice, chain

primeros = list(islice(range(1_000_000), 5))           # [0, 1, 2, 3, 4]
pares = list(islice(range(100), 0, 10, 2))              # [0, 2, 4, 6, 8]
todo = list(chain([1, 2], (3, 4), "ab"))                # [1, 2, 3, 4, 'a', 'b']
aplanado = list(chain.from_iterable([[1, 2], [3], [4, 5]]))  # [1, 2, 3, 4, 5]
```

- `islice(iterable, stop)` o `islice(iterable, start, stop, step)` funciona como el rebanado `[a:b:c]` pero sobre cualquier iterador, incluidos generadores y archivos.
- `chain(*iterables)` recorre varios iterables uno detrás de otro; `chain.from_iterable` aplana un nivel de anidamiento.

## Agrupar con groupby

`groupby(iterable, key=None)` agrupa elementos **consecutivos** que comparten la misma clave. Un error muy común es olvidar que sólo agrupa elementos contiguos: para agrupar todos los elementos con la misma clave hay que **ordenar primero** por esa misma clave.

```python
from itertools import groupby

ventas = [("norte", 10), ("sur", 5), ("norte", 7), ("sur", 3)]
ventas.sort(key=lambda v: v[0])                 # imprescindible
for region, grupo in groupby(ventas, key=lambda v: v[0]):
    print(region, sum(monto for _, monto in grupo))
# norte 17
# sur 8
```

Cada `grupo` es un iterador que se consume al avanzar al siguiente grupo; si se necesita después, hay que convertirlo en lista.

## Filtrar

- `takewhile(predicado, it)` entrega elementos mientras el predicado sea verdadero y se detiene en el primero falso.
- `dropwhile(predicado, it)` descarta elementos mientras el predicado sea verdadero y luego entrega el resto.
- `filterfalse(predicado, it)` entrega los elementos para los que el predicado es falso.
- `compress(datos, selectores)` filtra usando una secuencia de booleanos.

## Combinatoria

- `product(a, b)` produce el producto cartesiano (equivale a bucles anidados); `product(range(2), repeat=3)` genera todas las combinaciones binarias de largo 3.
- `permutations(it, r)` genera todas las ordenaciones posibles de `r` elementos.
- `combinations(it, r)` genera subconjuntos de `r` elementos sin importar el orden y sin repetición.
- `combinations_with_replacement(it, r)` permite repetir elementos.

```python
from itertools import combinations
list(combinations("ABC", 2))    # [('A', 'B'), ('A', 'C'), ('B', 'C')]
```

## Acumular, emparejar y agrupar en lotes

- `accumulate(it, func=operator.add)` devuelve los resultados parciales: `accumulate([1, 2, 3, 4])` produce 1, 3, 6, 10 (sumas acumuladas). Con `func=max` da el máximo acumulado.
- `pairwise(it)` (Python 3.10+) devuelve pares consecutivos: `pairwise("ABCD")` → AB, BC, CD. Útil para calcular diferencias entre elementos sucesivos.
- `batched(it, n)` (Python 3.12+) agrupa en **lotes** de hasta `n` elementos: `batched(range(7), 3)` → (0, 1, 2), (3, 4, 5), (6,). Muy útil para enviar pedidos a una API en bloques, por ejemplo al subir vectores a una base vectorial.
- `zip_longest(a, b, fillvalue=None)` es como `zip`, pero continúa hasta el iterable más largo rellenando los faltantes.
- `starmap(func, iterable)` aplica una función desempaquetando cada tupla como argumentos.
- `tee(it, n)` crea `n` iteradores independientes a partir de uno solo.

## Por qué preferir itertools

Las funciones de `itertools` están implementadas en C, son rápidas y consumen memoria constante. Combinadas con expresiones generadoras permiten escribir *pipelines* de procesamiento de datos claros y eficientes, sin listas intermedias.
