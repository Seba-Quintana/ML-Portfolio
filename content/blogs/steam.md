+++
title = 'Recomendaciones de juegos'
date = 2024-10-02T01:37:27-03:00
draft = false
+++

<img src="https://sm.ign.com/ign_es/tag/s/steam/steam_rbez.jpg" alt="drawing" width="50%" style="display: block; margin-left: auto; margin-right: auto; margin-bottom: 5%; width: 50%;"/>

### Preparacion de datos

La idea principal de este post es analizar los inicios de la preparación de datos en datasets reales con grandes cantidades de información.
Para esto se eligió el dataset de _[recomendaciones de juegos en Steam](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam)_ (plataforma de distribución digital de videojuegos), el cual cuenta con datos oficiales de la plataforma, y según la descripción del mismo esta compuesto por mas de 41 millones de recomendaciones de usuarios de Steam.
Este dataset contiene 3 CSVs, uno de juegos, otro de recomendaciones, y otro de usuarios, y tambien contiene un archivo json con metadata de los juegos.

Para comenzar con la preparación de datos, es importante conocer el dataset. Esto permitirá en primera instancia entender qué es lo que se puede hacer con los datos, y una vez definido el objetivo, validar que predictores son los más importantes para poder alcanzarlo.
Al tener 3 CSVs, será necesario también encontrar puntos en común para poder correlacionarlos, y de esa forma poder utilizar toda la información posible, lo que ayudará a conseguir un mejor resultado.

### Análisis de predictores

##### CSV de juegos:
Este CSV contiene datos acerca de una gran variedad de juegos de Steam. Dentro de los campos que este tiene se encuentran:
- **app_id**: ID del juego en Steam. Este campo será importante para poder "mergear" los datasets (luego se verá que tanto el json con metadata como el CSV de recomendaciones referencian este app_id) (campo integer)
- **title**: Nombre del juego. Notar que algunos nombres tienen mal el formato en excel, principalmente los que utilizan simbolos como tm (trademark) o copyright, pero dentro de un editor de texto se ven bien (lo que implica que hay que tener cuidado con el formato de caractéres disponibles a elegir). Tambien existen juegos en lenguajes con otros alfabetos, a primera vista se encuentran ejemplos como chino y japones.
- **date_release**: Fecha de salida del juego. En excel se pueden ver algunos campos marcados con ######, pero abiertos con un editor de texto se pueden ver las fechas sin problema (el formato es año-mes-dia)
- **win**: Indica si el juego es soportado por el sistema operativo Windows (campo booleano)
- **mac**: Indica si el juego es soportado por el sistema operativo Mac (campo booleano)
- **linux**: Indica si el juego es soportado por el sistema operativo Linux (campo booleano)
- **rating**: Que tanto recomienda la gente un juego (campo categórico polinomial, con categorías como: Positive, Very Positive, Mixed, Mostly Positive, Mostly Negative)
- **positive_ratio**: Ratio de feedback positivo (campo integer)
- **user_reviews**: Cantidad de reviews de usuarios (campo integer)
- **price_final**: Precio en dolares (campo real)
- **price_original**: Precio original del juego en dolares (previo al descuento), suele ser igual al price_final (campo real)
- **discount**: Descuento que se le hace al juego. Si el juego esta descontado, disminuye el price_final, lo que hace que sea distinto a price_original (campo real)
- **steam_deck**: Si el juego esta disponible o no para la Steam Deck (campo booleano)

##### CSV de usuarios:
Este CSV contiene información (anonimizada) acerca de usuarios registrados en Steam:
- **user_id**: ID de usuario autogenerada (campo integer)
- **products**: Cantidad de juegos comprados por el usuario (campo integer)
- **reviews**: Cantidad de reviews publicadas (campo integer)

##### CSV de recomendaciones:
Este CSV contiene información acerca de las recomendaciones de los usuarios sobre un juego en particular:
- **app_id**: ID del juego en Steam (campo integer)
- **helpful**: Cantidad de usuarios que encontrar la recomendación útil (campo integer)
- **funny**: Cantidad de usuarios que encontrar la recomendación graciosa (campo integer)
- **date**: Fecha de publicación (formato año-mes-día)
- **is_recommended**: Si el usuario recomienda o no el juego (campo booleano)
- **hours**: Cuantas horas jugó el usuario al juego (campo real)
- **user_id**: ID de usuario en Steam (campo integer)
- **review_id**: ID autogenerado de las reviews (campo integer)

##### JSON de metadata:
Este JSON profundiza y agrega información acerca de algunos de los juegos del CSV de juegos, con los siguientes datos:
- **app_id**: ID del juego en Steam (campo integer)
- **description**: Descripción del juego (campo string)
- **tags**: Lista de géneros del juego (campo categórico, con algunos ejemplos como: Action, Singleplayer, Hack and Slash, Controller, entre otros)

Una vez analizados y entendidos todos los datos, se puede evaluar que campos son los que nos interesan predecir. El caso más interesante sería el de ser capaz de predecir que juego le puede llegar a gustar a una persona (recomendacion de juegos).
Dado que la cantidad de datos en este dataset es tan grande, es de interés saber si podemos eliminar algun predictor que sepamos que no va a ser de utilidad. De esta forma podemos alivianar la carga sobre el modelo, ahorrando tiempo y costos de procesamiento. Para saber que predictores serán útiles para la tarea, se debe encontrar una forma de saber si a un usuario le podría gustar un juego o no. A lo largo de este proyecto se consideraron dos formas para recomendar juegos:
- Saber que juegos suelen jugar otros usuarios similares (por ejemplo los juegos que tienen o recomiendan sus amigos, otros usuarios de su mismo país, etc.)
- Recomendar juegos que tengan metadata similar a los que el usuario ya ha jugado y recomendado previamente


### Primer approach

Lo primero que se intentó fue mergear los tres CSVs y el JSON utilizando pandas, con el objetivo de poder ver de forma unificada la información, y para poder envíarsela a RapidMiner o a un modelo en Python de forma sencilla y poder trabajar sobre el mismo conjunto de datos.

```Python
import pandas as pd
import json

# leo los tres CSVs
games = pd.read_csv('games.csv')
recommendations = pd.read_csv('recommendations.csv')
users = pd.read_csv('users.csv')

# leo el JSON y lo convierto en un dataframe (tambien se puede usar pd.read_json())
json_file_path = 'games_metadata.json'
with open(json_file_path, 'r', encoding='utf-8') as f:
    json_data = [json.loads(line) for line in f]
df_metadata = pd.DataFrame(json_data)

# ahora que está todo en el mismo formato, puedo mergear los dataframes usando app_id y user_id
games_recommendations = pd.merge(recommendations, games, on='app_id')
full_data = pd.merge(games_recommendations, users, on='user_id')
full_data_metadata = pd.merge(full_data, df_metadata, on='app_id', how='left')
full_data_metadata.to_csv('steam_data.csv', index=False)

print(full_data_metadata.head())
```

(tambien es posible mergear los CSVs desde RapidMiner, pero consume una enorme cantidad de recursos, y de faltar no permite continuar con la ejecución del bloque)

El problema de este approach es que se genera un CSV con una cantidad de datos tan grande que RapidMiner, Excel, e incluso a editores de texto les cuesta incluso leerlo. Por lo tanto, se debe encontrar una forma de achicar los CSVs previo al mergeo.

### Segundo approach

Dado que simplemente mergear todos los datos resultó inviable, el siguiente paso fue buscar si habían filas que no se podrían enviar al modelo. Por ejemplo, los tags son absolutamente necesarios para poder realizar recomendaciones en base al género de un juego, por lo que si un juego no tiene tags, entonces se puede descartar el juego, dado que falta información suficiente como para recomendarlo (se pueden utilizar otros campos, pero la recomendación no será tan acertada). Por lo tanto, se puede reducir la cantidad de datos en gran medida removiendo las filas que no poseen tags. Tambien es posible aprovechar y remover las filas que tienen descripción vacía, para intentar reducir la cantidad de datos.
El problema que tiene esto es que aún así la cantidad de datos sigue siendo muy grande. Por esta razón, se creó un script que aparte de remover las filas con una descripción y una lista de tags vacía, segmenta el CSV resultante en distintos CSVs, dando como resultado unos aproximadamente 400 CSVs, cada uno con una porción del CSV original.

```Python
import dask.dataframe as dd
import numpy as np

dtype = {
    'description': 'object',
    'tags': 'object',
}

df = dd.read_csv('steam_data.csv', dtype=dtype)

# cambio las descripiones vacías por NaN
df['description'] = df['description'].replace('', np.nan)

# borro las filas donde los tags sean '[]'
df = df[df['tags'] != '[]']

# borro las filas que tengan NaN en la descripción
df = df.dropna(subset=['description'])

# Esta línea aparte de guardar el CSV lo divide en múltiples archivos (cleaned_dataset_1.csv, cleaned_dataset_2.csv, etc.)
df.to_csv('cleaned_dataset_*.csv', index=False)
```

### Siguiendo con el análisis...

Con este último paso ya se comienza a poseer una cantidad de datos manejable, que se puede introducir en RapidMiner para comenzar con el desarrollo del proceso. Pero esto si solo se considera el tamaño del dataset.
Otro problema que surge es que aun así RapidMiner no lo puede procesar, porque los géneros no dejan de ser una lista de tags separados por coma, lo que al momento de mergear con los CSVs genera un archivo que, al ingresar al RapidMiner, no sabe hasta donde llega cada columna, y por lo tanto genera errores.
Para solucionar esto hay varios caminos. El primero sería sustituir las comas por otro símbolo en la lista de tags. Esto soluciona el problema del parseo, pero queda pendiente el ver como se puede formatear el dato para que un modelo de machine learning lo pueda procesar.
Por otro lado, el segundo camino sería llevar cada uno de los tags a una columna binaria. Este segundo approach permite que la información se guarde de forma mas estructurada, y en forma de números binarios, que dependiendo del algoritmo de ML que se utilice, esto podría ser una ventaja (se pueden utilizar números o booleanos, y no representan un orden en particular, lo que sirve para este caso)

#### Bibliografía

- [Dataset](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam)
- [Pandas docs](http://pandas.pydata.org/pandas-docs/version/1.5/reference/api/pandas.DataFrame.dropna.html)
- [Remover celdas vacías](https://stackoverflow.com/questions/29314033/drop-rows-containing-empty-cells-from-a-pandas-dataframe)
- [Otros estudios del dataset](https://www.kaggle.com/code/felipereisdesouza/recommended-game-genres)
- [Otros estudios del dataset](https://www.kaggle.com/code/thakursankalp/steam-game-recommendation-engine)
