+++
title = 'Recomendaciones de juegos'
date = 2024-10-02T01:37:27-03:00
draft = true
+++

<img src="https://sm.ign.com/ign_es/tag/s/steam/steam_rbez.jpg" alt="drawing" width="300"/>

### Preparacion de datos

El problema elegido es el de recomendaciones de videojuegos de steam.
El dataset contiene datos oficiales de la plataforma, y esta compuesto por mas de 41 millones de recomendaciones de usuarios de steam.
Viene con 3 csvs, uno de juegos, otro de recomendaciones, y otro de usuarios, y tambien contiene un archivo json con metadata de los juegos.

csv de juegos:
Este csv contiene datos acerca de los distintos juegos de steam. Dentro de los campos que este tiene se encuentran:
- app_id: este campo sera importante para mergear los datasets
- title: nombre del juego. Notar que algunos nombres tienen mal el formato en excel, principalmente los que utilizan simbolos como tm o copyright, pero dentro del texto se ven bien. Tambien existen juegos en lenguajes con otros alfabetos, a primera vista se encuentran ejemplos como chino y japones.
- date_release: fecha de salida. En excel se pueden ver algunos campos marcados con ######, pero abiertos con un editor de texto se pueden ver las fechas sin problema. El formato es mes/dia/anio
- win, mac, linux: SO en el que se puede jugar el juego (boolean)
- rating: que tanto recomienda la gente un juego
- positive-ratio: numero?
- user_reviews: cantidad de user_reviews
- price_final: precio en dolares
- price_original: precio original del juego, suele ser igual al price_final
- discount: descuento que se le hace al juego. Si el juego esta descontado, disminuye el price_final, lo que hace que sea distinto a price_original
- steam_deck: si el juego esta disponible o no para la steam_deck

Lo primero que se realizo fue mergear los tres csvs, utilizando pandas (para mas detalles merge.py)

El problema, es que solo los datos de las resenas no son los datos mas utiles a la hora de saber si se le recomienda un juego a un usuario. Para este enfoque hay dos opciones (complementarias): saber que juegos juegan otros usuarios (segun amigos, pais, etc), o encontrar juegos similares a los que ya jugo (segun genero).

El archivo .json contiene una descripcion del juego yb los distintos generos que tiene, por lo que el siguiente paso seria mergear el json con los csvs.

El problema de esto es que se genera una cantidad de datos demasiado grande como para que rapid miner, excel, e incluso editores de texto lo procesen. Por lo tanto, se debe encontrar una forma de achicar los csvs.

Para este proposito, se puede separar ese csv en muchos csvs mas pequeños. En este caso el resultado es de aproximadamente 400 csvs.

Otro problema que surge, es que aun asi rapid miner no lo puede procesar, porque los generos solian ser una lista de tags separados por coma, lo que al mergear con el csvs genera un archivo con errores.
Para solucionar esto hay varios caminos. El primero seria sustituir las comas por otro simbolo en la lista de tags, y el segundo seria llevar cada uno de los tags a una columna binaria. Este approach permite que la informacion se guarde de forma mas estructurada, y en forma de numeros binarios. Dependiendo de el algoritmo de ML que se utilice, esto podria ser una ventaja, dado que el algoritmo podra procesar un predictor de tipo binario mas facil que una lista en json.

