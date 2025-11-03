How to use main.py

## Creamos un entorno virtual
python -m venv .venv

## Levantamos el entorno virtual
.\.venv\Scripts\activate (windows)
source .venv/bin/activate (linux)

## Instalamos las dependencias necesarias
pip install -r requirements.txt

## Bash
python main.py
Este comando crea la base de datos si es que no existe

## Elegi una opcion
1) Registrar persona
2) Reconocer persona
3) Salir
Elige:

1) ## En caso de elegir 1
Nombre de la persona: Alice
Ruta de la foto: images/alice.jpg
[OK] Persona Alice registrada en la BD

2) ## En caso de elegir 2
Ruta de la foto a reconocer: test.jpg
La foto se parece más a: Alice (distancia=0.23)

3) ## En caso de elegi 3
