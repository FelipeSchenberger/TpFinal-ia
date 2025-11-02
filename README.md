How to use main.py

python -m venv ia

.\ia\Scripts\activate (windows)
source ia/bin/activate (linux)

pip install -r requirements.txt

## Bash
python main.py
This command create the database (if it doesn’t exist).

## Choose an option
1) Registrar persona
2) Reconocer persona
Elige:

1) ## In case of choosing n1
Nombre de la persona: Alice
Ruta de la foto: images/alice.jpg
[OK] Persona Alice registrada en la BD

2) ## In case of choosing n2
Ruta de la foto a reconocer: test.jpg
La foto se parece más a: Alice (distancia=42)
