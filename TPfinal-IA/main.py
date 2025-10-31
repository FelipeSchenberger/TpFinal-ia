from db import create_db, insert_face, load_faces
from preprocess import image_to_vector
from hamming import HammingNetwork

def registrar_persona(nombre, path_foto):
    vec = image_to_vector(path_foto)
    if vec is not None:
        insert_face(nombre, vec)
        print(f"[OK] Persona '{nombre}' registrada en la BD.")

def reconocer_persona(path_foto):
    names, vectors = load_faces()
    if not names:
        print("[ERROR] No hay personas registradas en la base de datos.")
        return

    vec = image_to_vector(path_foto)
    if vec is not None:

        network = HammingNetwork(names, vectors, threshold=0.23)
        nombre, dist, best_index = network.classify(vec)

        if nombre == "Desconocido":
            closest_name = network.names[best_index]
            print(f"Resultado: Persona desconocida. Similitud más cercana con '{closest_name}' (distancia={dist:.4f}).")
            print("La distancia supera el umbral de reconocimiento.")
        else:
            print(f"Resultado: La persona reconocida es '{nombre}' (distancia={dist:.4f}).")

def main():
    create_db()
    while True:
        print("\n--- Menú Principal ---")
        opcion = input("1) Registrar persona\n2) Reconocer persona\n3) Salir\nElige una opción: ")

        try:
            if opcion == "1":
                nombre = input("Nombre de la persona: ")
                foto = input("Ruta de la foto: ")
                registrar_persona(nombre, foto)

            elif opcion == "2":
                foto = input("Ruta de la foto a reconocer: ")
                reconocer_persona(foto)

            elif opcion == "3":
                print("Saliendo del programa.")
                break

            else:
                print("[ERROR] Opción no válida. Inténtalo de nuevo.")
        except (FileNotFoundError, ValueError) as e:
            print(f"[ERROR] {e}")
        except Exception as e:
            print(f"[ERROR INESPERADO] {e}")

if __name__ == "__main__":
    main()
