import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict
from src.db import create_db, clear_db, insert_face, load_faces
from src.preprocess import image_to_vector
from src.hamming import HammingNetwork

images_folder = "images"
image_size = (63, 63) # mejor efectividad con esta resolución

def load_images(exclude_number):
    """
    Carga todas las imágenes de la carpeta images y las procesa.
    Las imágenes deben estar nombradas como: nombredelapersona-x.jpg
    Excluye las imágenes que tengan el número especificado.
    """

    if not os.path.exists(images_folder):
        print(f"La carpeta {images_folder} no existe")
        return

    image_paths = glob.glob(os.path.join(images_folder, "*.jpg"))

    if not image_paths:
        print(f"No se encontraron imágenes JPG en la carpeta {images_folder}")
        return

    processed_count = 0
    for image_path in image_paths:
        try:
            filename = os.path.basename(image_path)
            name_without_ext = os.path.splitext(filename)[0]

            if '-' in name_without_ext:
                person_name = '-'.join(name_without_ext.split('-')[:-1])
                image_number = name_without_ext.split('-')[-1]

                # Excluir imágenes con el número especificado
                if image_number == str(exclude_number):
                    continue
            else:
                person_name = name_without_ext

            vector = image_to_vector(image_path, size=image_size, visualizar=False)

            insert_face(person_name, vector)
            processed_count += 1

        except Exception as e:
            print(f"Error procesando {image_path}: {e}")

def test(test_image_number: int):
    names, vectors = load_faces()
    if not names:
        print("[ERROR] No hay personas registradas en la base de datos.")
        return []

    test_image_paths = glob.glob(os.path.join(images_folder, f"*-{test_image_number}.jpg"))

    if not test_image_paths:
        print(f"No se encontraron imágenes de prueba con el número {test_image_number} en la carpeta {images_folder}")
        return []

    results = []

    for threshold in range(1, 101):
        network = HammingNetwork(names, vectors, threshold=threshold / 100)

        for image_path in test_image_paths:
            try:
                filename = os.path.basename(image_path)
                name_without_ext = os.path.splitext(filename)[0]

                # Extraer el nombre real de la imagen
                if '-' in name_without_ext:
                    true_name = '-'.join(name_without_ext.split('-')[:-1])
                else:
                    true_name = name_without_ext

                vector = image_to_vector(image_path, size=image_size, visualizar=False)
                predicted_name, distance, best_index = network.classify(vector)

                # Determinar el tipo de resultado
                if predicted_name == "unknown":
                    result_type = "unknown"
                elif predicted_name == true_name:
                    result_type = "success"
                else:
                    result_type = "wrong_person"

                # Guardar resultado
                result = {
                    'filename': filename,
                    'true_name': true_name,
                    'predicted_name': predicted_name,
                    'result_type': result_type,
                    'threshold': threshold / 100,
                    'distance': distance
                }
                results.append(result)

            except Exception as e:
                print(f"Error procesando {image_path}: {e}")

    return results

def visualize_results(results):
    """
    Crea gráficas para visualizar los resultados del análisis
    """
    if not results:
        print("No hay resultados para visualizar")
        return

    # Crear figura con 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Gráfico 1: Gráfico de torta con porcentajes
    result_counts = Counter([r['result_type'] for r in results])
    total = len(results)

    labels = ['Éxitos', 'Fracasos', 'Unknown']
    values = [
        result_counts.get('success', 0),
        result_counts.get('wrong_person', 0),
        result_counts.get('unknown', 0)
    ]
    colors = ['#2ecc71', '#e74c3c', '#f39c12']

    # Filtrar valores que son 0 para que no aparezcan en la torta
    filtered_data = [(label, value, color) for label, value, color in zip(labels, values, colors) if value > 0]

    if filtered_data:
        filtered_labels, filtered_values, filtered_colors = zip(*filtered_data)

        # Calcular porcentajes
        percentages = [value/total*100 for value in filtered_values]

        # Crear gráfico de torta
        wedges, texts, autotexts = ax1.pie(filtered_values, labels=filtered_labels, colors=filtered_colors,
                                          autopct='%1.1f%%', startangle=90)

        # Mejorar la apariencia del texto
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)

        ax1.set_title('Distribución de Resultados (%)')
    else:
        ax1.text(0.5, 0.5, 'No hay datos para mostrar', ha='center', va='center', transform=ax1.transAxes)    # Gráfico 2: Análisis por threshold
    # Agrupar resultados por threshold
    threshold_stats = defaultdict(lambda: {'success': 0, 'wrong_person': 0, 'unknown': 0, 'total': 0})

    for result in results:
        threshold = result['threshold']
        result_type = result['result_type']
        threshold_stats[threshold][result_type] += 1
        threshold_stats[threshold]['total'] += 1

    # Preparar datos para el gráfico
    thresholds = sorted(threshold_stats.keys())
    success_rates = []
    unknown_rates = []
    wrong_rates = []

    for threshold in thresholds:
        stats = threshold_stats[threshold]
        total = stats['total']
        if total > 0:
            success_rates.append(stats['success'] / total * 100)
            unknown_rates.append(stats['unknown'] / total * 100)
            wrong_rates.append(stats['wrong_person'] / total * 100)
        else:
            success_rates.append(0)
            unknown_rates.append(0)
            wrong_rates.append(0)

    # Crear gráfico de líneas
    ax2.plot(thresholds, success_rates, label='Éxitos (%)', color='#2ecc71', linewidth=2)
    ax2.plot(thresholds, wrong_rates, label='Fracasos (%)', color='#e74c3c', linewidth=2)
    ax2.plot(thresholds, unknown_rates, label='Unknown (%)', color='#f39c12', linewidth=2)

    ax2.set_title('Rendimiento por Threshold')
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Porcentaje (%)')
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('validation_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    create_db()

    image_quantity_by_person = 5
    all_results = []

    for i in range(1, image_quantity_by_person + 1):
        clear_db()
        load_images(i)
        results = test(i)
        all_results.extend(results)

    visualize_results(all_results)

if __name__ == "__main__":
    main()
