import numpy as np

class HammingNetwork:
    def __init__(self, names, patterns, threshold=0.25, binary=False):
        """
        names: etiquetas o nombres de los patrones
        patterns: lista de patrones binarios (0/1) o vectores de características
        threshold: máximo valor de diferencia aceptado para considerar reconocimiento válido
        binary: si True, usa distancia de Hamming (solo para vectores 0/1)
        """
        self.names = names
        self.patterns = np.array(patterns, dtype=float)
        self.threshold = threshold
        self.binary = binary

        # Normalizamos solo si no son binarios
        if not self.binary:
            norms = np.linalg.norm(self.patterns, axis=1, keepdims=True)
            norms[norms == 0] = 1
            self.patterns = self.patterns / norms

    def classify(self, input_vector):
        x = np.array(input_vector, dtype=float)

        if not self.binary:
            # Normalizamos entrada (si son vectores de imagen o reales)
            norm = np.linalg.norm(x)
            if norm == 0:
                return "unknown", 1.0, -1
            x = x / norm

            # Similitud tipo coseno → luego la invertimos
            similarities = np.dot(self.patterns, x)
            similarities = np.clip(similarities, -1, 1)  # por estabilidad numérica

            # Convertimos similitud a "distancia"
            distances = 1 - similarities  # 0 = igual, 2 = opuesto
        else:
            # Distancia de Hamming directa (solo para patrones 0/1)
            distances = np.mean(np.abs(self.patterns - x), axis=1)

        # Elegimos el más parecido (menor distancia)
        best_index = np.argmin(distances)
        best_distance = distances[best_index]

        if best_distance > self.threshold:
            return "unknown", best_distance, best_index
        else:
            return self.names[best_index], best_distance, best_index
