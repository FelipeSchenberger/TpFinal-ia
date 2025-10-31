import numpy as np

class HammingNetwork:
    def __init__(self, names, patterns, threshold=0.23):
        self.names = names
        self.patterns = np.array(patterns, dtype=float)
        self.threshold = threshold

    def classify(self, input_vector):

        input_vector = np.array(input_vector, dtype=float)

        distances = np.mean(np.abs(self.patterns - input_vector), axis=1)

        best_index = np.argmin(distances)
        best_distance = distances[best_index]

        if best_distance > self.threshold:
            return "Desconocido", best_distance, best_index
        else:
            return self.names[best_index], best_distance, best_index
