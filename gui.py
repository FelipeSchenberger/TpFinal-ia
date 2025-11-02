import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
import os
from PIL import Image, ImageTk
from src.db import create_db, insert_face, load_faces, clear_db
from src.preprocess import image_to_vector
from src.hamming import HammingNetwork

class FaceRecognitionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Reconocimiento Facial - Red de Hamming")
        self.root.geometry("900x700")

        # Crear la base de datos
        create_db()

        # Variables
        self.selected_images = []
        self.current_image = None

        self.create_widgets()

    def create_widgets(self):
        # Título principal
        title_label = tk.Label(self.root, text="Sistema de Reconocimiento Facial",
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=10)

        # Frame principal
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # ========== SECCIÓN DE REGISTRO ==========
        register_frame = ttk.LabelFrame(main_frame, text="Registrar Personas en la Base de Datos", padding=10)
        register_frame.pack(fill=tk.X, pady=(0, 10))

        # Frame para botones de registro
        reg_buttons_frame = ttk.Frame(register_frame)
        reg_buttons_frame.pack(fill=tk.X, pady=(0, 10))

        # Botón para seleccionar múltiples imágenes
        self.select_images_btn = ttk.Button(reg_buttons_frame, text="Seleccionar Imágenes",
                                           command=self.select_multiple_images)
        self.select_images_btn.pack(side=tk.LEFT, padx=(0, 10))

        # Botón para limpiar selección
        self.clear_selection_btn = ttk.Button(reg_buttons_frame, text="Limpiar Selección",
                                             command=self.clear_selection)
        self.clear_selection_btn.pack(side=tk.LEFT, padx=(0, 10))

        # Botón para limpiar base de datos
        self.clear_db_btn = ttk.Button(reg_buttons_frame, text="Limpiar Base de Datos",
                                      command=self.clear_database)
        self.clear_db_btn.pack(side=tk.LEFT, padx=(0, 10))

        # Lista de imágenes seleccionadas
        self.images_listbox = tk.Listbox(register_frame, height=6)
        self.images_listbox.pack(fill=tk.X, pady=(0, 10))

        # Frame para guardar en BD
        save_frame = ttk.Frame(register_frame)
        save_frame.pack(fill=tk.X)

        ttk.Label(save_frame, text="Nombre de la persona:").pack(side=tk.LEFT)
        self.person_name_entry = ttk.Entry(save_frame, width=30)
        self.person_name_entry.pack(side=tk.LEFT, padx=(5, 10))

        self.save_images_btn = ttk.Button(save_frame, text="Guardar en Base de Datos",
                                         command=self.save_images_to_db)
        self.save_images_btn.pack(side=tk.LEFT)

        # ========== SECCIÓN DE RECONOCIMIENTO ==========
        recognition_frame = ttk.LabelFrame(main_frame, text="Reconocer Persona", padding=10)
        recognition_frame.pack(fill=tk.X, pady=(0, 10))

        # Frame para botones de reconocimiento
        rec_buttons_frame = ttk.Frame(recognition_frame)
        rec_buttons_frame.pack(fill=tk.X, pady=(0, 10))

        self.select_test_image_btn = ttk.Button(rec_buttons_frame, text="Seleccionar Imagen para Reconocer",
                                               command=self.select_test_image)
        self.select_test_image_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.recognize_btn = ttk.Button(rec_buttons_frame, text="Reconocer",
                                       command=self.recognize_person)
        self.recognize_btn.pack(side=tk.LEFT)

        # Frame para mostrar imagen y resultado
        result_frame = ttk.Frame(recognition_frame)
        result_frame.pack(fill=tk.X, pady=(0, 10))

        # Imagen seleccionada
        self.image_label = ttk.Label(result_frame, text="No hay imagen seleccionada")
        self.image_label.pack(side=tk.LEFT, padx=(0, 20))

        # Resultado del reconocimiento
        self.result_label = ttk.Label(result_frame, text="", font=("Arial", 12, "bold"))
        self.result_label.pack(side=tk.LEFT)

        # ========== CONFIGURACIÓN ==========
        config_frame = ttk.LabelFrame(main_frame, text="Configuración", padding=10)
        config_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(config_frame, text="Threshold:").pack(side=tk.LEFT)
        self.threshold_var = tk.DoubleVar(value=0.23)

        # Campo de entrada para threshold
        self.threshold_entry = ttk.Entry(config_frame, textvariable=self.threshold_var, width=8)
        self.threshold_entry.pack(side=tk.LEFT, padx=(5, 10))
        self.threshold_entry.bind('<Return>', self.on_threshold_entry_change)
        self.threshold_entry.bind('<FocusOut>', self.on_threshold_entry_change)

        # Slider para threshold
        self.threshold_scale = ttk.Scale(config_frame, from_=0.01, to=1.0,
                                        variable=self.threshold_var, orient=tk.HORIZONTAL, length=200)
        self.threshold_scale.pack(side=tk.LEFT, padx=(5, 10))

        # Actualizar slider cuando cambie el entry
        self.threshold_scale.configure(command=self.on_threshold_scale_change)

        # ========== LOG ==========
        log_frame = ttk.LabelFrame(main_frame, text="Log de Actividad", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = ScrolledText(log_frame, height=8, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Mensaje inicial
        self.log_message("Sistema iniciado. Base de datos creada.")

    def log_message(self, message):
        """Añade un mensaje al log"""
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.configure(state=tk.DISABLED)
        self.log_text.see(tk.END)

    def on_threshold_scale_change(self, value):
        """Maneja cambios en el slider del threshold"""
        # No necesita hacer nada extra, la variable ya se actualiza automáticamente
        pass

    def on_threshold_entry_change(self, event=None):
        """Maneja cambios en el campo de entrada del threshold"""
        try:
            value = float(self.threshold_entry.get())
            if 0.01 <= value <= 1.0:
                self.threshold_var.set(value)
            else:
                # Si está fuera del rango, restablecer al valor anterior
                self.threshold_entry.delete(0, tk.END)
                self.threshold_entry.insert(0, f"{self.threshold_var.get():.3f}")
                messagebox.showwarning("Valor inválido", "El threshold debe estar entre 0.01 y 1.0")
        except ValueError:
            # Si no es un número válido, restablecer al valor anterior
            self.threshold_entry.delete(0, tk.END)
            self.threshold_entry.insert(0, f"{self.threshold_var.get():.3f}")
            messagebox.showwarning("Valor inválido", "Por favor ingresa un número válido")

    def update_threshold_label(self, value):
        """Actualiza el label del threshold (método legacy - ya no se usa)"""
        pass

    def select_multiple_images(self):
        """Permite seleccionar múltiples imágenes"""
        file_types = [("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        files = filedialog.askopenfilenames(title="Seleccionar Imágenes", filetypes=file_types)

        if files:
            self.selected_images = list(files)
            self.update_images_listbox()
            self.log_message(f"Seleccionadas {len(files)} imágenes")

    def update_images_listbox(self):
        """Actualiza la lista de imágenes seleccionadas"""
        self.images_listbox.delete(0, tk.END)
        for img_path in self.selected_images:
            filename = os.path.basename(img_path)
            self.images_listbox.insert(tk.END, filename)

    def clear_selection(self):
        """Limpia la selección de imágenes"""
        self.selected_images = []
        self.update_images_listbox()
        self.log_message("Selección de imágenes limpiada")

    def clear_database(self):
        """Limpia la base de datos"""
        if messagebox.askyesno("Confirmar", "¿Estás seguro de que quieres limpiar toda la base de datos?"):
            clear_db()
            self.log_message("Base de datos limpiada")

    def save_images_to_db(self):
        """Guarda las imágenes seleccionadas en la base de datos"""
        person_name = self.person_name_entry.get().strip()

        if not person_name:
            messagebox.showerror("Error", "Por favor ingresa el nombre de la persona")
            return

        if not self.selected_images:
            messagebox.showerror("Error", "Por favor selecciona al menos una imagen")
            return

        success_count = 0
        error_count = 0

        for img_path in self.selected_images:
            try:
                vector = image_to_vector(img_path, visualizar=False)
                insert_face(person_name, vector)
                success_count += 1
                self.log_message(f"✓ Guardada: {os.path.basename(img_path)}")
            except Exception as e:
                error_count += 1
                self.log_message(f"✗ Error en {os.path.basename(img_path)}: {str(e)}")

        self.log_message(f"Proceso completado: {success_count} éxitos, {error_count} errores")

        if success_count > 0:
            messagebox.showinfo("Éxito", f"Se guardaron {success_count} imágenes de '{person_name}' en la base de datos")
            self.person_name_entry.delete(0, tk.END)
            self.clear_selection()

    def select_test_image(self):
        """Selecciona una imagen para reconocer"""
        file_types = [("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        file_path = filedialog.askopenfilename(title="Seleccionar Imagen para Reconocer", filetypes=file_types)

        if file_path:
            self.current_image = file_path
            self.display_image(file_path)
            self.log_message(f"Imagen seleccionada: {os.path.basename(file_path)}")

    def display_image(self, img_path):
        """Muestra la imagen seleccionada"""
        try:
            # Cargar y redimensionar imagen
            pil_image = Image.open(img_path)
            pil_image.thumbnail((150, 150))

            # Convertir a PhotoImage para tkinter
            photo = ImageTk.PhotoImage(pil_image)

            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo  # Mantener referencia
        except Exception as e:
            self.image_label.configure(text=f"Error al cargar imagen: {str(e)}")
            self.log_message(f"Error al mostrar imagen: {str(e)}")

    def recognize_person(self):
        """Reconoce la persona en la imagen seleccionada"""
        if not self.current_image:
            messagebox.showerror("Error", "Por favor selecciona una imagen primero")
            return

        try:
            # Cargar datos de la base de datos
            names, vectors = load_faces()
            if not names:
                messagebox.showerror("Error", "No hay personas registradas en la base de datos")
                return

            # Procesar imagen
            vector = image_to_vector(self.current_image, visualizar=False)

            # Reconocer con la red de Hamming
            threshold = self.threshold_var.get()
            network = HammingNetwork(names, vectors, threshold=threshold)
            nombre, distancia, best_index = network.classify(vector)

            # Mostrar resultado
            if nombre == "unknown":
                closest_name = network.names[best_index]
                result_text = f"Desconocido\n(Más parecido a: {closest_name})\nDistancia: {distancia:.4f}"
                self.result_label.configure(text=result_text, foreground="orange")
                self.log_message(f"Resultado: Persona desconocida. Más parecido a '{closest_name}' (distancia={distancia:.4f})")
            else:
                result_text = f"Reconocido: {nombre}\nDistancia: {distancia:.4f}"
                self.result_label.configure(text=result_text, foreground="green")
                self.log_message(f"Resultado: Persona reconocida como '{nombre}' (distancia={distancia:.4f})")

        except Exception as e:
            messagebox.showerror("Error", f"Error durante el reconocimiento: {str(e)}")
            self.log_message(f"Error en reconocimiento: {str(e)}")

def main():
    root = tk.Tk()
    app = FaceRecognitionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()