import sys
import numpy as np
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QLineEdit,
    QCheckBox,
)
from PySide6.QtCore import Qt, QTimer
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class SinusWidget(QWidget):
    """Widget principal affichant une courbe sinus interactive et animée.

    Cette classe intègre un graphique Matplotlib dans une interface PySide6
    avec des contrôles pour la fréquence, l'amplitude et la phase.
    """

    def __init__(self) -> None:
        """Initialise l'interface utilisateur et les paramètres du signal."""
        super().__init__()
        self.setWindowTitle("Exercice sinus animé ✨")

        # ---------- Paramètres initiaux ----------
        self.frequency: float = 1.0
        self.amplitude: float = 1.0
        self.phase: float = 0.0
        self.phase_increment: float = 0.1  # Variation de phase par frame

        # ---------- Graphique Matplotlib ----------
        self.fig: Figure = Figure()
        self.canvas: FigureCanvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot()
        self.x_array: np.ndarray = np.linspace(0, 10, 400)
        
        # Initialisation de la ligne du graphique
        (self.line,) = self.ax.plot(
            self.x_array, 
            self.amplitude * np.sin(self.frequency * self.x_array + self.phase)
        )
        self.ax.set_title("Sinus animé")
        self.ax.set_ylim(-5.5, 5.5)  # Fixe les limites pour éviter les sauts visuels
        self.ax.grid(True)

        # ---------- Mise en page (Layout) ----------
        layout: QVBoxLayout = QVBoxLayout(self)
        layout.addWidget(self.canvas)

        # Création des contrôles pour la fréquence, l'amplitude et la phase
        layout.addLayout(
            self.make_control("Fréquence", 0.1, 10.0, self.frequency, "frequency")
        )
        layout.addLayout(
            self.make_control("Amplitude", 0.1, 5.0, self.amplitude, "amplitude")
        )
        layout.addLayout(
            self.make_control("Phase", 0.0, 6.28, self.phase, "phase")
        )

        # Case à cocher pour l'animation
        self.animate_checkbox: QCheckBox = QCheckBox("Animer la phase")
        self.animate_checkbox.stateChanged.connect(self.toggle_animation)
        layout.addWidget(self.animate_checkbox)

        # ---------- Temporisateur (Timer) ----------
        self.timer: QTimer = QTimer()
        self.timer.timeout.connect(self.animate_phase)

    def make_control(self, label_text: str, min_val: float, max_val: float, 
                     default: float, attr: str) -> QHBoxLayout:
        """Crée un ensemble de contrôle composé d'un label, un slider et un champ texte.

        Args:
            label_text: Nom affiché du paramètre.
            min_val: Valeur minimale autorisée.
            max_val: Valeur maximale autorisée.
            default: Valeur initiale.
            attr: Nom de l'attribut de classe associé.

        Returns:
            Le layout horizontal contenant les widgets de contrôle.
        """
        hbox = QHBoxLayout()
        label = QLabel(label_text)

        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(100)
        # Calcul de la position initiale du slider (0-100)
        initial_slider_val = int(100 * (default - min_val) / (max_val - min_val))
        slider.setValue(initial_slider_val)

        line_edit = QLineEdit(f"{default:.2f}")

        # Stockage dynamique des widgets pour accès ultérieur
        setattr(self, f"{attr}_slider", slider)
        setattr(self, f"{attr}_edit", line_edit)
        setattr(self, f"{attr}_min", min_val)
        setattr(self, f"{attr}_max", max_val)

        # Connexion des signaux
        slider.valueChanged.connect(lambda val, a=attr: self.slider_changed(a, val))
        line_edit.editingFinished.connect(lambda a=attr: self.text_edited(a))

        hbox.addWidget(label)
        hbox.addWidget(slider)
        hbox.addWidget(line_edit)
        return hbox

    def slider_changed(self, attr: str, value: int) -> None:
        """Gère le changement de valeur via le curseur (slider).

        Args:
            attr: Nom du paramètre à modifier.
            value: Valeur entière brute du slider (0-100).
        """
        min_val = getattr(self, f"{attr}_min")
        max_val = getattr(self, f"{attr}_max")
        # Conversion de la valeur entière en valeur réelle flottante
        real_value = min_val + (max_val - min_val) * value / 100

        setattr(self, attr, real_value)
        line_edit = getattr(self, f"{attr}_edit")
        line_edit.setText(f"{real_value:.2f}")
        self.update_plot()

    def text_edited(self, attr: str) -> None:
        """Gère la modification manuelle du paramètre via le champ texte.

        Args:
            attr: Nom du paramètre à modifier.
        """
        line_edit = getattr(self, f"{attr}_edit")
        try:
            val = float(line_edit.text())
        except ValueError:
            # Réinitialise au texte précédent en cas d'erreur de saisie
            line_edit.setText(f"{getattr(self, attr):.2f}")
            return

        min_val = getattr(self, f"{attr}_min")
        max_val = getattr(self, f"{attr}_max")

        if min_val <= val <= max_val:
            setattr(self, attr, val)
            slider = getattr(self, f"{attr}_slider")
            # Mise à jour de la position du slider
            slider_val = int(100 * (val - min_val) / (max_val - min_val))
            slider.setValue(slider_val)
            self.update_plot()
        else:
            # Hors limites : on annule la saisie
            line_edit.setText(f"{getattr(self, attr):.2f}")

    def update_plot(self) -> None:
        """Met à jour les données du graphique et rafraîchit le canvas."""
        y = self.amplitude * np.sin(self.frequency * self.x_array + self.phase)
        self.line.set_ydata(y)
        self.canvas.draw_idle()  # Plus efficace que draw() pour des mises à jour fréquentes

    def animate_phase(self) -> None:
        """Incrémente la phase pour créer l'effet d'animation."""
        new_phase = self.phase + self.phase_increment
        # Boucle la phase entre 0 et 2π
        if new_phase > 2 * np.pi:
            new_phase -= 2 * np.pi
        self.phase = new_phase

        # Mise à jour visuelle du slider de phase
        slider = getattr(self, "phase_slider")
        min_val = getattr(self, "phase_min")
        max_val = getattr(self, "phase_max")
        slider_val = int(100 * (self.phase - min_val) / (max_val - min_val))
        
        # Bloque les signaux temporairement pour éviter une boucle de rétroaction
        slider.blockSignals(True)
        slider.setValue(slider_val)
        slider.blockSignals(False)
        
        self.update_plot()

    def toggle_animation(self, state: int) -> None:
        """Active ou désactive le timer d'animation selon l'état de la checkbox.

        Args:
            state: État de la case à cocher (0 ou 2 en Qt).
        """
        if state:
            self.timer.start(20)  # Mise à jour toutes les 20 ms (~50 FPS)
        else:
            self.timer.stop()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = SinusWidget()
    win.resize(700, 500)
    win.show()
    sys.exit(app.exec())