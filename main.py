"""
BeamSolve — Interactive 1D FEM Beam Analyzer

A professional desktop application for analyzing statically indeterminate
beams using the Direct Stiffness Method (Finite Element Analysis).

Usage:
    python main.py
"""

import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from beamsolve.ui.theme import Theme
from beamsolve.ui.main_window import MainWindow


def main():
    # Enable high-DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("BeamSolve")
    app.setOrganizationName("BeamSolve")

    # Apply dark theme
    Theme.apply_stylesheet(app)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
