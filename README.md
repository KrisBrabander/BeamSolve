# BeamSolve — Interactive 1D FEM Beam Analyzer

A professional desktop application for analyzing statically indeterminate beams using the **Direct Stiffness Method** (Finite Element Analysis).

## Features

- **Interactive Canvas** — Drag supports and loads along the beam in real-time
- **FEM Engine** — Euler-Bernoulli beam elements with Direct Stiffness Method
- **Statically Indeterminate** — Supports unlimited number of supports
- **Real-time Diagrams** — Shear force (V), bending moment (M), and deflection (w)
- **Exaggerated Deflection** — Visual deformed shape on the beam canvas
- **Dark Mode UI** — Modern, clean design inspired by premium SaaS dashboards
- **Preset Examples** — Simply supported, cantilever, and continuous beam

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

## Architecture

```
BeamSolve/
├── main.py                          # Entry point
├── requirements.txt
└── beamsolve/
    ├── core/
    │   ├── models.py                # Data models (Support, Load, etc.)
    │   └── solver.py                # FEM solver (Direct Stiffness Method)
    └── ui/
        ├── theme.py                 # Dark mode design system
        ├── beam_canvas.py           # Interactive QGraphicsView canvas
        ├── main_window.py           # Main application window
        └── widgets/
            ├── diagram_widget.py    # V, M, w diagram rendering
            └── properties_panel.py  # Side panel with controls
```

## Tech Stack

- **Python 3.10+**
- **PySide6** — Qt for Python (professional desktop GUI)
- **NumPy** — Matrix computations for FEM
