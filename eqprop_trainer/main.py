#!/usr/bin/env python3
"""
EqProp-Trainer: Entry point for the dashboard application.

Usage:
    python -m eqprop_trainer
    # or after pip install:
    eqprop-trainer
"""

import sys


def main():
    """Launch the EqProp Trainer dashboard."""
    try:
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import Qt
    except ImportError:
        print("ERROR: PyQt6 not installed. Install with: pip install PyQt6 pyqtgraph")
        sys.exit(1)
    
    # High DPI support
    if hasattr(Qt.ApplicationAttribute, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    if hasattr(Qt.ApplicationAttribute, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    app.setApplicationName("EqProp Trainer")
    app.setApplicationVersion("0.1.0")
    
    # Import and create dashboard
    from .dashboard import EqPropDashboard
    
    window = EqPropDashboard()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
