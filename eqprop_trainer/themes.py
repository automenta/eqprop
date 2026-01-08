"""
Dark Cyberpunk Theme for EqProp Trainer

Neon accents, glassmorphism, and futuristic aesthetics.
"""

CYBERPUNK_DARK = """
/* === Global === */
QWidget {
    background-color: #0a0a0f;
    color: #e0e0e0;
    font-family: 'Segoe UI', 'Inter', 'Roboto', sans-serif;
    font-size: 13px;
}

/* === Main Window === */
QMainWindow {
    background-color: #0a0a0f;
}

/* === Tab Widget === */
QTabWidget::pane {
    border: 1px solid #1a1a2e;
    background-color: rgba(15, 15, 25, 0.95);
    border-radius: 8px;
}

QTabBar::tab {
    background-color: #12121f;
    color: #808090;
    padding: 12px 24px;
    border: none;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    margin-right: 2px;
    font-weight: 600;
}

QTabBar::tab:selected {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #1e1e3f, stop:1 #14142a);
    color: #00ffff;
    border-bottom: 2px solid #00ffff;
}

QTabBar::tab:hover:!selected {
    background-color: #1a1a35;
    color: #a0a0b0;
}

/* === Group Boxes === */
QGroupBox {
    background-color: rgba(20, 20, 40, 0.7);
    border: 1px solid #252540;
    border-radius: 10px;
    margin-top: 14px;
    padding-top: 10px;
    font-weight: bold;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 15px;
    padding: 0 8px;
    color: #00ffff;
    font-size: 14px;
}

/* === Buttons === */
QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #2a2a4f, stop:1 #1a1a35);
    color: #e0e0e0;
    border: 1px solid #3a3a5f;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 600;
    min-width: 100px;
}

QPushButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #3a3a6f, stop:1 #2a2a4f);
    border-color: #00ffff;
}

QPushButton:pressed {
    background: #1a1a35;
}

QPushButton#trainButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #00aa88, stop:1 #006655);
    color: white;
    font-size: 16px;
    padding: 15px 40px;
    border: none;
}

QPushButton#trainButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #00ccaa, stop:1 #008877);
    box-shadow: 0 0 20px rgba(0, 255, 200, 0.4);
}

QPushButton#stopButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #cc3355, stop:1 #992244);
    color: white;
}

/* === Sliders === */
QSlider::groove:horizontal {
    height: 6px;
    background: #1a1a35;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    width: 18px;
    height: 18px;
    margin: -6px 0;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #00ffff, stop:1 #0088aa);
    border-radius: 9px;
    border: 2px solid #00aacc;
}

QSlider::handle:horizontal:hover {
    background: #00ffff;
    border-color: #00ffff;
}

QSlider::sub-page:horizontal {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #00aacc, stop:1 #00ffff);
    border-radius: 3px;
}

/* === Spin Boxes === */
QSpinBox, QDoubleSpinBox {
    background-color: #12121f;
    border: 1px solid #2a2a4f;
    border-radius: 6px;
    padding: 6px 10px;
    color: #e0e0e0;
}

QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: #00ffff;
}

QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {
    background-color: #1a1a35;
    border: none;
    width: 20px;
}

/* === Combo Boxes === */
QComboBox {
    background-color: #12121f;
    border: 1px solid #2a2a4f;
    border-radius: 6px;
    padding: 8px 12px;
    color: #e0e0e0;
    min-width: 150px;
}

QComboBox:hover {
    border-color: #00ffff;
}

QComboBox::drop-down {
    border: none;
    width: 30px;
}

QComboBox QAbstractItemView {
    background-color: #12121f;
    border: 1px solid #2a2a4f;
    selection-background-color: #00aacc;
    color: #e0e0e0;
}

/* === Progress Bar === */
QProgressBar {
    background-color: #12121f;
    border: 1px solid #2a2a4f;
    border-radius: 6px;
    height: 20px;
    text-align: center;
    color: white;
    font-weight: bold;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #00aacc, stop:0.5 #00ffff, stop:1 #00aacc);
    border-radius: 5px;
}

/* === Text Areas === */
QTextEdit, QPlainTextEdit {
    background-color: #0d0d15;
    border: 1px solid #252540;
    border-radius: 8px;
    color: #00ff88;
    font-family: 'JetBrains Mono', 'Consolas', monospace;
    font-size: 12px;
    padding: 10px;
}

/* === Labels === */
QLabel {
    color: #c0c0d0;
}

QLabel#headerLabel {
    font-size: 24px;
    font-weight: bold;
    color: #00ffff;
}

QLabel#metricLabel {
    font-size: 18px;
    font-weight: bold;
    color: #ff6688;
}

/* === Scrollbars === */
QScrollBar:vertical {
    background-color: #0a0a0f;
    width: 12px;
    border-radius: 6px;
}

QScrollBar::handle:vertical {
    background-color: #2a2a4f;
    border-radius: 6px;
    min-height: 40px;
}

QScrollBar::handle:vertical:hover {
    background-color: #3a3a6f;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

/* === Splitter === */
QSplitter::handle {
    background-color: #1a1a35;
}

QSplitter::handle:horizontal {
    width: 3px;
}

QSplitter::handle:vertical {
    height: 3px;
}

/* === Check Boxes === */
QCheckBox {
    spacing: 8px;
    color: #c0c0d0;
}

QCheckBox::indicator {
    width: 20px;
    height: 20px;
    border-radius: 4px;
    border: 2px solid #3a3a5f;
    background-color: #12121f;
}

QCheckBox::indicator:checked {
    background-color: #00aacc;
    border-color: #00ffff;
}

QCheckBox::indicator:hover {
    border-color: #00ffff;
}
"""

# Neon glow colors for plots
PLOT_COLORS = {
    'loss': '#ff5588',       # Pink
    'accuracy': '#00ff88',   # Green  
    'perplexity': '#ffaa00', # Orange
    'lipschitz': '#00ffff',  # Cyan
    'memory': '#aa88ff',     # Purple
    'gradient': '#ff88ff',   # Magenta
    'backprop': '#ff6666',   # Red
    'eqprop': '#66ff66',     # Green
}

# Animation durations (ms)
ANIMATION = {
    'button_glow': 300,
    'progress': 500,
    'plot_update': 50,
}
