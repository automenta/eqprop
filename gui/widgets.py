
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QCheckBox, QLabel, QGroupBox, QSplitter, QGridLayout, QFrame,
    QSpinBox, QSlider, QTableWidget, QTableWidgetItem, QHeaderView, QSizePolicy
)
from PyQt6.QtCore import Qt, QRectF, QPointF
from PyQt6.QtGui import QColor, QPainter, QPen, QFont, QBrush
import numpy as np

class ArchitectureView(QWidget):
    """Visualizes the network architecture dynamically."""
    def __init__(self):
        super().__init__()
        self.setMinimumSize(300, 150)
        self.depth = 20
        self.net_width = 128
        self.color = QColor(100, 100, 255)
        self.active_algos = []
    
    def set_arch(self, depth, width):
        self.depth = depth
        self.net_width = width
        self.update()

    def set_active_algos(self, algos):
        self.active_algos = algos
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.fillRect(self.rect(), QColor(25, 25, 25))
            
            w, h = self.width(), self.height()
            
            # If no active algos, show placeholder or generic wireframe
            if not self.active_algos:
                 painter.setPen(QColor(100, 100, 100))
                 painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No Algorithms Active")
                 return

            # Draw Layers
            # We visualize a generic schematic that represents the shared architecture 
            # (since depth/width are currently shared global controls)
            
            layer_gap = min(20, (w - 40) / (self.depth + 2))
            layer_h = min(h - 60, self.net_width / 2) # Scale width visually
            
            painter.setPen(QPen(QColor(60, 60, 60), 1))
            
            start_x = 20
            center_y = h / 2
            
            # Connections
            for i in range(self.depth + 1):
                 x = start_x + (i + 0.5) * layer_gap
                 next_x = start_x + (i + 1.5) * layer_gap
                 if i < self.depth:
                     painter.drawLine(QPointF(x, center_y), QPointF(next_x, center_y))

            # Nodes
            # Color code based on active algo (if single) or generic if multiple
            if len(self.active_algos) == 1:
                # Todo: pipe colors in. For now hardcoded blue.
                node_color = QColor(100, 100, 255) 
            else:
                node_color = QColor(200, 200, 200)

            painter.setBrush(QBrush(node_color))
            painter.setPen(Qt.PenStyle.NoPen)
            
            for i in range(self.depth):
                x = start_x + (i + 1) * layer_gap
                # Visual width
                vw = 4
                vh = layer_h
                painter.drawRoundedRect(QRectF(x - vw/2, center_y - vh/2, vw, vh), 2, 2)
                
            # Input / Output
            painter.setBrush(QBrush(QColor(200, 200, 200)))
            painter.drawEllipse(QPointF(start_x, center_y), 5, 5) # Input
            painter.drawEllipse(QPointF(start_x + (self.depth + 1) * layer_gap, center_y), 5, 5) # Output

            # Text Info
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont("Arial", 10))
            painter.drawText(10, 20, f"Architecture: Depth={self.depth}, Width={self.net_width}")
            
            # List active models
            painter.setFont(QFont("Arial", 8))
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(10, 35, f"Models: {', '.join(self.active_algos)}")

        finally:
            painter.end()


class StatsTable(QTableWidget):
    """Detailed statistics table."""
    def __init__(self):
        super().__init__()
        self.setColumnCount(6)
        self.setHorizontalHeaderLabels(["Algorithm", "Acc", "PPL", "VRAM", "Speed", "Params"])
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.verticalHeader().setVisible(False)
        self.setAlternatingRowColors(True)
        self.setStyleSheet("QTableWidget { background-color: #2b2b2b; color: white; gridline-color: #555; } "
                           "QHeaderView::section { background-color: #333; color: white; }")
    
    def update_stats(self, data):
        self.setRowCount(len(data))
        for i, (name, info) in enumerate(data.items()):
            self.setItem(i, 0, QTableWidgetItem(name))
            self.setItem(i, 1, QTableWidgetItem(f"{info['acc']:.1%}"))
            self.setItem(i, 2, QTableWidgetItem(f"{info['ppl']:.2f}"))
            self.setItem(i, 3, QTableWidgetItem(f"{info['vram']:.2f} GB"))
            self.setItem(i, 4, QTableWidgetItem(f"{1.0/max(info['time'], 1e-4):.1f} it/s"))
            self.setItem(i, 5, QTableWidgetItem(f"{info['params']/1e6:.1f}M"))

            # Highlight best accuracy
            if info.get('is_best_acc', False):
                 self.item(i, 1).setBackground(QColor(20, 100, 20))


class AlgorithmCard(QWidget):
    """Visual card for one algorithm."""
    
    def __init__(self, name, color):
        super().__init__()
        self.name = name
        self.color = QColor(color)
        self.state = None # Assigned later
        self.setMinimumSize(300, 150)
        self.accuracy = 0.0
        self.perplexity = 1.0
        self.sample = "..."
    
    def update_state(self, state):
        self.state = state
        self.accuracy = state.accuracy
        self.perplexity = state.perplexity
        self.sample = state.sample
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.fillRect(self.rect(), QColor(30, 30, 30))
            
            # Title
            painter.setPen(self.color)
            painter.setFont(QFont("Arial", 12, QFont.Weight.Bold))
            painter.drawText(10, 25, self.name)
            
            # Metrics overlay
            painter.setPen(QColor(200, 200, 200))
            painter.setFont(QFont("Arial", 9))
            painter.drawText(self.width() - 120, 25, f"Acc: {self.accuracy:.1%}")
            painter.drawText(self.width() - 120, 40, f"PPL: {self.perplexity:.1f}")

            # Sample text
            painter.setFont(QFont("Courier New", 9))
            painter.setPen(QColor(180, 180, 180))
            sample_txt = self.sample[:100] if self.sample else "..."
            rect = QRectF(10, 50, self.width()-20, 60)
            painter.drawText(rect, Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap, f'"{sample_txt}"')
            
        finally:
            painter.end()
