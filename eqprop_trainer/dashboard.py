"""
EqProp Trainer Dashboard

Main window with tabbed interface for Language Modeling and Vision training.
Features stunning dark cyberpunk theme with live pyqtgraph plots.
"""

import sys
from typing import Optional, Dict, List
from pathlib import Path

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QComboBox, QSpinBox, QDoubleSpinBox,
    QGroupBox, QTabWidget, QTextEdit, QProgressBar, QSlider,
    QSplitter, QFrame, QCheckBox, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont

try:
    import pyqtgraph as pg
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False

from .themes import CYBERPUNK_DARK, PLOT_COLORS
from .worker import TrainingWorker


class EqPropDashboard(QMainWindow):
    """Main dashboard window for EqProp training."""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("⚡ EqProp Trainer v0.1.0")
        self.setGeometry(100, 100, 1400, 900)
        
        # Apply theme
        self.setStyleSheet(CYBERPUNK_DARK)
        
        # Training state
        self.worker: Optional[TrainingWorker] = None
        self.model = None
        self.train_loader = None
        
        # Plot data
        self.loss_history: List[float] = []
        self.acc_history: List[float] = []
        self.lipschitz_history: List[float] = []
        
        # Initialize UI
        self._setup_ui()
        
        # Update timer for plots
        self.plot_timer = QTimer()
        self.plot_timer.timeout.connect(self._update_plots)
    
    def _setup_ui(self):
        """Set up the main user interface."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Header
        header = QLabel("⚡ EqProp Trainer")
        header.setObjectName("headerLabel")
        header.setFont(QFont("Segoe UI", 28, QFont.Weight.Bold))
        layout.addWidget(header)
        
        # Main content area with tabs
        tabs = QTabWidget()
        layout.addWidget(tabs, stretch=1)
        
        # Tab 1: Language Modeling
        lm_tab = self._create_lm_tab()
        tabs.addTab(lm_tab, "🔤 Language Model")
        
        # Tab 2: Vision
        vision_tab = self._create_vision_tab()
        tabs.addTab(vision_tab, "📷 Vision")
        
        # Status bar
        self.status_label = QLabel("Ready. Select a model and dataset to begin training.")
        self.status_label.setStyleSheet("color: #808090; padding: 5px;")
        layout.addWidget(self.status_label)
    
    def _create_lm_tab(self) -> QWidget:
        """Create the Language Modeling tab."""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setSpacing(15)
        
        # Left panel: Controls
        left_panel = QVBoxLayout()
        layout.addLayout(left_panel, stretch=1)
        
        # Model Selection
        model_group = QGroupBox("🧠 Model")
        model_layout = QGridLayout(model_group)
        
        model_layout.addWidget(QLabel("Architecture:"), 0, 0)
        self.lm_model_combo = QComboBox()
        self.lm_model_combo.addItems([
            "FullEqProp Transformer",
            "Attention-Only EqProp", 
            "Recurrent Core EqProp",
            "Hybrid EqProp",
            "LoopedMLP LM",
        ])
        model_layout.addWidget(self.lm_model_combo, 0, 1)
        
        model_layout.addWidget(QLabel("Hidden Dim:"), 1, 0)
        self.lm_hidden_spin = QSpinBox()
        self.lm_hidden_spin.setRange(64, 1024)
        self.lm_hidden_spin.setValue(256)
        self.lm_hidden_spin.setSingleStep(64)
        model_layout.addWidget(self.lm_hidden_spin, 1, 1)
        
        model_layout.addWidget(QLabel("Layers:"), 2, 0)
        self.lm_layers_spin = QSpinBox()
        self.lm_layers_spin.setRange(1, 12)
        self.lm_layers_spin.setValue(4)
        model_layout.addWidget(self.lm_layers_spin, 2, 1)
        
        model_layout.addWidget(QLabel("Eq Steps:"), 3, 0)
        self.lm_steps_spin = QSpinBox()
        self.lm_steps_spin.setRange(5, 50)
        self.lm_steps_spin.setValue(15)
        model_layout.addWidget(self.lm_steps_spin, 3, 1)
        
        left_panel.addWidget(model_group)
        
        # Dataset Selection
        data_group = QGroupBox("📚 Dataset")
        data_layout = QGridLayout(data_group)
        
        data_layout.addWidget(QLabel("Dataset:"), 0, 0)
        self.lm_dataset_combo = QComboBox()
        self.lm_dataset_combo.addItems([
            "tiny_shakespeare",
            "wikitext-2",
            "ptb",
        ])
        data_layout.addWidget(self.lm_dataset_combo, 0, 1)
        
        data_layout.addWidget(QLabel("Seq Length:"), 1, 0)
        self.lm_seqlen_spin = QSpinBox()
        self.lm_seqlen_spin.setRange(32, 512)
        self.lm_seqlen_spin.setValue(128)
        data_layout.addWidget(self.lm_seqlen_spin, 1, 1)
        
        data_layout.addWidget(QLabel("Batch Size:"), 2, 0)
        self.lm_batch_spin = QSpinBox()
        self.lm_batch_spin.setRange(8, 256)
        self.lm_batch_spin.setValue(64)
        data_layout.addWidget(self.lm_batch_spin, 2, 1)
        
        left_panel.addWidget(data_group)
        
        # Training Settings  
        train_group = QGroupBox("⚙️ Training")
        train_layout = QGridLayout(train_group)
        
        train_layout.addWidget(QLabel("Epochs:"), 0, 0)
        self.lm_epochs_spin = QSpinBox()
        self.lm_epochs_spin.setRange(1, 500)
        self.lm_epochs_spin.setValue(50)
        train_layout.addWidget(self.lm_epochs_spin, 0, 1)
        
        train_layout.addWidget(QLabel("Learning Rate:"), 1, 0)
        self.lm_lr_spin = QDoubleSpinBox()
        self.lm_lr_spin.setRange(0.0001, 0.1)
        self.lm_lr_spin.setValue(0.001)
        self.lm_lr_spin.setSingleStep(0.0001)
        self.lm_lr_spin.setDecimals(4)
        train_layout.addWidget(self.lm_lr_spin, 1, 1)
        
        self.lm_compile_check = QCheckBox("torch.compile (2x speedup)")
        self.lm_compile_check.setChecked(True)
        train_layout.addWidget(self.lm_compile_check, 2, 0, 1, 2)
        
        left_panel.addWidget(train_group)
        
        # Train/Stop Buttons
        btn_layout = QHBoxLayout()
        
        self.lm_train_btn = QPushButton("▶ Train")
        self.lm_train_btn.setObjectName("trainButton")
        self.lm_train_btn.clicked.connect(lambda: self._start_training('lm'))
        btn_layout.addWidget(self.lm_train_btn)
        
        self.lm_stop_btn = QPushButton("⏹ Stop")
        self.lm_stop_btn.setObjectName("stopButton")
        self.lm_stop_btn.setEnabled(False)
        self.lm_stop_btn.clicked.connect(self._stop_training)
        btn_layout.addWidget(self.lm_stop_btn)
        
        left_panel.addLayout(btn_layout)
        
        # Progress bar
        self.lm_progress = QProgressBar()
        self.lm_progress.setTextVisible(True)
        self.lm_progress.setFormat("Epoch %v / %m")
        left_panel.addWidget(self.lm_progress)
        
        # Stretch to push everything up
        left_panel.addStretch()
        
        # Right panel: Plots and Generation
        right_panel = QVBoxLayout()
        layout.addLayout(right_panel, stretch=2)
        
        if HAS_PYQTGRAPH:
            # Configure pyqtgraph for dark theme
            pg.setConfigOptions(antialias=True)
            
            # Loss/Accuracy plot
            metrics_group = QGroupBox("📊 Training Metrics")
            metrics_layout = QVBoxLayout(metrics_group)
            
            self.lm_loss_plot = pg.PlotWidget()
            self.lm_loss_plot.setBackground('#0a0a0f')
            self.lm_loss_plot.setLabel('left', 'Loss', color=PLOT_COLORS['loss'])
            self.lm_loss_plot.setLabel('bottom', 'Epoch')
            self.lm_loss_plot.showGrid(x=True, y=True, alpha=0.2)
            self.lm_loss_curve = self.lm_loss_plot.plot(pen=pg.mkPen(PLOT_COLORS['loss'], width=2))
            self.lm_acc_curve = self.lm_loss_plot.plot(pen=pg.mkPen(PLOT_COLORS['accuracy'], width=2))
            metrics_layout.addWidget(self.lm_loss_plot)
            
            # Lipschitz plot
            self.lm_lip_plot = pg.PlotWidget()
            self.lm_lip_plot.setBackground('#0a0a0f')
            self.lm_lip_plot.setLabel('left', 'Lipschitz L', color=PLOT_COLORS['lipschitz'])
            self.lm_lip_plot.setLabel('bottom', 'Epoch')
            self.lm_lip_plot.showGrid(x=True, y=True, alpha=0.2)
            self.lm_lip_plot.addLine(y=1.0, pen=pg.mkPen('r', width=1, style=Qt.PenStyle.DashLine))
            self.lm_lip_curve = self.lm_lip_plot.plot(pen=pg.mkPen(PLOT_COLORS['lipschitz'], width=2))
            metrics_layout.addWidget(self.lm_lip_plot)
            
            right_panel.addWidget(metrics_group, stretch=2)
        else:
            # Fallback text display
            no_plot_label = QLabel("Install pyqtgraph for live plots: pip install pyqtgraph")
            no_plot_label.setStyleSheet("color: #808090; padding: 20px;")
            right_panel.addWidget(no_plot_label)
        
        # Generation panel
        gen_group = QGroupBox("✨ Text Generation")
        gen_layout = QVBoxLayout(gen_group)
        
        gen_controls = QHBoxLayout()
        gen_controls.addWidget(QLabel("Temperature:"))
        self.temp_slider = QSlider(Qt.Orientation.Horizontal)
        self.temp_slider.setRange(1, 20)  # 0.1 to 2.0
        self.temp_slider.setValue(10)
        gen_controls.addWidget(self.temp_slider)
        self.temp_label = QLabel("1.0")
        self.temp_label.setFixedWidth(40)
        gen_controls.addWidget(self.temp_label)
        self.temp_slider.valueChanged.connect(lambda v: self.temp_label.setText(f"{v/10:.1f}"))
        
        gen_btn = QPushButton("Generate")
        gen_btn.clicked.connect(self._generate_text)
        gen_controls.addWidget(gen_btn)
        gen_layout.addLayout(gen_controls)
        
        self.gen_output = QTextEdit()
        self.gen_output.setReadOnly(True)
        self.gen_output.setPlaceholderText("Generated text will appear here...")
        gen_layout.addWidget(self.gen_output)
        
        right_panel.addWidget(gen_group, stretch=1)
        
        return tab
    
    def _create_vision_tab(self) -> QWidget:
        """Create the Vision training tab."""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setSpacing(15)
        
        # Left panel: Controls
        left_panel = QVBoxLayout()
        layout.addLayout(left_panel, stretch=1)
        
        # Model Selection
        model_group = QGroupBox("🧠 Model")
        model_layout = QGridLayout(model_group)
        
        model_layout.addWidget(QLabel("Architecture:"), 0, 0)
        self.vis_model_combo = QComboBox()
        
        # Core EqProp models
        model_items = [
            "LoopedMLP",
            "ConvEqProp",
            "BackpropMLP (baseline)",
        ]
        
        # Try to add bio-plausible algorithms from research codebase
        try:
            from eqprop_torch import HAS_BIOPLAUSIBLE, ALGORITHM_REGISTRY
            if HAS_BIOPLAUSIBLE:
                model_items.append("--- Bio-Plausible Research Models ---")
                # Add algorithms with descriptions
                for key, desc in ALGORITHM_REGISTRY.items():
                    # Format: "algorithm_key - Description"
                    model_items.append(f"{key} - {desc}")
        except:
            pass  # Bio-plausible models not available
        
        self.vis_model_combo.addItems(model_items)
        model_layout.addWidget(self.vis_model_combo, 0, 1)
        
        model_layout.addWidget(QLabel("Hidden Dim:"), 1, 0)
        self.vis_hidden_spin = QSpinBox()
        self.vis_hidden_spin.setRange(64, 1024)
        self.vis_hidden_spin.setValue(256)
        model_layout.addWidget(self.vis_hidden_spin, 1, 1)
        
        model_layout.addWidget(QLabel("Max Steps:"), 2, 0)
        self.vis_steps_spin = QSpinBox()
        self.vis_steps_spin.setRange(5, 100)
        self.vis_steps_spin.setValue(30)
        model_layout.addWidget(self.vis_steps_spin, 2, 1)
        
        left_panel.addWidget(model_group)
        
        # Dataset
        data_group = QGroupBox("📚 Dataset")
        data_layout = QGridLayout(data_group)
        
        data_layout.addWidget(QLabel("Dataset:"), 0, 0)
        self.vis_dataset_combo = QComboBox()
        self.vis_dataset_combo.addItems([
            "MNIST",
            "Fashion-MNIST",
            "CIFAR-10",
            "KMNIST",
        ])
        data_layout.addWidget(self.vis_dataset_combo, 0, 1)
        
        data_layout.addWidget(QLabel("Batch Size:"), 1, 0)
        self.vis_batch_spin = QSpinBox()
        self.vis_batch_spin.setRange(16, 512)
        self.vis_batch_spin.setValue(64)
        data_layout.addWidget(self.vis_batch_spin, 1, 1)
        
        left_panel.addWidget(data_group)
        
        # Training
        train_group = QGroupBox("⚙️ Training")
        train_layout = QGridLayout(train_group)
        
        train_layout.addWidget(QLabel("Epochs:"), 0, 0)
        self.vis_epochs_spin = QSpinBox()
        self.vis_epochs_spin.setRange(1, 100)
        self.vis_epochs_spin.setValue(10)
        train_layout.addWidget(self.vis_epochs_spin, 0, 1)
        
        train_layout.addWidget(QLabel("Learning Rate:"), 1, 0)
        self.vis_lr_spin = QDoubleSpinBox()
        self.vis_lr_spin.setRange(0.0001, 0.1)
        self.vis_lr_spin.setValue(0.001)
        self.vis_lr_spin.setDecimals(4)
        train_layout.addWidget(self.vis_lr_spin, 1, 1)
        
        self.vis_compile_check = QCheckBox("torch.compile")
        self.vis_compile_check.setChecked(True)
        train_layout.addWidget(self.vis_compile_check, 2, 0, 1, 2)
        
        left_panel.addWidget(train_group)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.vis_train_btn = QPushButton("▶ Train")
        self.vis_train_btn.setObjectName("trainButton")
        self.vis_train_btn.clicked.connect(lambda: self._start_training('vision'))
        btn_layout.addWidget(self.vis_train_btn)
        
        self.vis_stop_btn = QPushButton("⏹ Stop")
        self.vis_stop_btn.setObjectName("stopButton")
        self.vis_stop_btn.setEnabled(False)
        self.vis_stop_btn.clicked.connect(self._stop_training)
        btn_layout.addWidget(self.vis_stop_btn)
        
        left_panel.addLayout(btn_layout)
        
        self.vis_progress = QProgressBar()
        self.vis_progress.setFormat("Epoch %v / %m")
        left_panel.addWidget(self.vis_progress)
        
        left_panel.addStretch()
        
        # Right panel: Plots
        right_panel = QVBoxLayout()
        layout.addLayout(right_panel, stretch=2)
        
        if HAS_PYQTGRAPH:
            metrics_group = QGroupBox("📊 Training Metrics")
            metrics_layout = QVBoxLayout(metrics_group)
            
            self.vis_loss_plot = pg.PlotWidget()
            self.vis_loss_plot.setBackground('#0a0a0f')
            self.vis_loss_plot.setLabel('left', 'Loss / Accuracy')
            self.vis_loss_plot.setLabel('bottom', 'Epoch')
            self.vis_loss_plot.showGrid(x=True, y=True, alpha=0.2)
            self.vis_loss_curve = self.vis_loss_plot.plot(pen=pg.mkPen(PLOT_COLORS['loss'], width=2), name='Loss')
            self.vis_acc_curve = self.vis_loss_plot.plot(pen=pg.mkPen(PLOT_COLORS['accuracy'], width=2), name='Accuracy')
            metrics_layout.addWidget(self.vis_loss_plot)
            
            self.vis_lip_plot = pg.PlotWidget()
            self.vis_lip_plot.setBackground('#0a0a0f')
            self.vis_lip_plot.setLabel('left', 'Lipschitz L')
            self.vis_lip_plot.setLabel('bottom', 'Epoch')
            self.vis_lip_plot.showGrid(x=True, y=True, alpha=0.2)
            self.vis_lip_plot.addLine(y=1.0, pen=pg.mkPen('r', width=1, style=Qt.PenStyle.DashLine))
            self.vis_lip_curve = self.vis_lip_plot.plot(pen=pg.mkPen(PLOT_COLORS['lipschitz'], width=2))
            metrics_layout.addWidget(self.vis_lip_plot)
            
            right_panel.addWidget(metrics_group)
        
        # Stats display
        stats_group = QGroupBox("📈 Results")
        stats_layout = QGridLayout(stats_group)
        
        stats_layout.addWidget(QLabel("Test Accuracy:"), 0, 0)
        self.vis_acc_label = QLabel("--")
        self.vis_acc_label.setObjectName("metricLabel")
        stats_layout.addWidget(self.vis_acc_label, 0, 1)
        
        stats_layout.addWidget(QLabel("Final Loss:"), 1, 0)
        self.vis_loss_label = QLabel("--")
        stats_layout.addWidget(self.vis_loss_label, 1, 1)
        
        stats_layout.addWidget(QLabel("Lipschitz:"), 2, 0)
        self.vis_lip_label = QLabel("--")
        stats_layout.addWidget(self.vis_lip_label, 2, 1)
        
        right_panel.addWidget(stats_group)
        right_panel.addStretch()
        
        return tab
    
    def _start_training(self, mode: str):
        """Start training in background thread."""
        try:
            import torch
            from eqprop_torch import LoopedMLP, ConvEqProp, BackpropMLP
            
            if mode == 'vision':
                self._start_vision_training()
            else:
                self._start_lm_training()
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start training:\n{e}")
    
    def _start_vision_training(self):
        """Start vision model training."""
        import torch
        from eqprop_torch import LoopedMLP, ConvEqProp, BackpropMLP
        from eqprop_torch.datasets import get_vision_dataset
        from torch.utils.data import DataLoader
        
        # Get dataset
        dataset_name = self.vis_dataset_combo.currentText().lower().replace('-', '_')
        use_flatten = 'MLP' in self.vis_model_combo.currentText()
        
        train_data = get_vision_dataset(dataset_name, train=True, flatten=use_flatten)
        self.train_loader = DataLoader(train_data, batch_size=self.vis_batch_spin.value(), shuffle=True)
        
        # Create model
        hidden = self.vis_hidden_spin.value()
        model_name = self.vis_model_combo.currentText()
        
        # Check if it's a bio-plausible research algorithm
        if ' - ' in model_name:  # Bio-plausible models are formatted as "key - description"
            algorithm_key = model_name.split(' - ')[0]
            try:
                from eqprop_torch import HAS_BIOPLAUSIBLE
                from algorithms import create_model, AlgorithmConfig
                
                if not HAS_BIOPLAUSIBLE:
                    raise ImportError("Research algorithms not available")
                
                # Determine input_dim based on dataset
                if 'MNIST' in self.vis_dataset_combo.currentText():
                    input_dim = 784
                else:  # CIFAR-10
                    input_dim = 3072
                
                # Create research algorithm model directly (first-class nn.Module)
                self.model = create_model(
                    algorithm_key,
                    input_dim,
                    [hidden],  # Single hidden layer
                    10  # 10 classes
                )
            except Exception as e:
                QMessageBox.warning(self, "Model Creation Failed", 
                                   f"Could not create {algorithm_key}: {e}")
        elif 'LoopedMLP' in model_name:
            input_dim = 784 if 'MNIST' in self.vis_dataset_combo.currentText() else 3072
            self.model = LoopedMLP(input_dim, hidden, 10, max_steps=self.vis_steps_spin.value())
        elif 'ConvEqProp' in model_name:
            channels = 1 if 'MNIST' in self.vis_dataset_combo.currentText() else 3
            self.model = ConvEqProp(channels, hidden // 4, 10)
        else:  # BackpropMLP
            input_dim = 784 if 'MNIST' in self.vis_dataset_combo.currentText() else 3072
            self.model = BackpropMLP(input_dim, hidden, 10)
        
        # Clear history
        self.loss_history.clear()
        self.acc_history.clear()
        self.lipschitz_history.clear()
        
        # Create and start worker
        self.worker = TrainingWorker(
            self.model,
            self.train_loader,
            epochs=self.vis_epochs_spin.value(),
            lr=self.vis_lr_spin.value(),
            use_compile=self.vis_compile_check.isChecked(),
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        
        # Update UI
        self.vis_train_btn.setEnabled(False)
        self.vis_stop_btn.setEnabled(True)
        self.vis_progress.setMaximum(self.vis_epochs_spin.value())
        self.vis_progress.setValue(0)
        
        self.status_label.setText(f"Training {model_name}...")
        self.plot_timer.start(100)
        self.worker.start()
    
    def _start_lm_training(self):
        """Start language model training."""
        self.status_label.setText("LM training coming soon...")
    
    def _stop_training(self):
        """Stop training."""
        if self.worker:
            self.worker.stop()
            self.status_label.setText("Stopping training...")
    
    def _on_progress(self, metrics: dict):
        """Handle training progress update."""
        self.loss_history.append(metrics['loss'])
        self.acc_history.append(metrics['accuracy'])
        self.lipschitz_history.append(metrics['lipschitz'])
        
        # Update progress bar
        self.vis_progress.setValue(metrics['epoch'])
        self.lm_progress.setValue(metrics['epoch'])
        
        # Update labels
        self.vis_acc_label.setText(f"{metrics['accuracy']:.1%}")
        self.vis_loss_label.setText(f"{metrics['loss']:.4f}")
        self.vis_lip_label.setText(f"{metrics['lipschitz']:.4f}")
        
        self.status_label.setText(
            f"Epoch {metrics['epoch']}/{metrics['total_epochs']} | "
            f"Loss: {metrics['loss']:.4f} | "
            f"Acc: {metrics['accuracy']:.1%} | "
            f"L: {metrics['lipschitz']:.4f}"
        )
    
    def _update_plots(self):
        """Update plot curves."""
        if not HAS_PYQTGRAPH:
            return
        
        if self.loss_history:
            epochs = list(range(1, len(self.loss_history) + 1))
            self.vis_loss_curve.setData(epochs, self.loss_history)
            self.vis_acc_curve.setData(epochs, self.acc_history)
            self.vis_lip_curve.setData(epochs, self.lipschitz_history)
    
    def _on_finished(self, result: dict):
        """Handle training completion."""
        self.plot_timer.stop()
        self.vis_train_btn.setEnabled(True)
        self.vis_stop_btn.setEnabled(False)
        self.lm_train_btn.setEnabled(True)
        self.lm_stop_btn.setEnabled(False)
        
        if result.get('success'):
            self.status_label.setText(f"✓ Training complete! ({result['epochs_completed']} epochs)")
        else:
            self.status_label.setText("Training stopped.")
    
    def _on_error(self, error: str):
        """Handle training error."""
        self.plot_timer.stop()
        self.vis_train_btn.setEnabled(True)
        self.vis_stop_btn.setEnabled(False)
        self.status_label.setText("Training error!")
        QMessageBox.critical(self, "Training Error", error)
    
    def _generate_text(self):
        """Generate text from the model."""
        if self.model is None or not hasattr(self.model, 'generate'):
            self.gen_output.setText("No model loaded or model doesn't support generation.")
            return
        
        temperature = self.temp_slider.value() / 10.0
        self.gen_output.setText("Generating...")
        # In a real implementation, this would use the GenerationWorker
        self.gen_output.setText("Generation requires a trained LM model.")
