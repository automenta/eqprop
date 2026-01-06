
import sys
import torch
import numpy as np

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QCheckBox, QLabel, QGroupBox, QSplitter, QGridLayout,
    QSpinBox, QSlider, QTableWidgetItem
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPalette, QColor
import pyqtgraph as pg

from .widgets import ArchitectureView, StatsTable, AlgorithmCard
from .worker import TrainingWorker
from .algorithms import AlgorithmWrapper
from .types import TrainingState
from .utils import load_shakespeare

class BioTrainerGUI(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bio-Trainer v5.0")
        self.setGeometry(100, 100, 1600, 1000)
        
        # Data
        self.data, self.c2i, self.i2c = load_shakespeare()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # State
        self.algorithms = {}  # name -> AlgorithmWrapper
        self.history = {}  # name -> list of states
        self.stats_cache = {} # name -> dict
        
        # Curve Colors
        self.colors = {
            'Backprop': '#ff6b6b',
            'EqProp': '#4ecdc4',
            'DFA': '#45b7d1',
            'CHL': '#f9ca24',
            'Deep Hebbian': '#6c5ce7'
        }

        # Worker
        self.worker = TrainingWorker({}, self.data, self.device)
        self.worker.update_signal.connect(self.on_worker_update)
        self.worker.start()
        self.worker.pause() # Start paused
        
        # UI
        self.init_ui()
        
        # Timers
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_plots)
        self.update_timer.start(100) # 10 FPS UI updates
        
        self.gen_timer = QTimer()
        self.gen_timer.timeout.connect(self.trigger_generation)
        self.gen_timer.start(2000) # Text gen every 2s
        
    def init_ui(self):
        widget = QWidget()
        self.setCentralWidget(widget)
        layout = QVBoxLayout(widget)
        
        # Top-Level Splitter (Vertical): Top (Plots/Cards) | Bottom (Controls/Stats) ?
        # Or Left/Right? User asked for resizable areas.
        # Let's do:
        # Left: Controls + Stats + Arch (Vertical Splitter)
        # Right: Arena + Cards (Vertical Splitter)
        # Main: Horizontal Splitter
        
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # --- Left Panel ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0,0,0,0)
        
        left_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # 1. Controls
        self.controls = self.create_controls()
        left_splitter.addWidget(self.controls)
        
        # 2. Stats
        stats_group = QGroupBox("Real-time Statistics")
        sl = QVBoxLayout(stats_group)
        self.stats_table = StatsTable()
        sl.addWidget(self.stats_table)
        left_splitter.addWidget(stats_group)
        
        # 3. Architecture
        arch_group = QGroupBox("Network Topology")
        al = QVBoxLayout(arch_group)
        self.arch_view = ArchitectureView()
        al.addWidget(self.arch_view)
        left_splitter.addWidget(arch_group)
        
        left_splitter.setStretchFactor(0, 0)
        left_splitter.setStretchFactor(1, 1)
        left_splitter.setStretchFactor(2, 1)
        
        left_layout.addWidget(left_splitter)
        main_splitter.addWidget(left_panel)
        
        # --- Right Panel ---
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # 1. Arena
        self.arena = self.create_arena()
        right_splitter.addWidget(self.arena)
        
        # 2. Mechanics (Cards)
        # Wrap mechanics in a widget with scroll area if many? 
        # For now just the widget as before
        self.mechanics = self.create_mechanics()
        right_splitter.addWidget(self.mechanics)
        
        right_splitter.setStretchFactor(0, 7) # Arena
        right_splitter.setStretchFactor(1, 3) # Mechanics
        
        main_splitter.addWidget(right_splitter)
        main_splitter.setStretchFactor(0, 3) # Left
        main_splitter.setStretchFactor(1, 7) # Right
        
        layout.addWidget(main_splitter)

    def create_controls(self):
        gb = QGroupBox("Experiment Controls")
        layout = QVBoxLayout(gb)
        
        # Network Select
        layout.addWidget(QLabel("<b>Algorithms:</b>"))
        grid = QGridLayout()
        self.checks = {}
        names = ['Backprop', 'EqProp', 'DFA', 'CHL', 'Deep Hebbian']
        for i, name in enumerate(names):
            cb = QCheckBox(name)
            cb.stateChanged.connect(self.on_selection_change)
            self.checks[name] = cb
            grid.addWidget(cb, i//2, i%2)
        layout.addLayout(grid)
        
        layout.addSpacing(10)
        
        # Architecture
        layout.addWidget(QLabel("<b>Architecture:</b>"))
        arch_layout = QHBoxLayout()
        
        # Depth
        d_layout = QVBoxLayout()
        d_layout.addWidget(QLabel("Depth"))
        self.depth_spin = QSpinBox()
        self.depth_spin.setRange(1, 100)
        self.depth_spin.setValue(20)
        self.depth_spin.valueChanged.connect(self.on_arch_change)
        d_layout.addWidget(self.depth_spin)
        arch_layout.addLayout(d_layout)
        
        # Width
        w_layout = QVBoxLayout()
        w_layout.addWidget(QLabel("Width"))
        self.width_spin = QSpinBox()
        self.width_spin.setRange(8, 512)
        self.width_spin.setValue(128)
        self.width_spin.valueChanged.connect(self.on_arch_change)
        w_layout.addWidget(self.width_spin)
        arch_layout.addLayout(w_layout)
        
        layout.addLayout(arch_layout)
        
        layout.addSpacing(10)
        
        # Hyperparameters
        layout.addWidget(QLabel("<b>Hyperparameters:</b>"))
        
        # Learning Rate
        layout.addWidget(QLabel("Learning Rate (log)"))
        self.lr_slider = QSlider(Qt.Orientation.Horizontal)
        self.lr_slider.setRange(-50, -10) # 1e-5 to 1e-1
        self.lr_slider.setValue(-30) # 1e-3
        self.lr_slider.valueChanged.connect(self.on_param_change)
        layout.addWidget(self.lr_slider)
        
        # Beta (Nudge)
        layout.addWidget(QLabel("Nudge Strength (Beta)"))
        self.beta_spin = QSpinBox() # Using spinbox for precision
        self.beta_spin.setRange(0, 100) # x 0.01
        self.beta_spin.setValue(22) # 0.22
        self.beta_spin.valueChanged.connect(self.on_param_change)
        layout.addWidget(self.beta_spin)
        
        # Steps
        layout.addWidget(QLabel("Equilibrium Steps"))
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 100)
        self.steps_spin.setValue(30)
        self.steps_spin.valueChanged.connect(self.on_param_change)
        layout.addWidget(self.steps_spin)
        
        layout.addSpacing(15)
        
        # Action Buttons
        btn_layout = QHBoxLayout()
        self.btn = QPushButton("▶ Start")
        self.btn.clicked.connect(self.toggle_training)
        self.btn.setStyleSheet("font-weight: bold; background-color: #27ae60; color: white; padding: 5px;")
        btn_layout.addWidget(self.btn)
        
        self.reset_btn = QPushButton("↺ Reset")
        self.reset_btn.clicked.connect(self.reset_model)
        self.reset_btn.setStyleSheet("background-color: #c0392b; color: white; padding: 5px;")
        btn_layout.addWidget(self.reset_btn)
        
        layout.addLayout(btn_layout)
        
        return gb
    
    def create_arena(self):
        """Top comparison panel."""
        widget = QWidget()
        layout = QGridLayout(widget) # 2x2 Grid
        
        self.loss_plot = pg.PlotWidget(title="Loss (Log)")
        self.loss_plot.setLogMode(y=True)
        self.loss_plot.addLegend()
        self.loss_plot.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.loss_plot, 0, 0)
        
        self.acc_plot = pg.PlotWidget(title="Accuracy")
        self.acc_plot.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.acc_plot, 0, 1)
        
        self.ppl_plot = pg.PlotWidget(title="Perplexity (Log)")
        self.ppl_plot.setLogMode(y=True)
        self.ppl_plot.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.ppl_plot, 1, 0)
        
        self.time_plot = pg.PlotWidget(title="Time/Iter (ms)")
        self.time_plot.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.time_plot, 1, 1)
        
        self.curves = {}
        return widget

    def create_mechanics(self):
        widget = QWidget()
        self.mech_layout = QGridLayout(widget)
        self.cards = {}
        
        for name, color in self.colors.items():
            card = AlgorithmCard(name, color)
            self.cards[name] = card
        return widget

    def on_arch_change(self):
        depth = self.depth_spin.value()
        width = self.width_spin.value()
        self.arch_view.set_arch(depth, width)
        
        if not self.worker.paused:
             self.toggle_training() # Pause
        self.reset_model()

    def on_param_change(self):
        # Update live parameters without reset
        lr = 10 ** (self.lr_slider.value() / 10.0)
        beta = self.beta_spin.value() / 100.0
        steps = self.steps_spin.value()
        
        for name, algo in self.algorithms.items():
            algo.update_hyperparams(lr=lr, beta=beta, steps=steps)

    def on_selection_change(self):
        active = [n for n, cb in self.checks.items() if cb.isChecked()]
        self.arch_view.set_active_algos(active) # Update Architecture View
        
        # Logic to remove inactive algos from wrapper if needed, 
        # but simpler to just rebuild list when not running or ignore them.
        # For responsiveness, we just update the worker's list of active names implicitly
        # (needs reset to fully remove objects or complex sync)
        
        if not self.algorithms: # First time
             pass # Will init on start
        
        # Update UI layout
        self.rebuild_plots(active)
        self.rebuild_cards(active)

    def rebuild_plots(self, active_names):
        self.loss_plot.clear()
        self.acc_plot.clear()
        self.ppl_plot.clear()
        self.time_plot.clear()
        self.curves = {}
        
        for name in active_names:
            pen = pg.mkPen(color=self.colors[name], width=2)
            self.curves[('loss', name)] = self.loss_plot.plot(name=name, pen=pen)
            self.curves[('acc', name)] = self.acc_plot.plot(name=name, pen=pen)
            self.curves[('ppl', name)] = self.ppl_plot.plot(name=name, pen=pen)
            self.curves[('time', name)] = self.time_plot.plot(name=name, pen=pen)

    def rebuild_cards(self, active_names):
        # Clear
        for i in reversed(range(self.mech_layout.count())):
            self.mech_layout.itemAt(i).widget().setParent(None)
        
        if active_names:
            cols = max(1, len(active_names))
            for i, name in enumerate(active_names):
                if name in self.cards:
                    # Reset state if re-adding
                    if not self.cards[name].state:
                        self.cards[name].update_state(TrainingState())
                    self.mech_layout.addWidget(self.cards[name], 0, i)
    
    def reset_model(self):
        self.worker.mutex.lock()
        self.algorithms = {}
        self.history = {}
        self.stats_cache = {}
        self.worker.algorithms = {}
        self.worker.iteration = 0
        self.worker.mutex.unlock()
        
        # Clear visuals
        for name in self.cards:
            self.cards[name].update_state(TrainingState())
        self.rebuild_plots([n for n, cb in self.checks.items() if cb.isChecked()])
        print("Model Reset")

    def toggle_training(self):
        if not self.worker.paused:
            self.worker.pause()
            self.btn.setText("▶ Start")
            self.btn.setStyleSheet("font-weight: bold; background-color: #27ae60; color: white; padding: 5px;")
        else:
            # Init if needed
            active = [n for n, cb in self.checks.items() if cb.isChecked()]
            if not active: return
            
            if not self.algorithms:
                # Create wrapper objects
                depth = self.depth_spin.value()
                width = self.width_spin.value()
                
                new_algos = {}
                for name in active:
                    new_algos[name] = AlgorithmWrapper(
                        name, len(self.c2i),
                        hidden_dim=width,
                        num_layers=depth,
                        device=self.device
                    )
                    # Apply current hyperparams
                    new_algos[name].update_hyperparams(
                         lr = 10 ** (self.lr_slider.value() / 10.0),
                         beta = self.beta_spin.value() / 100.0,
                         steps = self.steps_spin.value()
                    )
                    self.history[name] = []
                    self.stats_cache[name] = {
                        'acc': 0, 'ppl': 0, 'vram': 0, 'time': 0, 
                        'params': new_algos[name].param_count,
                        'best_acc': 0
                    }
                
                self.worker.mutex.lock()
                self.algorithms = new_algos
                self.worker.algorithms = new_algos
                self.worker.mutex.unlock()
            
            self.worker.resume()
            self.btn.setText("⏸ Pause")
            self.btn.setStyleSheet("font-weight: bold; background-color: #f39c12; color: white; padding: 5px;")

    def on_worker_update(self, name, state):
        # Called from background thread signal
        if name not in self.history: self.history[name] = []
        
        # Update history
        self.history[name].append(state)
        
        # Update stats cache
        best = self.stats_cache[name]['best_acc']
        if state.accuracy > best: best = state.accuracy
        
        self.stats_cache[name] = {
            'acc': state.accuracy,
            'ppl': state.perplexity,
            'vram': state.vram_gb,
            'time': state.iter_time,
            'params': getattr(self.algorithms[name], 'param_count', 0),
            'best_acc': best
        }

    def update_plots(self):
        """Redraw plots and table."""
        if not self.history: return
        
        # Update Table
        # Find global best acc
        global_best = 0
        for info in self.stats_cache.values():
            if info['acc'] > global_best: global_best = info['acc']
            
        # Mark best in cache
        for info in self.stats_cache.values():
            info['is_best_acc'] = (info['acc'] == global_best and global_best > 0)
            
        self.stats_table.update_stats(self.stats_cache)
        
        # Update Plots (Smoothed)
        window = 10
        for name, states in self.history.items():
            if not states: continue
            
            # Simple downsampling for performance
            # Only take last N points
            view_states = states[-1000:] 
            
            losses = [s.loss for s in view_states]
            accs = [s.accuracy for s in view_states]
            ppls = [s.perplexity for s in view_states]
            times = [s.iter_time * 1000 for s in view_states]
            
            # Smooth
            def smooth(v):
                if len(v) < window: return v
                return np.convolve(v, np.ones(window)/window, mode='valid')
            
            if ('loss', name) in self.curves:
                self.curves[('loss', name)].setData(smooth(losses))
            if ('acc', name) in self.curves:
                self.curves[('acc', name)].setData(smooth(accs))
            if ('ppl', name) in self.curves:
                self.curves[('ppl', name)].setData(smooth(ppls))
            if ('time', name) in self.curves:
                 self.curves[('time', name)].setData(smooth(times))

    def trigger_generation(self):
        """Trigger text generation in background or simply update UI cards."""
        # For simplicity/safety, we do generation in the UI thread for now, 
        # or we could push a request to the worker. 
        # Given it's infrequent (every 2s), UI thread is okay IF quick.
        # But generation can be slow. Let's trust the worker's training pause isn't needed
        # if we just use the model which is on GPU.
        # Ideally, we should pause worker, generate, resume.
        
        if self.worker.paused: return # Only gen while running or if requested manually
        
        # Only gen for active algos
        for name, algo in self.algorithms.items():
             # We need a seed
             seed = self.data[torch.randint(0, len(self.data)-64, (1,)).item():]
             try:
                 # Quick generation (short)
                 txt = algo.generate(seed[:64], self.i2c, length=30)
                 # Update latest state in history to reflect this sample
                 if self.history[name]:
                     self.history[name][-1].sample = txt
                     # Update card
                     if name in self.cards:
                         self.cards[name].update_state(self.history[name][-1])
             except:
                 pass
