# EqProp Trainer Dashboard

🚀 **Stunning PyQt6 dashboard for training Equilibrium Propagation models**

![Dark Cyberpunk Theme](https://placeholder.com/cyberpunk-theme.png)

## Features

- 🎨 **Dark Cyberpunk Theme** - Neon accents, glassmorphism, smooth animations
- 📊 **Live PyQtGraph Plots** - Real-time loss, accuracy, and Lipschitz tracking
- 🔤 **Language Modeling** - 5 EqProp LM variants with HuggingFace datasets
- 📷 **Vision Training** - MNIST, CIFAR-10, Fashion-MNIST support
- ⚙️ **Threaded Training** - Never freezes, stop anytime
- 🎯 **Model Presets** - Quick-start configurations

## Installation

```bash
# Install eqprop-torch library first
cd /path/to/eqprop
pip install -e .

# Install app dependencies
pip install PyQt6 pyqtgraph

# Or install with [app] extra
pip install -e ".[app]"
```

## Quick Start

```bash
# Launch dashboard
python -m eqprop_trainer

# Or directly
python eqprop_trainer/main.py
```

## Usage

### Vision Training

1. Select **Vision** tab
2. Choose model: `LoopedMLP`, `ConvEqProp`, or `BackpropMLP`
3. Select dataset: MNIST, Fashion-MNIST, CIFAR-10, KMNIST
4. Adjust hyperparameters (hidden dim, epochs, learning rate)
5. Click **▶ Train**
6. Watch live plots update in real-time!

### Language Modeling (Coming Soon)

1. Select **Language Model** tab
2. Choose EqProp variant:
   - Full EqProp Transformer
   - Attention-Only EqProp
   - Recurrent Core EqProp
   - Hybrid EqProp
   - LoopedMLP LM
3. Select dataset: tiny_shakespeare, wikitext-2, PTB
4. Click **▶ Train**
5. Generate text with temperature slider

## Architecture

```
eqprop_trainer/
├── main.py          # Entry point
├── dashboard.py     # Main window (LM + Vision tabs)
├── worker.py        # Background training thread
├── themes.py        # Dark cyberpunk QSS (250 lines)
└── __init__.py
```

## Customization

### Add Custom Theme

Edit `themes.py` and modify `CYBERPUNK_DARK` QSS string, or create your own:

```python
MY_THEME = """
QPushButton {
    background-color: #custom;
}
"""
```

### Add Custom Model

Register in the dashboard model selectors:

```python
self.vis_model_combo.addItems([
    "MyCustomModel",
])
```

## Troubleshooting

**No plots showing?**
- Install pyqtgraph: `pip install pyqtgraph`

**Training doesn't start?**
- Check you selected a dataset
- Ensure eqprop_torch library is installed

**App looks broken?**
- Make sure you have PyQt6 (not PyQt5): `pip install PyQt6`

## Requirements

- Python >= 3.9
- PyQt6 >= 6.0
- pyqtgraph >= 0.13
- eqprop-torch (from parent directory)

## Credits

Built with ❤️ for the Equilibrium Propagation research community.

Theme inspired by cyberpunk aesthetics and modern ML tools.
