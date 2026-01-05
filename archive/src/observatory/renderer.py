"""
Observatory Renderer: PyGame-based real-time visualization.

Features:
- Side-by-side layer heatmaps
- Metric overlay display
- Real-time fps counter
- Optional video recording to frames
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

# Lazy import pygame to allow running without it for tests
pygame = None

def _ensure_pygame():
    global pygame
    if pygame is None:
        import pygame as pg
        pg.init()
        pygame = pg
    return pygame


@dataclass
class RendererConfig:
    """Configuration for the observatory renderer."""
    window_width: int = 1280
    window_height: int = 720
    layer_size: int = 128  # Pixel size for each layer heatmap
    background_color: Tuple[int, int, int] = (20, 20, 30)
    text_color: Tuple[int, int, int] = (200, 200, 200)
    fps: int = 30
    show_metrics: bool = True
    record_frames: bool = False
    frame_output_dir: str = "results/observatory_frames"


class ObservatoryRenderer:
    """PyGame-based real-time renderer for network dynamics.
    
    Displays:
    - Multi-layer RGB heatmaps (side by side)
    - Legend for RGB channels
    - Live metrics (T_relax, D_nudge, etc.)
    - Lipschitz σ slider control
    """
    
    def __init__(self, config: Optional[RendererConfig] = None):
        self.config = config or RendererConfig()
        self.pg = None
        self.screen = None
        self.clock = None
        self.font = None
        self.font_large = None
        self.running = False
        self.frame_count = 0
        self.sigma_value = 1.0  # Lipschitz constraint slider
        
    def init(self):
        """Initialize PyGame window."""
        self.pg = _ensure_pygame()
        self.screen = self.pg.display.set_mode(
            (self.config.window_width, self.config.window_height)
        )
        self.pg.display.set_caption("TorEq Dynamic Observatory")
        self.clock = self.pg.time.Clock()
        self.font = self.pg.font.SysFont("monospace", 16)
        self.font_large = self.pg.font.SysFont("monospace", 24, bold=True)
        self.running = True
        
        if self.config.record_frames:
            import os
            os.makedirs(self.config.frame_output_dir, exist_ok=True)
    
    def handle_events(self) -> bool:
        """Process PyGame events. Returns False if window closed."""
        for event in self.pg.event.get():
            if event.type == self.pg.QUIT:
                self.running = False
                return False
            elif event.type == self.pg.KEYDOWN:
                if event.key == self.pg.K_ESCAPE:
                    self.running = False
                    return False
                elif event.key == self.pg.K_UP:
                    self.sigma_value = min(2.0, self.sigma_value + 0.05)
                elif event.key == self.pg.K_DOWN:
                    self.sigma_value = max(0.1, self.sigma_value - 0.05)
        return True
    
    def render_heatmap(self, rgb_array: np.ndarray, x: int, y: int, 
                       label: str = ""):
        """Render a single RGB heatmap array at position."""
        # Create surface from numpy array
        # PyGame expects [width, height, channels] but numpy gives [height, width, channels]
        # Need to transpose
        surface = self.pg.surfarray.make_surface(rgb_array.swapaxes(0, 1))
        self.screen.blit(surface, (x, y))
        
        # Draw label
        if label:
            text = self.font.render(label, True, self.config.text_color)
            self.screen.blit(text, (x, y - 20))
    
    def render_legend(self, x: int, y: int):
        """Render RGB channel legend."""
        legend_items = [
            ("Red: Activation |s|", (255, 100, 100)),
            ("Green: Velocity Δs", (100, 255, 100)),
            ("Blue: Nudge (credit)", (100, 100, 255)),
            ("White: Lipschitz violation", (255, 255, 255)),
        ]
        
        for i, (text, color) in enumerate(legend_items):
            # Color box
            self.pg.draw.rect(self.screen, color, (x, y + i * 25, 20, 16))
            # Text
            label = self.font.render(text, True, self.config.text_color)
            self.screen.blit(label, (x + 28, y + i * 25))
    
    def render_metrics(self, metrics: Dict[str, float], x: int, y: int):
        """Render metrics overlay."""
        title = self.font_large.render("Metrics", True, self.config.text_color)
        self.screen.blit(title, (x, y))
        
        y_offset = 35
        for name, value in metrics.items():
            if isinstance(value, float):
                text = f"{name}: {value:.2f}"
            else:
                text = f"{name}: {value}"
            label = self.font.render(text, True, self.config.text_color)
            self.screen.blit(label, (x, y + y_offset))
            y_offset += 22
    
    def render_sigma_slider(self, x: int, y: int, width: int = 200):
        """Render the Lipschitz σ control slider."""
        title = self.font.render(f"σ (Lipschitz): {self.sigma_value:.2f}", 
                                 True, self.config.text_color)
        self.screen.blit(title, (x, y))
        
        # Slider background
        slider_y = y + 25
        self.pg.draw.rect(self.screen, (80, 80, 80), (x, slider_y, width, 10))
        
        # Slider position (map 0.1-2.0 to 0-width)
        pos = (self.sigma_value - 0.1) / 1.9 * width
        self.pg.draw.circle(self.screen, (200, 200, 200), 
                           (int(x + pos), slider_y + 5), 8)
        
        # Instructions
        hint = self.font.render("↑/↓ to adjust", True, (120, 120, 120))
        self.screen.blit(hint, (x, slider_y + 20))
    
    def render_frame(self, 
                     layer_heatmaps: Dict[str, np.ndarray],
                     metrics: Optional[Dict[str, float]] = None,
                     epoch: int = 0,
                     step: int = 0):
        """Render a complete frame with all visualizations.
        
        Args:
            layer_heatmaps: Dict mapping layer names to RGB arrays [H, W, 3]
            metrics: Optional metrics dict to display
            epoch: Current training epoch
            step: Current step within epoch
        """
        if not self.handle_events():
            return
        
        # Clear screen
        self.screen.fill(self.config.background_color)
        
        # Title
        title = self.font_large.render(
            f"TorEq Dynamic Observatory - Epoch {epoch}, Step {step}", 
            True, self.config.text_color
        )
        self.screen.blit(title, (20, 10))
        
        # Render layer heatmaps in a row
        x_offset = 20
        y_offset = 60
        for name in sorted(layer_heatmaps.keys()):
            heatmap = layer_heatmaps[name]
            # Resize if needed
            if heatmap.shape[0] != self.config.layer_size:
                import cv2
                heatmap = cv2.resize(heatmap, 
                    (self.config.layer_size, self.config.layer_size),
                    interpolation=cv2.INTER_NEAREST)
            
            self.render_heatmap(heatmap, x_offset, y_offset, label=name)
            x_offset += self.config.layer_size + 20
        
        # Legend (right side)
        legend_x = self.config.window_width - 280
        self.render_legend(legend_x, 60)
        
        # Metrics (below legend)
        if metrics and self.config.show_metrics:
            self.render_metrics(metrics, legend_x, 200)
        
        # Sigma slider (bottom right)
        self.render_sigma_slider(legend_x, self.config.window_height - 100)
        
        # FPS counter
        fps = int(self.clock.get_fps())
        fps_text = self.font.render(f"FPS: {fps}", True, (100, 100, 100))
        self.screen.blit(fps_text, (self.config.window_width - 80, 10))
        
        # Update display
        self.pg.display.flip()
        self.clock.tick(self.config.fps)
        
        # Save frame if recording
        if self.config.record_frames:
            filename = f"{self.config.frame_output_dir}/frame_{self.frame_count:05d}.png"
            self.pg.image.save(self.screen, filename)
        
        self.frame_count += 1
    
    def close(self):
        """Close the PyGame window."""
        if self.pg:
            self.pg.quit()
        self.running = False


class HeadlessRenderer:
    """Headless renderer for testing without display.
    
    Records frames to numpy arrays instead of displaying.
    """
    
    def __init__(self, config: Optional[RendererConfig] = None):
        self.config = config or RendererConfig()
        self.frames: List[np.ndarray] = []
        self.running = True
        self.sigma_value = 1.0
    
    def init(self):
        pass
    
    def handle_events(self) -> bool:
        return self.running
    
    def render_frame(self, 
                     layer_heatmaps: Dict[str, np.ndarray],
                     metrics: Optional[Dict[str, float]] = None,
                     epoch: int = 0,
                     step: int = 0):
        """Record frame to internal buffer."""
        # Just concatenate heatmaps horizontally
        if layer_heatmaps:
            combined = np.concatenate(
                [layer_heatmaps[k] for k in sorted(layer_heatmaps.keys())],
                axis=1
            )
            self.frames.append(combined)
    
    def close(self):
        self.running = False
    
    def save_gif(self, filename: str = "results/observatory.gif", fps: int = 10):
        """Save recorded frames as GIF."""
        if not self.frames:
            return
        
        try:
            from PIL import Image
            images = [Image.fromarray(f) for f in self.frames]
            images[0].save(
                filename,
                save_all=True,
                append_images=images[1:],
                duration=1000 // fps,
                loop=0
            )
            print(f"Saved {len(self.frames)} frames to {filename}")
        except ImportError:
            print("PIL not available, cannot save GIF")
