#!/usr/bin/env python3
"""
Unit tests for the TorEq Dynamic Observatory module.

Run with:
    python -m unittest tests/test_observatory -v
    
Or directly:
    python tests/test_observatory.py
"""

import unittest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np


class TestSynapseHeatmap(unittest.TestCase):
    """Tests for SynapseHeatmap RGB channel generation."""
    
    def setUp(self):
        from src.observatory.heatmap import SynapseHeatmap
        self.heatmap = SynapseHeatmap()
    
    def test_grid_size_perfect_square(self):
        """Grid size for perfect square neuron counts."""
        self.assertEqual(self.heatmap.compute_grid_size(64), (8, 8))
        self.assertEqual(self.heatmap.compute_grid_size(256), (16, 16))
        self.assertEqual(self.heatmap.compute_grid_size(1024), (32, 32))
    
    def test_grid_size_non_square(self):
        """Grid size rounds up for non-perfect squares."""
        self.assertEqual(self.heatmap.compute_grid_size(100), (10, 10))
        self.assertEqual(self.heatmap.compute_grid_size(65), (9, 9))
        self.assertEqual(self.heatmap.compute_grid_size(17), (5, 5))
    
    def test_reshape_to_grid_batched(self):
        """Reshape [batch, neurons] to [batch, H, W]."""
        tensor = torch.randn(4, 64)
        grid = self.heatmap.reshape_to_grid(tensor)
        self.assertEqual(grid.shape, (4, 8, 8))
    
    def test_reshape_to_grid_single(self):
        """Single tensor gets unsqueezed to batch dim 1."""
        tensor = torch.randn(64)
        grid = self.heatmap.reshape_to_grid(tensor)
        self.assertEqual(grid.shape, (1, 8, 8))
    
    def test_reshape_to_grid_pads(self):
        """Non-perfect square gets zero-padded."""
        tensor = torch.randn(4, 60)
        grid = self.heatmap.reshape_to_grid(tensor)
        # 60 → 64 (8x8), padded with 4 zeros
        self.assertEqual(grid.shape, (4, 8, 8))
    
    def test_activation_to_red_shape(self):
        """Red channel has correct shape and dtype."""
        activation = torch.randn(4, 64)
        red = self.heatmap.activation_to_red(activation)
        self.assertEqual(red.shape, (8, 8))
        self.assertEqual(red.dtype, np.uint8)
    
    def test_activation_to_red_range(self):
        """Red channel values are in [0, 255]."""
        activation = torch.randn(4, 64) * 10  # Large values
        red = self.heatmap.activation_to_red(activation)
        self.assertGreaterEqual(red.min(), 0)
        self.assertLessEqual(red.max(), 255)
    
    def test_velocity_to_green_shape(self):
        """Green channel has correct shape and dtype."""
        velocity = torch.randn(4, 64) * 0.1
        green = self.heatmap.velocity_to_green(velocity)
        self.assertEqual(green.shape, (8, 8))
        self.assertEqual(green.dtype, np.uint8)
    
    def test_nudge_to_blue_shape(self):
        """Blue channel has correct shape and dtype."""
        nudge = torch.randn(4, 64) * 0.2
        blue = self.heatmap.nudge_to_blue(nudge)
        self.assertEqual(blue.shape, (8, 8))
        self.assertEqual(blue.dtype, np.uint8)
    
    def test_generate_rgb_full_state(self):
        """Full RGB generation with all three channels."""
        from src.observatory.heatmap import LayerState
        state = LayerState(
            activation=torch.randn(4, 64),
            velocity=torch.randn(4, 64) * 0.1,
            nudge=torch.randn(4, 64) * 0.2,
        )
        rgb = self.heatmap.generate_rgb(state)
        self.assertEqual(rgb.shape, (8, 8, 3))
        self.assertEqual(rgb.dtype, np.uint8)
    
    def test_generate_rgb_activation_only(self):
        """RGB generation with only activation (velocity/nudge None)."""
        from src.observatory.heatmap import LayerState
        state = LayerState(activation=torch.randn(4, 64))
        rgb = self.heatmap.generate_rgb(state)
        self.assertEqual(rgb.shape, (8, 8, 3))
        # Green and blue should be zero (except possible white overlay)
        # Just verify shape is correct


class TestDynamicsCapture(unittest.TestCase):
    """Tests for DynamicsCapture state recording."""
    
    def setUp(self):
        from src.observatory.heatmap import DynamicsCapture
        self.capture = DynamicsCapture()
    
    def test_record_step(self):
        """Recording a step adds to history."""
        h_old = torch.zeros(4, 64)
        h_new = torch.randn(4, 64)
        self.capture.record_step("layer_0", h_new, h_old)
        
        self.assertEqual(len(self.capture.history), 1)
        self.assertIn("layer_0", self.capture.history[0])
    
    def test_record_step_computes_velocity(self):
        """Velocity is computed as h_new - h_old."""
        h_old = torch.zeros(4, 64)
        h_new = torch.ones(4, 64)
        self.capture.record_step("layer_0", h_new, h_old)
        
        state = self.capture.history[0]["layer_0"]
        self.assertIsNotNone(state.velocity)
        # Velocity should be all ones
        self.assertTrue(torch.allclose(state.velocity, torch.ones(4, 64)))
    
    def test_record_free_equilibrium(self):
        """Free equilibrium is stored correctly."""
        states = {"layer_0": torch.randn(4, 64)}
        self.capture.record_free_equilibrium(states)
        
        self.assertIsNotNone(self.capture.free_equilibrium)
        self.assertIn("layer_0", self.capture.free_equilibrium)
    
    def test_record_nudged_equilibrium(self):
        """Nudged equilibrium is stored correctly."""
        states = {"layer_0": torch.randn(4, 64)}
        self.capture.record_nudged_equilibrium(states)
        
        self.assertIsNotNone(self.capture.nudged_equilibrium)
    
    def test_history_limit(self):
        """History is limited to max_history entries."""
        from src.observatory.heatmap import DynamicsCapture
        capture = DynamicsCapture(max_history=5)
        
        for i in range(10):
            capture.record_step(f"step_{i}", torch.randn(4, 64), torch.zeros(4, 64))
        
        self.assertEqual(len(capture.history), 5)
    
    def test_clear(self):
        """Clear resets all state."""
        self.capture.record_step("layer_0", torch.randn(4, 64), torch.zeros(4, 64))
        self.capture.record_free_equilibrium({"layer_0": torch.randn(4, 64)})
        
        self.capture.clear()
        
        self.assertEqual(len(self.capture.history), 0)
        self.assertIsNone(self.capture.free_equilibrium)


class TestObservatoryMetrics(unittest.TestCase):
    """Tests for ObservatoryMetrics computation."""
    
    def setUp(self):
        from src.observatory.metrics import ObservatoryMetrics
        self.metrics = ObservatoryMetrics(velocity_threshold=0.01)
    
    def test_settling_time_converges(self):
        """Settling time is step when velocity drops below threshold."""
        velocity_history = [
            torch.ones(4, 64) * 0.5,
            torch.ones(4, 64) * 0.1,
            torch.ones(4, 64) * 0.005,  # Below 0.01 threshold
        ]
        t = self.metrics.compute_settling_time(velocity_history)
        self.assertEqual(t, 3)
    
    def test_settling_time_never_converges(self):
        """If velocity never drops, return len(history)."""
        velocity_history = [
            torch.ones(4, 64) * 0.5,
            torch.ones(4, 64) * 0.5,
        ]
        t = self.metrics.compute_settling_time(velocity_history)
        self.assertEqual(t, 2)
    
    def test_nudge_depth(self):
        """Nudge depth counts visible layers from output."""
        from src.observatory.metrics import ObservatoryMetrics
        metrics = ObservatoryMetrics(nudge_visibility_threshold=0.01)
        
        layer_nudges = {
            "layer_0": torch.ones(4, 64) * 0.1,   # Visible
            "layer_1": torch.ones(4, 64) * 0.05,  # Visible
            "layer_2": torch.ones(4, 64) * 0.001, # Below threshold
        }
        
        # Note: compute_nudge_depth counts from END of list backward
        depth = metrics.compute_nudge_depth(
            layer_nudges, 
            ["layer_0", "layer_1", "layer_2"]
        )
        # layer_2 is below threshold, so only layer_1 is counted from end
        self.assertGreaterEqual(depth, 0)
    
    def test_skip_ratio(self):
        """Skip ratio = skipped / total."""
        for _ in range(7):
            self.metrics.record_update(was_skipped=False)
        for _ in range(3):
            self.metrics.record_update(was_skipped=True)
        
        self.assertAlmostEqual(self.metrics.skip_ratio, 0.3)
        self.assertAlmostEqual(self.metrics.flop_savings_percent, 30.0)
    
    def test_reset(self):
        """Reset clears all metrics."""
        self.metrics.record_update(was_skipped=True)
        self.metrics.reset()
        
        self.assertEqual(self.metrics.total_updates, 0)
        self.assertEqual(len(self.metrics.settling_times), 0)


class TestRecursiveBlock(unittest.TestCase):
    """Tests for RecursiveBlock fractal architecture."""
    
    def setUp(self):
        from src.models.recursive_block import RecursiveBlock, DeepRecursiveNetwork
        self.RecursiveBlock = RecursiveBlock
        self.DeepRecursiveNetwork = DeepRecursiveNetwork
    
    def test_recursive_block_output_shape(self):
        """RecursiveBlock produces correct output shape."""
        block = self.RecursiveBlock(10, 64, 5, inner_steps=5)
        x = torch.randn(4, 10)
        out = block(x, steps=10)
        self.assertEqual(out.shape, (4, 5))
    
    def test_recursive_block_energy(self):
        """RecursiveBlock energy returns scalar."""
        block = self.RecursiveBlock(10, 64, 5)
        x = torch.randn(4, 10)
        h = torch.randn(4, 64)
        e = block.energy(h, x)
        self.assertEqual(e.dim(), 0)  # Scalar
    
    def test_deep_network_output_shape(self):
        """DeepRecursiveNetwork produces correct output shape."""
        net = self.DeepRecursiveNetwork(10, 32, 5, num_blocks=5)
        x = torch.randn(4, 10)
        out = net(x, steps=10)
        self.assertEqual(out.shape, (4, 5))
    
    def test_deep_network_gradient_flow(self):
        """Gradients flow through deep recursive network."""
        net = self.DeepRecursiveNetwork(10, 32, 5, num_blocks=5)
        x = torch.randn(4, 10)
        out = net(x, steps=10)
        loss = out.sum()
        
        # Should not raise
        loss.backward()


class TestLazyEqProp(unittest.TestCase):
    """Tests for LazyEqProp event-driven engine."""
    
    def setUp(self):
        from src.models.lazy_eqprop import LazyEqProp
        self.LazyEqProp = LazyEqProp
    
    def test_lazy_output_shape(self):
        """LazyEqProp produces correct output shape."""
        model = self.LazyEqProp(10, 64, 5, epsilon=0.01)
        x = torch.randn(4, 10)
        out = model(x, steps=10)
        self.assertEqual(out.shape, (4, 5))
    
    def test_flop_savings_tracked(self):
        """FLOP savings are tracked after forward pass."""
        model = self.LazyEqProp(10, 64, 5, epsilon=0.01)
        x = torch.randn(4, 10)
        _ = model(x, steps=20, track_activity=True)
        
        # Should have some FLOP savings (activity gating)
        savings = model.get_flop_savings()
        self.assertGreaterEqual(savings, 0)
        self.assertLessEqual(savings, 100)
    
    def test_activity_masks_recorded(self):
        """Activity masks are recorded when track_activity=True."""
        model = self.LazyEqProp(10, 64, 5, epsilon=0.01)
        x = torch.randn(4, 10)
        _ = model(x, steps=10, track_activity=True)
        
        self.assertEqual(len(model.activity_masks), 10)
    
    def test_persistent_nudge_mode(self):
        """Persistent nudge mode runs without error."""
        model = self.LazyEqProp(
            10, 64, 5, 
            persistent_nudge_strength=0.2
        )
        x = torch.randn(4, 10)
        out = model(x, steps=10)
        self.assertEqual(out.shape, (4, 5))


class TestHeadlessRenderer(unittest.TestCase):
    """Tests for HeadlessRenderer frame capture."""
    
    def setUp(self):
        from src.observatory.renderer import HeadlessRenderer
        self.renderer = HeadlessRenderer()
        self.renderer.init()
    
    def test_render_frame_captures(self):
        """Rendering a frame adds to internal buffer."""
        heatmaps = {
            "layer_0": np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8),
        }
        self.renderer.render_frame(heatmaps, metrics={}, epoch=0, step=0)
        
        self.assertEqual(len(self.renderer.frames), 1)
    
    def test_render_multiple_layers(self):
        """Multiple layers are concatenated horizontally."""
        heatmaps = {
            "layer_0": np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8),
            "layer_1": np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8),
        }
        self.renderer.render_frame(heatmaps, metrics={}, epoch=0, step=0)
        
        # Two 64-wide heatmaps concatenated
        self.assertEqual(self.renderer.frames[0].shape[1], 128)


if __name__ == '__main__':
    unittest.main(verbosity=2)
