
import time
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Callable

from .notebook import VerificationNotebook, TrackResult
from .tracks import core_tracks, advanced_tracks, scaling_tracks, special_tracks, hardware_tracks, analysis_tracks, application_tracks, engine_validation_tracks

class Verifier:
    """Complete verification suite for all research tracks."""
    
    def __init__(self, quick_mode: bool = False, seed: int = 42, n_seeds_override: Optional[int] = None, export_data: bool = False):
        self.quick_mode = quick_mode
        self.seed = seed
        self.export_data = export_data
        self.notebook = VerificationNotebook()
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.epochs = 5 if quick_mode else 15 # Optimized for ~10min run
        self.n_samples = 200 if quick_mode else 400 # Sufficient for stats
        
        # Redundancy Configuration
        if quick_mode:
            self.n_seeds = 1
        elif n_seeds_override is not None:
            self.n_seeds = n_seeds_override
        else:
            self.n_seeds = 3 # Default standard redundancy
            
        self.data_records = [] # For CSV export
        self.current_seed = seed # Track current seed for logging
        
        # Track definitions
        self.tracks = {
            1: ("Spectral Normalization Stability", core_tracks.track_1_spectral_norm),
            2: ("EqProp vs Backprop Parity", core_tracks.track_2_backprop_parity),
            3: ("Adversarial Self-Healing", core_tracks.track_3_adversarial_healing),
            4: ("Ternary Weights", advanced_tracks.track_4_ternary_weights),
            5: ("Neural Cube 3D Topology", scaling_tracks.track_5_neural_cube),
            6: ("Feedback Alignment", advanced_tracks.track_6_feedback_alignment),
            7: ("Temporal Resonance", advanced_tracks.track_7_temporal_resonance),
            8: ("Homeostatic Stability", advanced_tracks.track_8_homeostatic),
            9: ("Gradient Alignment", advanced_tracks.track_9_gradient_alignment),
            10: ("O(1) Memory Scaling", scaling_tracks.track_10_memory_scaling),
            11: ("Deep Network (100 layers)", scaling_tracks.track_11_deep_network),
            12: ("Lazy Event-Driven Updates", scaling_tracks.track_12_lazy_updates),
            13: ("Convolutional EqProp", special_tracks.track_13_conv_eqprop),
            14: ("Transformer EqProp", special_tracks.track_14_transformer),
            15: ("PyTorch vs Kernel", special_tracks.track_15_kernel_comparison),
            16: ("FPGA Bit Precision", hardware_tracks.track_16_fpga_quantization),
            17: ("Analog/Photonics Noise", hardware_tracks.track_17_analog_photonics),
            18: ("DNA/Thermodynamic", hardware_tracks.track_18_thermodynamic_dna),
            19: ("Criticality Analysis", analysis_tracks.track_19_criticality),
            20: ("Transfer Learning", application_tracks.track_20_transfer_learning),
            21: ("Continual Learning", application_tracks.track_21_continual_learning),
            22: ("Golden Reference Harness", engine_validation_tracks.track_22_golden_reference),
            23: ("Extreme Depth Signal Probe", engine_validation_tracks.track_23_extreme_depth_signal),
            24: ("Lazy Updates Wall-Clock", engine_validation_tracks.track_24_lazy_wallclock),
        }
    
    def print_header(self):
        print("=" * 70)
        print("       TOREQPROP COMPREHENSIVE VERIFICATION SUITE")
        print("       Undeniable Evidence for All Research Claims")
        print("=" * 70)
        print(f"\n📋 Configuration:")
        print(f"   Seed: {self.seed}")
        print(f"   Mode: {'⚡ Quick' if self.quick_mode else '🔬 Full'}")
        print(f"   Epochs: {self.epochs}")
        print(f"   Samples: {self.n_samples}")
        print(f"   Seeds: {self.n_seeds} {'(Override)' if self.n_seeds != 3 and not self.quick_mode else ''}")
        print(f"   Tracks: {len(self.tracks)}")
        if self.export_data:
            print(f"   Export: Enabled (results/data.csv)")
        print("=" * 70)

    def record_metric(self, track_id: int, seed: int, step: int, metric_name: str, value: float):
        """Record a data point for export."""
        if self.export_data:
            self.data_records.append({
                "track_id": track_id,
                "seed": seed,
                "step": step,
                "metric": metric_name,
                "value": value,
                "timestamp": datetime.now().isoformat()
            })

    def evaluate_robustness(self, track_fn, n_seeds: int = 3) -> Dict:
        """Run a track logic multiple times with different seeds."""
        scores = []
        metrics_list = []
        
        # Determine number of seeds to run
        # override rules:
        # 1. if quick_mode -> 1
        # 2. if --seeds X provided -> X
        # 3. if default (3) -> use track-specific n_seeds (arg)
        
        run_count = self.n_seeds
        if self.n_seeds == 3 and not self.quick_mode:
             run_count = n_seeds
        
        print(f"      Running robustness check ({run_count} seeds)...")
        
        for i in range(run_count):
            seed = self.seed + i*100
            self.current_seed = seed # Update state for loggers
            
            # Temporarily set seed
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            try:
                score, metrics = track_fn()
                scores.append(score)
                metrics_list.append(metrics)
                
                # Record aggregations for export
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        self.record_metric(0, seed, 0, k, v) # Track ID 0 is generic/unknown here
                        
            except Exception as e:
                print(f"        Seed {seed}: Failed ({e})")
                import traceback
                traceback.print_exc()
                scores.append(0)
                metrics_list.append({})
                
        mean_score = np.mean(scores)
        std_score = np.std(scores) if len(scores) > 1 else 0.0
        
        # Aggregate metrics
        agg_metrics = {}
        if metrics_list:
            keys = metrics_list[0].keys()
            for k in keys:
                vals = [m[k] for m in metrics_list if k in m and isinstance(m[k], (int, float))]
                if vals:
                    agg_metrics[f"{k}_mean"] = np.mean(vals)
                    agg_metrics[f"{k}_std"] = np.std(vals) if len(vals) > 1 else 0.0
                    
        return {
            "mean_score": mean_score,
            "std_score": std_score,
            "metrics": agg_metrics,
            "all_scores": scores
        }
    
    def run_tracks(self, track_ids: Optional[List[int]] = None) -> Dict:
        """Run specified tracks (or all if None)."""
        self.print_header()
        self.notebook.add_header(self.seed)
        
        if track_ids is None:
            track_ids = list(self.tracks.keys())
        
        results = {}
        start_time = time.time()
        
        for i, track_id in enumerate(track_ids):
            if track_id not in self.tracks:
                print(f"⚠️ Unknown track: {track_id}")
                continue
            
            name, method = self.tracks[track_id]
            
            try:
                # Pass self (Verifier) to the track method
                result = method(self)
                results[track_id] = result
                self.notebook.add_track_result(result)
                
                icon = {"pass": "✅", "fail": "❌", "partial": "⚠️", "stub": "🔧"}[result.status]
                print(f"\n{icon} Track {track_id}: {name} - {result.status.upper()} ({result.score:.0f}/100)")
                
            except Exception as e:
                print(f"\n❌ Track {track_id} failed: {e}")
                import traceback
                traceback.print_exc()
            
            # Progress
            elapsed = time.time() - start_time
            completed = i + 1
            remaining = len(track_ids) - completed
            if remaining > 0:
                eta = (elapsed / completed) * remaining
                print(f"   Progress: {completed}/{len(track_ids)} | Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s")
        
        # Save
        total_time = time.time() - start_time
        # output path relative to original verify.py location
        output_path = Path(__file__).parent.parent / "results" / "verification_notebook.md"
        self.notebook.save(output_path)
        
        # Summary
        print("\n" + "=" * 70)
        print("🎉 VERIFICATION COMPLETE")
        print("=" * 70)
        print(f"⏱️  Total time: {total_time:.1f}s")
        print(f"📓 Output: {output_path}")
        
        passed = sum(1 for r in results.values() if r.status == "pass")
        total = len(results)
        print(f"\n📊 Results: {passed}/{total} tracks passed")
        
        if self.export_data and self.data_records:
            import csv
            csv_path = Path(__file__).parent.parent / "results" / "data.csv"
            keys = self.data_records[0].keys()
            with open(csv_path, 'w', newline='') as f:
                dict_writer = csv.DictWriter(f, keys)
                dict_writer.writeheader()
                dict_writer.writerows(self.data_records)
            print(f"💾 Data exported to: {csv_path}")
        
        return results

    def list_tracks(self):
        """Print all available tracks."""
        print("\nAvailable Verification Tracks:")
        print("-" * 60)
        for tid, (name, _) in self.tracks.items():
            print(f"  {tid:2d}. {name}")
        print("-" * 60)
