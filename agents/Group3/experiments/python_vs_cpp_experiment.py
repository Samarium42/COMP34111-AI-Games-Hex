"""
Python vs C++ MCTS Performance Comparison

This experiment compares the performance of:
1. Pure Python MCTS implementation
2. C++ MCTS implementation (via ctypes)

Tests speed, memory usage, and playing strength.
"""

import sys
import time
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.Game import Game
from src.Player import Player
from src.Colour import Colour
from agents.Group3.CPPGraveNN import CPPGraveNN
from agents.Group3.PythonMCTSAgent import PythonMCTSAgent


class PythonVsCppExperiment:
    """Compare Python and C++ MCTS implementations."""
    
    def __init__(self, output_dir="python_vs_cpp_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
    
    def run_single_game(self, agent1, agent2, name1, name2):
        """Run a single game and collect timing data."""
        player1 = Player(name1, agent1)
        player2 = Player(name2, agent2)
        
        start_time = time.time()
        game = Game(player1, player2, board_size=11, silent=True)
        result = game.run()
        total_time = time.time() - start_time
        
        return {
            "winner": result["winner"],
            "player1": name1,
            "player2": name2,
            "player1_time": float(result["player1_move_time"]),
            "player2_time": float(result["player2_move_time"]),
            "player1_turns": int(result["player1_turns"]),
            "player2_turns": int(result["player2_turns"]),
            "total_time": total_time,
        }
    
    def experiment_speed_comparison(self, sim_counts=[100, 300, 500], num_games=10):
        """
        Test 1: Pure speed comparison - Python vs C++ at various sim counts.
        Measures time per move, not win rate (both use same algorithm).
        """
        print(f"\n{'='*70}")
        print("EXPERIMENT 1: Speed Comparison (Python vs C++)")
        print(f"{'='*70}")
        print("Measuring raw computational speed at different simulation counts")
        print("Note: Both implementations use identical algorithms")
        print(f"{'='*70}\n")
        
        speed_data = []
        
        for sims in sim_counts:
            print(f"\n[Testing {sims} simulations]")
            print(f"{'─'*50}")
            
            python_times = []
            cpp_times = []
            python_wins = 0
            
            for game_idx in range(num_games):
                print(f"  Game {game_idx+1}/{num_games}...", end=" ")
                
                if game_idx % 2 == 0:
                    # Python plays RED
                    python_agent = PythonMCTSAgent(
                        Colour.RED, sims=sims, use_grave=False
                    )
                    cpp_agent = CPPGraveNN(
                        Colour.BLUE, sims=sims, use_grave=False
                    )
                    result = self.run_single_game(
                        python_agent, cpp_agent, "Python", "C++"
                    )
                    python_times.append(result["player1_time"])
                    cpp_times.append(result["player2_time"])
                else:
                    # C++ plays RED
                    cpp_agent = CPPGraveNN(
                        Colour.RED, sims=sims, use_grave=False
                    )
                    python_agent = PythonMCTSAgent(
                        Colour.BLUE, sims=sims, use_grave=False
                    )
                    result = self.run_single_game(
                        cpp_agent, python_agent, "C++", "Python"
                    )
                    cpp_times.append(result["player1_time"])
                    python_times.append(result["player2_time"])
                
                if result["winner"] == "Python":
                    python_wins += 1
                
                print(f"{result['winner']} wins")
            
            py_avg = np.mean(python_times)
            py_std = np.std(python_times)
            cpp_avg = np.mean(cpp_times)
            cpp_std = np.std(cpp_times)
            speedup = py_avg / cpp_avg
            
            speed_data.append({
                "sims": sims,
                "python_avg_time": py_avg,
                "python_std_time": py_std,
                "cpp_avg_time": cpp_avg,
                "cpp_std_time": cpp_std,
                "speedup": speedup,
                "python_wins": python_wins,
            })
            
            print(f"\nResults for {sims} simulations:")
            print(f"  Python: {py_avg:.3f}s ± {py_std:.3f}s per game")
            print(f"  C++:    {cpp_avg:.3f}s ± {cpp_std:.3f}s per game")
            print(f"  Speedup: {speedup:.2f}x (C++ is {speedup:.2f}x faster)")
            print(f"  Python win rate: {python_wins/num_games*100:.1f}%")
        
        self.results["speed_comparison"] = speed_data
        return speed_data
    
    def experiment_equivalent_strength(self, base_sims=500, num_games=10):
        """
        Test 2: Find Python sim count that matches C++ speed.
        If C++ is 10x faster, can Python-50 match C++-500 in time?
        """
        print(f"\n{'='*70}")
        print("EXPERIMENT 2: Speed-Equivalent Comparison")
        print(f"{'='*70}")
        print("Find Python sim count that takes same time as C++-500")
        print(f"{'='*70}\n")
        
        # First, measure C++ time at base_sims
        print(f"Step 1: Measure C++-{base_sims} average time...")
        cpp_times = []
        
        for i in range(5):
            cpp_agent = CPPGraveNN(Colour.RED, sims=base_sims, use_grave=False)
            dummy_agent = CPPGraveNN(Colour.BLUE, sims=100, use_grave=False)
            result = self.run_single_game(cpp_agent, dummy_agent, "C++", "Dummy")
            cpp_times.append(result["player1_time"])
        
        cpp_avg_time = np.mean(cpp_times)
        print(f"  C++-{base_sims} average time: {cpp_avg_time:.3f}s")
        
        # Estimate Python sim count (rough: Python is ~10-30x slower)
        estimated_ratio = 15  # Conservative estimate
        python_sims = max(50, base_sims // estimated_ratio)
        
        print(f"\nStep 2: Test Python-{python_sims} (estimated equivalent)...")
        
        python_wins = 0
        results = []
        
        for game_idx in range(num_games):
            print(f"  Game {game_idx+1}/{num_games}...", end=" ")
            
            if game_idx % 2 == 0:
                python_agent = PythonMCTSAgent(
                    Colour.RED, sims=python_sims, use_grave=False
                )
                cpp_agent = CPPGraveNN(
                    Colour.BLUE, sims=base_sims, use_grave=False
                )
                result = self.run_single_game(
                    python_agent, cpp_agent, 
                    f"Python-{python_sims}", f"C++-{base_sims}"
                )
            else:
                cpp_agent = CPPGraveNN(
                    Colour.RED, sims=base_sims, use_grave=False
                )
                python_agent = PythonMCTSAgent(
                    Colour.BLUE, sims=python_sims, use_grave=False
                )
                result = self.run_single_game(
                    cpp_agent, python_agent,
                    f"C++-{base_sims}", f"Python-{python_sims}"
                )
            
            if f"Python-{python_sims}" in result["winner"]:
                python_wins += 1
            
            results.append(result)
            print(f"{result['winner']} wins")
        
        win_rate = python_wins / num_games
        
        print(f"\n{'─'*70}")
        print("Results:")
        print(f"  C++-{base_sims} time: {cpp_avg_time:.3f}s/game")
        print(f"  Python-{python_sims} vs C++-{base_sims}")
        print(f"  Python win rate: {win_rate*100:.1f}%")
        print(f"  Interpretation: Python with {python_sims/base_sims*100:.1f}% of sims")
        print(f"{'─'*70}")
        
        self.results["equivalent_strength"] = {
            "cpp_sims": base_sims,
            "python_sims": python_sims,
            "cpp_avg_time": cpp_avg_time,
            "python_win_rate": win_rate,
            "results": results,
        }
    
    def experiment_grave_overhead(self, sims=300, num_games=8):
        """
        Test 3: Compare GRAVE overhead in Python vs C++.
        """
        print(f"\n{'='*70}")
        print("EXPERIMENT 3: GRAVE Overhead Comparison")
        print(f"{'='*70}")
        print("Measure additional cost of GRAVE in Python vs C++")
        print(f"{'='*70}\n")
        
        results = {}
        
        for impl in ["Python", "C++"]:
            print(f"\n[Testing {impl} implementation]")
            print(f"{'─'*50}")
            
            for use_grave in [False, True]:
                mode = "GRAVE" if use_grave else "Plain"
                print(f"  {mode} mode...", end=" ")
                
                times = []
                
                for i in range(num_games):
                    if impl == "Python":
                        agent = PythonMCTSAgent(
                            Colour.RED, sims=sims, use_grave=use_grave
                        )
                        dummy = PythonMCTSAgent(
                            Colour.BLUE, sims=100, use_grave=False
                        )
                    else:
                        agent = CPPGraveNN(
                            Colour.RED, sims=sims, use_grave=use_grave
                        )
                        dummy = CPPGraveNN(
                            Colour.BLUE, sims=100, use_grave=False
                        )
                    
                    result = self.run_single_game(agent, dummy, "Test", "Dummy")
                    times.append(result["player1_time"])
                
                avg_time = np.mean(times)
                print(f"{avg_time:.3f}s")
                
                key = f"{impl}_{mode}"
                results[key] = {
                    "avg_time": avg_time,
                    "times": times,
                }
            
            # Compute overhead
            plain_time = results[f"{impl}_Plain"]["avg_time"]
            grave_time = results[f"{impl}_GRAVE"]["avg_time"]
            overhead = (grave_time - plain_time) / plain_time * 100
            
            print(f"  GRAVE overhead: {overhead:+.1f}%")
        
        self.results["grave_overhead"] = results
        
        print(f"\n{'─'*70}")
        print("Summary:")
        py_overhead = (results["Python_GRAVE"]["avg_time"] - 
                      results["Python_Plain"]["avg_time"]) / results["Python_Plain"]["avg_time"] * 100
        cpp_overhead = (results["C++_GRAVE"]["avg_time"] - 
                       results["C++_Plain"]["avg_time"]) / results["C++_Plain"]["avg_time"] * 100
        print(f"  Python GRAVE overhead: {py_overhead:+.1f}%")
        print(f"  C++ GRAVE overhead: {cpp_overhead:+.1f}%")
        print(f"{'─'*70}")
    
    def save_results(self):
        """Save results to JSON."""
        output_file = self.output_dir / "python_vs_cpp_results.json"
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Results saved to {output_file}")
    
    def plot_results(self):
        """Generate comparison plots."""
        print(f"\n{'='*70}")
        print("Generating Plots...")
        print(f"{'='*70}\n")
        
        fig = plt.figure(figsize=(16, 10))
        
        # Plot 1: Speed comparison
        if "speed_comparison" in self.results:
            ax1 = plt.subplot(2, 3, 1)
            data = self.results["speed_comparison"]
            
            sims = [d["sims"] for d in data]
            py_times = [d["python_avg_time"] for d in data]
            cpp_times = [d["cpp_avg_time"] for d in data]
            
            x = np.arange(len(sims))
            width = 0.35
            
            ax1.bar(x - width/2, py_times, width, label='Python', 
                   color='#3498db', alpha=0.7)
            ax1.bar(x + width/2, cpp_times, width, label='C++', 
                   color='#e74c3c', alpha=0.7)
            
            ax1.set_xlabel('Simulations', fontweight='bold')
            ax1.set_ylabel('Time per Game (seconds)', fontweight='bold')
            ax1.set_title('Speed Comparison', fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(sims)
            ax1.legend()
            ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Speedup factor
        if "speed_comparison" in self.results:
            ax2 = plt.subplot(2, 3, 2)
            data = self.results["speed_comparison"]
            
            sims = [d["sims"] for d in data]
            speedups = [d["speedup"] for d in data]
            
            ax2.plot(sims, speedups, 'o-', linewidth=2, markersize=10,
                    color='#2ecc71')
            ax2.set_xlabel('Simulations', fontweight='bold')
            ax2.set_ylabel('Speedup Factor (x)', fontweight='bold')
            ax2.set_title('C++ Speedup Over Python', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # Add speedup labels
            for s, sp in zip(sims, speedups):
                ax2.text(s, sp + 0.5, f'{sp:.1f}x', 
                        ha='center', fontweight='bold')
        
        # Plot 3: Time per simulation
        if "speed_comparison" in self.results:
            ax3 = plt.subplot(2, 3, 3)
            data = self.results["speed_comparison"]
            
            sims = [d["sims"] for d in data]
            py_per_sim = [d["python_avg_time"]/d["sims"]*1000 for d in data]
            cpp_per_sim = [d["cpp_avg_time"]/d["sims"]*1000 for d in data]
            
            ax3.plot(sims, py_per_sim, 'o-', linewidth=2, label='Python',
                    color='#3498db', markersize=8)
            ax3.plot(sims, cpp_per_sim, 'o-', linewidth=2, label='C++',
                    color='#e74c3c', markersize=8)
            
            ax3.set_xlabel('Simulations', fontweight='bold')
            ax3.set_ylabel('Time per Simulation (ms)', fontweight='bold')
            ax3.set_title('Per-Simulation Cost', fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: GRAVE overhead comparison
        if "grave_overhead" in self.results:
            ax4 = plt.subplot(2, 3, 4)
            data = self.results["grave_overhead"]
            
            categories = ['Python\nPlain', 'Python\nGRAVE', 'C++\nPlain', 'C++\nGRAVE']
            times = [
                data["Python_Plain"]["avg_time"],
                data["Python_GRAVE"]["avg_time"],
                data["C++_Plain"]["avg_time"],
                data["C++_GRAVE"]["avg_time"],
            ]
            colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
            
            bars = ax4.bar(categories, times, color=colors, alpha=0.7)
            ax4.set_ylabel('Time per Game (seconds)', fontweight='bold')
            ax4.set_title('GRAVE Overhead', fontweight='bold')
            
            for bar, val in zip(bars, times):
                ax4.text(bar.get_x() + bar.get_width()/2., val,
                        f'{val:.2f}s', ha='center', va='bottom', fontweight='bold')
        
        # Plot 5: Relative overhead
        if "grave_overhead" in self.results:
            ax5 = plt.subplot(2, 3, 5)
            data = self.results["grave_overhead"]
            
            py_plain = data["Python_Plain"]["avg_time"]
            py_grave = data["Python_GRAVE"]["avg_time"]
            cpp_plain = data["C++_Plain"]["avg_time"]
            cpp_grave = data["C++_GRAVE"]["avg_time"]
            
            py_overhead = (py_grave - py_plain) / py_plain * 100
            cpp_overhead = (cpp_grave - cpp_plain) / cpp_plain * 100
            
            categories = ['Python', 'C++']
            overheads = [py_overhead, cpp_overhead]
            colors = ['#3498db', '#e74c3c']
            
            bars = ax5.bar(categories, overheads, color=colors, alpha=0.7)
            ax5.set_ylabel('GRAVE Overhead (%)', fontweight='bold')
            ax5.set_title('GRAVE Performance Cost', fontweight='bold')
            ax5.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax5.grid(axis='y', alpha=0.3)
            
            for bar, val in zip(bars, overheads):
                ax5.text(bar.get_x() + bar.get_width()/2., val + 1,
                        f'{val:+.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Plot 6: Summary table
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        if "speed_comparison" in self.results:
            data = self.results["speed_comparison"]
            avg_speedup = np.mean([d["speedup"] for d in data])
            
            summary_text = [
                "Performance Summary",
                "=" * 40,
                f"Average C++ Speedup: {avg_speedup:.1f}x",
                "",
                "Interpretation:",
                f"• C++ is ~{avg_speedup:.0f}x faster than Python",
                "• Same algorithm, different languages",
                "• C++ recommended for production use",
                "• Python useful for prototyping/testing",
            ]
            
            ax6.text(0.1, 0.9, '\n'.join(summary_text),
                    transform=ax6.transAxes,
                    fontsize=11,
                    verticalalignment='top',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        output_file = self.output_dir / "python_vs_cpp_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to {output_file}")
        plt.close()


def main():
    """Run Python vs C++ comparison experiments."""
    print("\n" + "="*70)
    print(" Python vs C++ MCTS Performance Comparison")
    print("="*70)
    print("\nThis experiment compares:")
    print("  1. Pure Python MCTS implementation")
    print("  2. C++ MCTS implementation (via ctypes)")
    print("\nNote: Both use identical algorithms (PUCT + GRAVE + NN)")
    print("="*70)
    
    runner = PythonVsCppExperiment(output_dir="python_vs_cpp_results")
    
    try:
        # Experiment 1: Speed comparison at various sim counts
        runner.experiment_speed_comparison(
            sim_counts=[100, 300, 500],
            num_games=10
        )
        
        # Experiment 2: Find speed-equivalent sim count
        # runner.experiment_equivalent_strength(
        #     base_sims=500,
        #     num_games=10
        # )
        
        # Experiment 3: GRAVE overhead comparison
        runner.experiment_grave_overhead(
            sims=300,
            num_games=8
        )
        
        # Save and plot
        runner.save_results()
        runner.plot_results()
        
        print("\n" + "="*70)
        print(" Experiments Complete!")
        print("="*70)
        print(f"\nResults saved in: {runner.output_dir}/")
        print("  - python_vs_cpp_results.json")
        print("  - python_vs_cpp_analysis.png")
        
        # Print summary
        if "speed_comparison" in runner.results:
            data = runner.results["speed_comparison"]
            avg_speedup = np.mean([d["speedup"] for d in data])
            print(f"\nKey Finding: C++ is ~{avg_speedup:.1f}x faster than Python")
        
        print("\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Experiment interrupted")
        runner.save_results()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        runner.save_results()


if __name__ == "__main__":
    main()