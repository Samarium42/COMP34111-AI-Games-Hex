"""
Visualization Script for MCTS Experiment Results

Generates plots showing:
1. Wins per second vs simulation count (KEY METRIC)
2. Win rate vs simulation count
3. Average move time vs simulation count
4. Trade-off visualization
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


def load_experiment_data(stats_file: str):
    """Load experiment statistics from JSON."""
    with open(stats_file, 'r') as f:
        data = json.load(f)
    
    # Convert keys back to integers
    return {int(k): v for k, v in data.items()}


def create_visualizations(stats: dict, output_dir: str = "/mnt/user-data/outputs"):
    """
    Create comprehensive visualizations of experiment results.
    """
    # Extract data
    sim_counts = sorted(stats.keys())
    wins_per_sec = [stats[s]['wins_per_second'] for s in sim_counts]
    win_rates = [stats[s]['win_rate'] * 100 for s in sim_counts]
    move_times = [stats[s]['avg_time_per_move'] for s in sim_counts]
    game_times = [stats[s]['avg_game_time'] for s in sim_counts]
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Color scheme
    primary_color = '#2E86AB'
    secondary_color = '#A23B72'
    success_color = '#06A77D'
    warning_color = '#F18F01'
    
    # ========================================================================
    # PLOT 1: WINS PER SECOND (KEY METRIC) - Large, prominent
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(sim_counts, wins_per_sec, marker='o', markersize=10, 
             linewidth=3, color=success_color, label='Wins/Second')
    ax1.fill_between(sim_counts, wins_per_sec, alpha=0.3, color=success_color)
    
    # Highlight the optimal point
    max_idx = np.argmax(wins_per_sec)
    optimal_sims = sim_counts[max_idx]
    optimal_wps = wins_per_sec[max_idx]
    
    ax1.scatter([optimal_sims], [optimal_wps], s=300, c='red', 
                marker='*', zorder=5, edgecolors='black', linewidths=2,
                label=f'Optimal: {optimal_sims} sims')
    
    ax1.set_xlabel('MCTS Simulations', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Wins per Second', fontsize=14, fontweight='bold')
    ax1.set_title('WINS PER SECOND vs SIMULATION COUNT\n(Higher is Better)', 
                  fontsize=16, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=12, loc='best')
    
    # Add value labels
    for i, (s, w) in enumerate(zip(sim_counts, wins_per_sec)):
        ax1.annotate(f'{w:.4f}', 
                    xy=(s, w), 
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center',
                    fontsize=10,
                    fontweight='bold')
    
    # ========================================================================
    # PLOT 2: WIN RATE
    # ========================================================================
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.bar(range(len(sim_counts)), win_rates, color=primary_color, alpha=0.7)
    ax2.axhline(y=50, color='red', linestyle='--', linewidth=2, alpha=0.5, 
                label='50% (Expected)')
    ax2.set_xticks(range(len(sim_counts)))
    ax2.set_xticklabels(sim_counts)
    ax2.set_xlabel('MCTS Simulations', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Win Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Win Rate (RED Perspective)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax2.legend()
    ax2.set_ylim([0, 100])
    
    # Add value labels on bars
    for i, (s, wr) in enumerate(zip(sim_counts, win_rates)):
        ax2.text(i, wr + 2, f'{wr:.1f}%', ha='center', fontweight='bold')
    
    # ========================================================================
    # PLOT 3: AVERAGE MOVE TIME
    # ========================================================================
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(sim_counts, move_times, marker='s', markersize=8, 
             linewidth=2.5, color=warning_color, label='Move Time')
    ax3.fill_between(sim_counts, move_times, alpha=0.3, color=warning_color)
    ax3.set_xlabel('MCTS Simulations', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax3.set_title('Average Move Time', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend()
    
    # Add value labels
    for s, mt in zip(sim_counts, move_times):
        ax3.annotate(f'{mt:.3f}s', 
                    xy=(s, mt), 
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center',
                    fontsize=9)
    
    # ========================================================================
    # PLOT 4: GAME TIME
    # ========================================================================
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(sim_counts, game_times, marker='D', markersize=8,
             linewidth=2.5, color=secondary_color, label='Game Time')
    ax4.fill_between(sim_counts, game_times, alpha=0.3, color=secondary_color)
    ax4.set_xlabel('MCTS Simulations', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax4.set_title('Average Game Duration', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.legend()
    
    # Add value labels
    for s, gt in zip(sim_counts, game_times):
        ax4.annotate(f'{gt:.1f}s', 
                    xy=(s, gt), 
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center',
                    fontsize=9)
    
    # ========================================================================
    # PLOT 5: EFFICIENCY SCATTER (Win Rate vs Move Time)
    # ========================================================================
    ax5 = fig.add_subplot(gs[2, 1])
    
    # Scatter plot with size based on wins/sec
    sizes = [w * 5000 for w in wins_per_sec]  # Scale for visibility
    scatter = ax5.scatter(move_times, win_rates, s=sizes, 
                         c=sim_counts, cmap='viridis', 
                         alpha=0.6, edgecolors='black', linewidths=1.5)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax5)
    cbar.set_label('Simulations', fontsize=10, fontweight='bold')
    
    # Annotate each point with sim count
    for i, (mt, wr, s) in enumerate(zip(move_times, win_rates, sim_counts)):
        ax5.annotate(f'{s}', 
                    xy=(mt, wr), 
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=10,
                    fontweight='bold')
    
    # Highlight optimal point
    ax5.scatter([move_times[max_idx]], [win_rates[max_idx]], 
               s=500, c='red', marker='*', zorder=5,
               edgecolors='black', linewidths=2)
    
    ax5.set_xlabel('Average Move Time (seconds)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Win Rate (%)', fontsize=12, fontweight='bold')
    ax5.set_title('Efficiency Trade-off\n(Bubble size = Wins/Second)', 
                 fontsize=13, fontweight='bold')
    ax5.grid(True, alpha=0.3, linestyle='--')
    
    # Overall title
    fig.suptitle('MCTS SIMULATION COUNT OPTIMIZATION RESULTS', 
                fontsize=18, fontweight='bold', y=0.995)
    
    # Save figure
    output_path = Path(output_dir) / "mcts_optimization_plots.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved visualization to: {output_path}")
    
    plt.close()
    
    # ========================================================================
    # BONUS: Create a simple focused plot for wins/sec only
    # ========================================================================
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(sim_counts, wins_per_sec, marker='o', markersize=12,
            linewidth=3, color=success_color)
    ax.fill_between(sim_counts, wins_per_sec, alpha=0.3, color=success_color)
    
    # Highlight optimal
    ax.scatter([optimal_sims], [optimal_wps], s=400, c='red',
              marker='*', zorder=5, edgecolors='black', linewidths=2)
    
    ax.set_xlabel('MCTS Simulations', fontsize=14, fontweight='bold')
    ax.set_ylabel('Wins per Second', fontsize=14, fontweight='bold')
    ax.set_title(f'Optimal Configuration: {optimal_sims} Simulations\n'
                f'Wins/Second: {optimal_wps:.4f}', 
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add value labels
    for s, w in zip(sim_counts, wins_per_sec):
        ax.annotate(f'{s} sims\n{w:.4f} w/s', 
                   xy=(s, w), 
                   xytext=(0, 15),
                   textcoords='offset points',
                   ha='center',
                   fontsize=11,
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', 
                            facecolor='yellow', 
                            alpha=0.7))
    
    output_path2 = Path(output_dir) / "optimal_simulations.png"
    plt.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved focused plot to: {output_path2}")
    
    plt.close()
    
    return optimal_sims, optimal_wps


def print_analysis(stats: dict):
    """Print detailed analysis of results."""
    sim_counts = sorted(stats.keys())
    wins_per_sec = [stats[s]['wins_per_second'] for s in sim_counts]
    
    max_idx = np.argmax(wins_per_sec)
    optimal_sims = sim_counts[max_idx]
    optimal_wps = wins_per_sec[max_idx]
    
    print("\n" + "="*70)
    print(" OPTIMIZATION ANALYSIS")
    print("="*70)
    
    print(f"\n🎯 OPTIMAL CONFIGURATION: {optimal_sims} simulations")
    print(f"   Wins per Second: {optimal_wps:.4f}")
    print(f"   Win Rate: {stats[optimal_sims]['win_rate']*100:.1f}%")
    print(f"   Avg Move Time: {stats[optimal_sims]['avg_time_per_move']:.3f}s")
    
    print(f"\n📊 EFFICIENCY RANKINGS:")
    sorted_configs = sorted(sim_counts, 
                           key=lambda s: stats[s]['wins_per_second'], 
                           reverse=True)
    
    for rank, s in enumerate(sorted_configs, 1):
        wps = stats[s]['wins_per_second']
        wr = stats[s]['win_rate'] * 100
        mt = stats[s]['avg_time_per_move']
        symbol = "⭐" if rank == 1 else "  "
        print(f"   {symbol} #{rank}: {s:4d} sims - "
              f"WPS={wps:.4f}, WinRate={wr:.1f}%, MoveTime={mt:.3f}s")
    
    print(f"\n💡 INSIGHTS:")
    
    # Check for diminishing returns
    for i in range(1, len(sim_counts)):
        prev_sims = sim_counts[i-1]
        curr_sims = sim_counts[i]
        prev_wps = wins_per_sec[i-1]
        curr_wps = wins_per_sec[i]
        
        improvement = ((curr_wps - prev_wps) / prev_wps * 100) if prev_wps > 0 else 0
        
        if improvement < 0:
            print(f"   • {prev_sims}→{curr_sims} sims: "
                  f"DECREASING efficiency ({improvement:.1f}% worse)")
        elif improvement < 5:
            print(f"   • {prev_sims}→{curr_sims} sims: "
                  f"Diminishing returns ({improvement:.1f}% gain)")
    
    # Time budget analysis
    TIME_BUDGET = 180  # seconds per player
    for s in sim_counts:
        mt = stats[s]['avg_time_per_move']
        moves_possible = TIME_BUDGET / mt
        print(f"   • {s:4d} sims: Can play ~{moves_possible:.0f} moves "
              f"in {TIME_BUDGET}s budget")
    
    print("="*70 + "\n")


def main():
    """Load results and create visualizations."""
    stats_file = "/mnt/user-data/outputs/mcts_experiment_stats.json"
    
    print("Loading experiment results...")
    try:
        stats = load_experiment_data(stats_file)
    except FileNotFoundError:
        print(f"ERROR: Results file not found at {stats_file}")
        print("Please run the experiment first (mcts_experiment.py)")
        return
    
    print(f"Loaded data for {len(stats)} configurations")
    
    # Print analysis
    print_analysis(stats)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    optimal_sims, optimal_wps = create_visualizations(stats)
    
    print(f"\n{'='*70}")
    print(f"🏆 RECOMMENDATION: Use {optimal_sims} simulations for optimal performance")
    print(f"   This configuration achieves {optimal_wps:.4f} wins per second")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()