#!/usr/bin/env python3
"""
Master script to run the complete MCTS optimization experiment.

This script:
1. Checks prerequisites
2. Compiles C++ engine if needed
3. Runs the experiment
4. Generates visualizations
5. Produces final report
"""

import sys
import os
import subprocess
import time
from pathlib import Path


def print_header(text):
    """Print a formatted header."""
    print("\n" + "="*80)
    print(f" {text}")
    print("="*80 + "\n")


def check_prerequisites():
    """Check if all required files and dependencies exist."""
    print_header("CHECKING PREREQUISITES")
    
    checks_passed = True
    
    # Check Python version
    print("Python version:", sys.version)
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        checks_passed = False
    else:
        print("✓ Python version OK")
    
    # Check required files
    required_files = [
        ('hex_mcts_engine.cpp', 'C++ MCTS engine source'),
        ('interface.py', 'C++ Python interface'),
        ('CppMCTSAgent.py', 'Agent wrapper'),
        ('mcts_experiment.py', 'Experiment runner'),
        ('visualize_results.py', 'Visualization script'),
    ]
    
    print("\nChecking files:")
    for filename, description in required_files:
        path = Path(filename)
        if path.exists():
            print(f"✓ {filename} - {description}")
        else:
            print(f"❌ {filename} missing - {description}")
            checks_passed = False
    
    # Check for model
    model_path = Path("/mnt/project/models/hex11-20180712-3362.policy.pth")
    if model_path.exists():
        print(f"\n✓ Azalea model found")
    else:
        print(f"\n❌ Azalea model not found at {model_path}")
        print("   Please ensure the model is available")
        checks_passed = False
    
    # Check for Python packages
    print("\nChecking Python packages:")
    packages = ['torch', 'numpy', 'matplotlib']
    for package in packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"❌ {package} not installed")
            checks_passed = False
    
    # Check for g++
    print("\nChecking compiler:")
    try:
        result = subprocess.run(['g++', '--version'], 
                              capture_output=True, 
                              text=True,
                              timeout=5)
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            print(f"✓ g++ available: {version}")
        else:
            print("⚠ g++ found but may have issues")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("⚠ g++ not found - will attempt auto-install")
    
    if not checks_passed:
        print("\n❌ Prerequisites check FAILED")
        print("Please fix the issues above before running the experiment")
        return False
    
    print("\n✓ All prerequisites satisfied")
    return True


def compile_cpp_engine():
    """Compile the C++ MCTS engine."""
    print_header("COMPILING C++ ENGINE")
    
    lib_path = Path("hex_mcts_engine.so")
    
    if lib_path.exists():
        print("Shared library already exists")
        response = input("Recompile? (y/N): ").strip().lower()
        if response != 'y':
            print("Skipping compilation")
            return True
    
    print("Compiling hex_mcts_engine.cpp...")
    print("Command: g++ -O3 -std=c++17 -fPIC -shared hex_mcts_engine.cpp -o hex_mcts_engine.so")
    
    try:
        result = subprocess.run(
            [
                'g++',
                '-O3',
                '-std=c++17',
                '-fPIC',
                '-shared',
                'hex_mcts_engine.cpp',
                '-o',
                'hex_mcts_engine.so'
            ],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print("✓ Compilation successful")
            if lib_path.exists():
                size = lib_path.stat().st_size / 1024
                print(f"  Library size: {size:.1f} KB")
            return True
        else:
            print("❌ Compilation failed")
            print("STDERR:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Compilation timed out")
        return False
    except Exception as e:
        print(f"❌ Compilation error: {e}")
        return False


def run_experiment():
    """Run the main experiment."""
    print_header("RUNNING EXPERIMENT")
    
    print("This will take several hours (estimated 4-7 hours)")
    print("The experiment will run 200 games total:")
    print("  - 100 simulations: 50 games")
    print("  - 500 simulations: 50 games")
    print("  - 1000 simulations: 50 games")
    print("  - 2000 simulations: 50 games")
    
    response = input("\nProceed? (y/N): ").strip().lower()
    if response != 'y':
        print("Experiment cancelled")
        return False
    
    print("\nStarting experiment...")
    start_time = time.time()
    
    try:
        # Run experiment as subprocess to capture all output
        result = subprocess.run(
            [sys.executable, 'mcts_experiment.py'],
            timeout=28800  # 8 hour timeout
        )
        
        if result.returncode == 0:
            elapsed = time.time() - start_time
            print(f"\n✓ Experiment completed in {elapsed/60:.1f} minutes")
            return True
        else:
            print(f"\n❌ Experiment failed with return code {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("\n❌ Experiment timed out after 8 hours")
        return False
    except KeyboardInterrupt:
        print("\n\n⚠ Experiment interrupted by user")
        print("Partial results may be available")
        return False
    except Exception as e:
        print(f"\n❌ Experiment error: {e}")
        return False


def generate_visualizations():
    """Generate result visualizations."""
    print_header("GENERATING VISUALIZATIONS")
    
    # Check if results exist
    stats_file = Path("/mnt/user-data/outputs/mcts_experiment_stats.json")
    if not stats_file.exists():
        print("❌ No experiment results found")
        print("Please run the experiment first")
        return False
    
    print("Generating plots from experiment results...")
    
    try:
        result = subprocess.run(
            [sys.executable, 'visualize_results.py'],
            timeout=60
        )
        
        if result.returncode == 0:
            print("\n✓ Visualizations generated successfully")
            return True
        else:
            print(f"\n❌ Visualization failed with return code {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("\n❌ Visualization timed out")
        return False
    except Exception as e:
        print(f"\n❌ Visualization error: {e}")
        return False


def show_final_report():
    """Display final report with results summary."""
    print_header("EXPERIMENT COMPLETE")
    
    # Check for output files
    output_dir = Path("/mnt/user-data/outputs")
    files_created = []
    
    expected_files = [
        ('mcts_experiment_results.json', 'Detailed game results'),
        ('mcts_experiment_stats.json', 'Configuration statistics'),
        ('mcts_optimization_plots.png', 'Full visualization'),
        ('optimal_simulations.png', 'Optimal config plot'),
    ]
    
    print("Generated files:")
    for filename, description in expected_files:
        path = output_dir / filename
        if path.exists():
            size = path.stat().st_size / 1024
            print(f"  ✓ {filename}")
            print(f"    {description} ({size:.1f} KB)")
            files_created.append(str(path))
        else:
            print(f"  ⚠ {filename} - not found")
    
    if files_created:
        print(f"\n📁 All outputs saved to: {output_dir}")
        print("\nTo view results:")
        print("  1. Download the PNG files for visualizations")
        print("  2. Open JSON files for detailed data")
        print("  3. Check console output above for optimal simulation count")
    
    print("\n" + "="*80)
    print(" Thank you for running the MCTS optimization experiment!")
    print("="*80 + "\n")


def main():
    """Main execution flow."""
    print("\n" + "#"*80)
    print("#" + " "*78 + "#")
    print("#" + "  MCTS SIMULATION COUNT OPTIMIZATION EXPERIMENT".center(78) + "#")
    print("#" + " "*78 + "#")
    print("#"*80)
    
    # Step 1: Check prerequisites
    if not check_prerequisites():
        print("\n❌ Cannot proceed without prerequisites")
        sys.exit(1)
    
    # Step 2: Compile C++ engine
    if not compile_cpp_engine():
        print("\n❌ Cannot proceed without C++ engine")
        sys.exit(1)
    
    # Step 3: Run experiment
    if not run_experiment():
        print("\n⚠ Experiment did not complete successfully")
        print("Attempting to generate visualizations from partial results...")
    
    # Step 4: Generate visualizations
    generate_visualizations()
    
    # Step 5: Show final report
    show_final_report()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Script interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)