#!/usr/bin/env python3
"""
Plot dispersion and dissipation analysis from wave equation solver.

Usage: python plot_dispersion.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_energy_conservation():
    """Plot energy conservation over time."""
    data = pd.read_csv('../results/time_series.csv')
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Absolute energy
    ax1.plot(data['time'], data['energy'], 'b-', linewidth=2)
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Total Energy', fontsize=12)
    ax1.set_title('Energy vs Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Relative energy change
    initial_energy = data['energy'].iloc[0]
    if initial_energy > 1e-14:
        energy_drift = (data['energy'] - initial_energy) / initial_energy * 100
        ax2.plot(data['time'], energy_drift, 'r-', linewidth=2)
        ax2.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='1% threshold')
        ax2.axhline(y=-1, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Time', fontsize=12)
        ax2.set_ylabel('Energy drift (%)', fontsize=12)
        ax2.set_title('Relative Energy Drift', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../results/energy_conservation.png', dpi=300, bbox_inches='tight')
    print("Saved: results/energy_conservation.png")
    plt.close()

def plot_energy_budget():
    """Plot energy rate vs power, accumulated work, and energy balance."""
    data = pd.read_csv('../results/time_series.csv')

    # Check for new columns
    required = {'power', 'dE_dt', 'residual', 'work', 'energy_balance'}
    if not required.issubset(set(data.columns)):
        print("Skipping energy budget plot: required columns not found.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Rate vs Power
    ax1.plot(data['time'], data['dE_dt'], label='dE/dt', color='tab:blue', linewidth=2)
    ax1.plot(data['time'], data['power'], label='Power ∫ f v', color='tab:orange', linewidth=2)
    ax1.plot(data['time'], data['residual'], label='Residual (dE/dt − Power)', color='tab:red', linewidth=1)
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Rate / Power', fontsize=12)
    ax1.set_title('Energy Rate vs Power Input', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Work and Energy Balance
    ax2.plot(data['time'], data['work'], label='Accumulated Work', color='tab:green', linewidth=2)
    ax2.plot(data['time'], data['energy_balance'], label='Energy Balance E(t) − E(0) − W(t)', color='k', linewidth=2)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Energy / Work', fontsize=12)
    ax2.set_title('Work and Energy Balance', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../results/energy_budget.png', dpi=300, bbox_inches='tight')
    print("Saved: results/energy_budget.png")
    plt.close()

def plot_errors_vs_time():
    """Plot errors vs time for manufactured solution tests."""
    data = pd.read_csv('../results/time_series.csv')
    
    # Check if we have error data (manufactured solution)
    if data['eL2_u'].max() > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # L2 errors
        ax1.semilogy(data['time'], data['eL2_u'], 'b-', linewidth=2, label='L2(u)')
        ax1.semilogy(data['time'], data['eL2_v'], 'r-', linewidth=2, label='L2(v)')
        ax1.set_xlabel('Time', fontsize=12)
        ax1.set_ylabel('L2 Error', fontsize=12)
        ax1.set_title('L2 Errors vs Time', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # H1 error
        ax2.semilogy(data['time'], data['eH1_u'], 'g-', linewidth=2, label='H1(u)')
        ax2.set_xlabel('Time', fontsize=12)
        ax2.set_ylabel('H1 Error', fontsize=12)
        ax2.set_title('H1 Error vs Time', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../results/errors_vs_time.png', dpi=300, bbox_inches='tight')
        print("Saved: results/errors_vs_time.png")
        plt.close()

def analyze_dispersion():
    """Analyze numerical dispersion from energy and error data."""
    data = pd.read_csv('../results/time_series.csv')
    
    fig = plt.figure(figsize=(10, 6))
    
    # Combined view: energy and max displacement
    ax1 = plt.subplot(111)
    ax1.plot(data['time'], data['energy'], 'b-', linewidth=2, label='Energy')
    # Overlay energy balance if present
    if 'energy_balance' in data.columns:
        ax1.plot(data['time'], data['energy_balance'], 'k--', linewidth=1.5, label='Energy Balance')
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Energy', fontsize=12, color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_title('Wave Propagation Analysis', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../results/dispersion_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved: results/dispersion_analysis.png")
    plt.close()

def print_summary_statistics():
    """Print summary statistics from simulation."""
    data = pd.read_csv('../results/time_series.csv')
    
    print("\n" + "="*60)
    print("DISPERSION/DISSIPATION ANALYSIS SUMMARY")
    print("="*60)
    
    initial_energy = data['energy'].iloc[0]
    final_energy = data['energy'].iloc[-1]
    max_energy = data['energy'].max()
    min_energy = data['energy'].min()
    
    print(f"\nEnergy Statistics:")
    print(f"  Initial energy:     {initial_energy:.6e}")
    print(f"  Final energy:       {final_energy:.6e}")
    print(f"  Maximum energy:     {max_energy:.6e}")
    print(f"  Minimum energy:     {min_energy:.6e}")

    # Energy budget if available
    if {'power','dE_dt','residual','work','energy_balance'}.issubset(set(data.columns)):
        final_work = data['work'].iloc[-1]
        final_balance = data['energy_balance'].iloc[-1]
        max_abs_residual = np.max(np.abs(data['residual'].values))
        print(f"\nEnergy Budget:")
        print(f"  Final work:         {final_work:.6e}")
        print(f"  Final energy balance (E−E0−W): {final_balance:+.6e}")
        print(f"  Max |dE/dt − Power|: {max_abs_residual:.6e}")
    
    if initial_energy > 1e-14:
        energy_drift_pct = (final_energy - initial_energy) / initial_energy * 100
        max_drift_pct = (max_energy - initial_energy) / initial_energy * 100
        min_drift_pct = (min_energy - initial_energy) / initial_energy * 100
        
        print(f"\nEnergy Drift:")
        print(f"  Final drift:        {energy_drift_pct:+.4f}%")
        print(f"  Maximum drift:      {max_drift_pct:+.4f}%")
        print(f"  Minimum drift:      {min_drift_pct:+.4f}%")
        
        if abs(energy_drift_pct) < 1.0:
            print(f"\n  ✓ Energy conservation: GOOD (< 1% drift)")
        else:
            print(f"\n  ✗ Energy conservation: POOR (> 1% drift)")
    
    # Error statistics (if available)
    if data['eL2_u'].max() > 0:
        print(f"\nError Statistics (Manufactured Solution):")
        print(f"  Final L2(u):        {data['eL2_u'].iloc[-1]:.6e}")
        print(f"  Final H1(u):        {data['eH1_u'].iloc[-1]:.6e}")
        print(f"  Final L2(v):        {data['eL2_v'].iloc[-1]:.6e}")
    
    print("\n" + "="*60 + "\n")

def main():
    """Main plotting function."""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print("Plotting dispersion/dissipation analysis...")
    
    if os.path.exists('../results/time_series.csv'):
        plot_energy_conservation()
        plot_energy_budget()
        plot_errors_vs_time()
        analyze_dispersion()
        print_summary_statistics()
    else:
        print("Error: time_series.csv not found")
        print("Run wave-dispersion executable first.")
        return
    
    print("\nDone! Check results/ directory for plots.")

if __name__ == "__main__":
    main()