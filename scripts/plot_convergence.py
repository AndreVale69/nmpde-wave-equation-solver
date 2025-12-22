#!/usr/bin/env python3
"""
Plot convergence results from wave equation solver.

Usage: python plot_convergence.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_temporal_convergence():
    """Plot temporal convergence rates."""
    data = pd.read_csv('../results/convergence_temporal.csv')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # L2 error
    ax1.loglog(data['dt'], data['eL2_u'], 'o-', label='L2 error (u)', linewidth=2, markersize=8)
    ax1.loglog(data['dt'], data['eH1_u'], 's-', label='H1 error (u)', linewidth=2, markersize=8)
    
    # Reference lines for order 1 and 2
    dt_ref = data['dt'].values
    ax1.loglog(dt_ref, dt_ref**2 * data['eL2_u'].values[0] / dt_ref[0]**2, 
               'k--', alpha=0.5, label='Order 2')
    ax1.loglog(dt_ref, dt_ref * data['eL2_u'].values[0] / dt_ref[0], 
               'k:', alpha=0.5, label='Order 1')
    
    ax1.set_xlabel('Time step Δt', fontsize=12)
    ax1.set_ylabel('Error', fontsize=12)
    ax1.set_title('Temporal Convergence', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Convergence rates
    ax2.semilogx(data['dt'][1:], data['order_L2_u'][1:], 'o-', 
                 label='L2 rate', linewidth=2, markersize=8)
    ax2.semilogx(data['dt'][1:], data['order_H1_u'][1:], 's-', 
                 label='H1 rate', linewidth=2, markersize=8)
    ax2.axhline(y=2, color='k', linestyle='--', alpha=0.5, label='Order 2 (expected)')
    ax2.set_xlabel('Time step Δt', fontsize=12)
    ax2.set_ylabel('Convergence order', fontsize=12)
    ax2.set_title('Convergence Rates', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 3])
    
    plt.tight_layout()
    plt.savefig('../results/convergence_temporal.png', dpi=300, bbox_inches='tight')
    print("Saved: results/convergence_temporal.png")
    plt.close()

def plot_spatial_convergence():
    """Plot spatial convergence rates."""
    data = pd.read_csv('../results/convergence_spatial.csv')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # L2 and H1 errors
    ax1.loglog(data['h'], data['eL2_u'], 'o-', label='L2 error (u)', linewidth=2, markersize=8)
    ax1.loglog(data['h'], data['eH1_u'], 's-', label='H1 error (u)', linewidth=2, markersize=8)
    
    # Reference lines for different polynomial degrees
    h_ref = data['h'].values
    ax1.loglog(h_ref, h_ref**3 * data['eL2_u'].values[0] / h_ref[0]**3, 
               'k--', alpha=0.5, label='Order 3 (P2 expected)')
    ax1.loglog(h_ref, h_ref**2 * data['eH1_u'].values[0] / h_ref[0]**2, 
               'k:', alpha=0.5, label='Order 2 (H1 expected)')
    
    ax1.set_xlabel('Mesh size h', fontsize=12)
    ax1.set_ylabel('Error', fontsize=12)
    ax1.set_title('Spatial Convergence', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Convergence rates
    ax2.semilogx(data['h'][1:], data['order_L2_u'][1:], 'o-', 
                 label='L2 rate', linewidth=2, markersize=8)
    ax2.semilogx(data['h'][1:], data['order_H1_u'][1:], 's-', 
                 label='H1 rate', linewidth=2, markersize=8)
    ax2.axhline(y=3, color='k', linestyle='--', alpha=0.5, label='Order 3 (L2, P2)')
    ax2.axhline(y=2, color='k', linestyle=':', alpha=0.5, label='Order 2 (H1, P2)')
    ax2.set_xlabel('Mesh size h', fontsize=12)
    ax2.set_ylabel('Convergence order', fontsize=12)
    ax2.set_title('Convergence Rates', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 4])
    
    plt.tight_layout()
    plt.savefig('../results/convergence_spatial.png', dpi=300, bbox_inches='tight')
    print("Saved: results/convergence_spatial.png")
    plt.close()

def main():
    """Main plotting function."""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print("Plotting convergence results...")
    
    if os.path.exists('../results/convergence_temporal.csv'):
        plot_temporal_convergence()
    else:
        print("Warning: convergence_temporal.csv not found")
    
    if os.path.exists('../results/convergence_spatial.csv'):
        plot_spatial_convergence()
    else:
        print("Warning: convergence_spatial.csv not found")
    
    print("\nDone! Check results/ directory for plots.")

if __name__ == "__main__":
    main()
