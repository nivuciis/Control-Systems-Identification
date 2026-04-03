import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_data(filepath):
    data = np.loadtxt(filepath, delimiter=',')
    y = data[:, 0]  
    t = data[:, 1]   
    sort_idx = np.argsort(t)
    return t[sort_idx], y[sort_idx]

def simulate_fopdt(t, K, tau, theta, y0, delta_u=1.0):
    y_sim = np.ones_like(t) * y0
    for i, tp in enumerate(t):
        if tp > theta:
            y_sim[i] = y0 + K * delta_u * (1 - np.exp(-(tp - theta) / tau))
    return y_sim

def mollenkamp_identification(t, y, delta_u=1.0):
    y0 = np.mean(y[:5])
    y_final = np.mean(y[-10:])
    dy = y_final - y0
    K = dy / delta_u
    
    t_shifted = t - t[0]
    y_shifted = y - y0
    
    e = dy - y_shifted
    
    A0 = np.trapezoid(e, t_shifted)
    A1 = np.trapezoid(t_shifted * e, t_shifted)
    
    T_sum = A0 / dy
    T_sq = A1 / dy
    
    val_inside_sqrt = 2 * T_sq - T_sum**2
    
    if val_inside_sqrt < 0:
        tau = T_sum
        theta = 0.0
    else:
        tau = np.sqrt(val_inside_sqrt)
        theta = max(0.0, T_sum - tau)

    return {
        'K': K, 'tau': tau, 'theta': theta,
        'y0': y0, 'y_final': y_final, 'dy': dy,
        'A0': A0, 'A1': A1
    }

def plot_mollenkamp_results(t, y, res, delta_u=1.0, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 7))

    K, tau, theta = res['K'], res['tau'], res['theta']
    y0, y_final = res['y0'], res['y_final']

    ax.plot(t, y, 'b-', linewidth=2, label='Real Data', alpha=0.5)

    y_sim = simulate_fopdt(t, K, tau, theta, y0, delta_u)
    ax.plot(t, y_sim, 'g-', linewidth=2.5, 
            label=f"FOPDT (Mollenkamp)\n$\\tau={tau:.3f}s, \\theta={theta:.3f}s$")

    ax.fill_between(t, y, y_final, where=(y < y_final) if res['dy'] > 0 else (y > y_final),
                    color='gray', alpha=0.2, label='Integration Area ($A_0$)')

    ax.axhline(y_final, color='gray', linestyle=':', alpha=0.7)
    ax.axhline(y0, color='gray', linestyle=':', alpha=0.7)

    ax.set_title("System Identification - Mollenkamp Method", fontsize=14, fontweight='bold')
    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("Amplitude", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=11)

    margin = (np.max(y) - np.min(y)) * 0.1
    ax.set_ylim(np.min(y) - margin, np.max(y) + margin)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Process Identification using Mollenkamp's Method")
    parser.add_argument('--data', required=True, help='Path to data file')
    parser.add_argument('--delta_u', type=float, default=1.0, help='Step amplitude')
    parser.add_argument('--save', help='Path to save plot')
    args = parser.parse_args()

    filepath = Path(args.data)
    if not filepath.exists():
        print(f"Error: File not found - {filepath}")
        return 1

    t, y = load_data(filepath)
    res = mollenkamp_identification(t, y, args.delta_u)

    print("\n" + "=" * 50)
    print(" MOLLENKAMP RESULTS ")
    print("=" * 50)
    print(f"Static Gain (K):     {res['K']:.4f}")
    print(f"Time Const (tau):    {res['tau']:.4f} s")
    print(f"Dead Time (theta):   {res['theta']:.4f} s")
    print("-" * 50)
    print(f"Area 0 (A0):         {res['A0']:.4f}")
    print(f"Area 1 (A1):         {res['A1']:.4f}")
    print("=" * 50 + "\n")

    plot_mollenkamp_results(t, y, res, args.delta_u, args.save)
    return 0

if __name__ == '__main__':
    exit(main())