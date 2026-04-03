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

def find_crossing_time(t, y, target_value, is_positive_step=True):
    t_cross = t[-1]
    for i in range(1, len(t)):
        if (is_positive_step and y[i] >= target_value) or \
           (not is_positive_step and y[i] <= target_value):
            dy = y[i] - y[i-1]
            dt = t[i] - t[i-1]
            if dy != 0:
                t_cross = t[i-1] + (target_value - y[i-1]) * (dt / dy)
            else:
                t_cross = t[i]
            break
    return t_cross

def simulate_fopdt(t, K, tau, theta, y0, delta_u=1.0):
    y_sim = np.ones_like(t) * y0
    for i, tp in enumerate(t):
        if tp > theta:
            y_sim[i] = y0 + K * delta_u * (1 - np.exp(-(tp - theta) / tau))
    return y_sim

def sundaresan_identification(t, y, delta_u=1.0):
    y0 = np.mean(y[:5])
    y_final = np.mean(y[-5:])
    dy = y_final - y0
    K = dy / delta_u
    is_pos = dy > 0

    y_35 = y0 + 0.353 * dy
    y_85 = y0 + 0.853 * dy

    t1 = find_crossing_time(t, y, y_35, is_pos)
    t2 = find_crossing_time(t, y, y_85, is_pos)

    tau = 0.67 * (t2 - t1)
    theta = max(0.0, 1.3 * t1 - 0.29 * t2)

    return {
        'K': K, 'tau': tau, 'theta': theta,
        'y0': y0, 'y_final': y_final, 'dy': dy,
        't1': t1, 't2': t2, 'y_35': y_35, 'y_85': y_85
    }

def plot_sundaresan_results(t, y, res, delta_u=1.0, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 7))

    K, tau, theta = res['K'], res['tau'], res['theta']
    y0, y_final = res['y0'], res['y_final']
    t1, t2 = res['t1'], res['t2']
    y_35, y_85 = res['y_35'], res['y_85']

    ax.plot(t, y, 'b-', linewidth=2, label='Real Data', alpha=0.5)

    y_sim = simulate_fopdt(t, K, tau, theta, y0, delta_u)
    ax.plot(t, y_sim, 'g-', linewidth=2.5, 
            label=f"FOPDT (Sundaresan)\n$\\tau={tau:.3f}s, \\theta={theta:.3f}s$")

    ax.plot(t1, y_35, 'ro', markersize=7, label='35.3% ($t_1$)')
    ax.plot(t2, y_85, 'ro', markersize=7, label='85.3% ($t_2$)')

    ax.axhline(y_final, color='gray', linestyle=':', alpha=0.7)
    ax.axhline(y0, color='gray', linestyle=':', alpha=0.7)
    ax.axhline(y_35, color='red', linestyle=':', alpha=0.3)
    ax.axhline(y_85, color='red', linestyle=':', alpha=0.3)

    ax.set_title("System Identification - Sundaresan/Krishnaswamy Method", fontsize=14, fontweight='bold')
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
    parser = argparse.ArgumentParser(description="Process Identification using Sundaresan's Method")
    parser.add_argument('--data', required=True, help='Path to data file')
    parser.add_argument('--delta_u', type=float, default=1.0, help='Step amplitude')
    parser.add_argument('--save', help='Path to save plot')
    args = parser.parse_args()

    filepath = Path(args.data)
    if not filepath.exists():
        print(f"Error: File not found - {filepath}")
        return 1

    t, y = load_data(filepath)
    res = sundaresan_identification(t, y, args.delta_u)

    print("\n" + "=" * 50)
    print(" SUNDARESAN / KRISHNASWAMY RESULTS ")
    print("=" * 50)
    print(f"Static Gain (K):     {res['K']:.4f}")
    print(f"Time Const (tau):    {res['tau']:.4f} s")
    print(f"Dead Time (theta):   {res['theta']:.4f} s")
    print("-" * 50)
    print(f"Time at 35.3% (t1):  {res['t1']:.4f} s")
    print(f"Time at 85.3% (t2):  {res['t2']:.4f} s")
    print("=" * 50 + "\n")

    plot_sundaresan_results(t, y, res, args.delta_u, args.save)
    return 0

if __name__ == '__main__':
    exit(main())