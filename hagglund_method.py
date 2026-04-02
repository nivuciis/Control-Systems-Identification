import argparse
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from pathlib import Path

def load_data(filepath):
    data = np.loadtxt(filepath, delimiter=',')
    y = data[:, 0]  
    t = data[:, 1]   
    sort_idx = np.argsort(t)
    return t[sort_idx], y[sort_idx]

def simulate_fopdt(t, K, tau, theta, y0, delta_u=1.0):
    y_sim = np.zeros_like(t)
    for i, time_point in enumerate(t):
        if time_point <= theta:
            y_sim[i] = y0
        else:
            y_sim[i] = y0 + K * delta_u * (1 - np.exp(-(time_point - theta) / tau))
    return y_sim

def hagglund_identification(t, y, delta_u=1.0):
    y0 = np.mean(y[:5])
    y_final = np.mean(y[-5:])
    K = (y_final - y0) / delta_u

    window = min(11, max(5, len(y) // 10))
    if window % 2 == 0: window += 1 
    
    if len(y) > window:
        y_smooth = savgol_filter(y, window_length=window, polyorder=3)
    else:
        y_smooth = y

    dydt = np.gradient(y_smooth, t)
    idx_max = np.argmax(np.abs(dydt))
    
    m = dydt[idx_max]
    t_inf = t[idx_max]
    y_inf = y_smooth[idx_max]

    if abs(m) > 1e-9:
        t_cross = t_inf - (y_inf - y0) / m
    else:
        t_cross = t[0]
        
    theta = max(0.0, t_cross)

    y_target = y0 + 0.632 * (y_final - y0)
    
    is_positive_step = (y_final > y0)
    t2 = t[-1] 
    
    for i in range(1, len(t)):
        if (is_positive_step and y_smooth[i] >= y_target) or \
           (not is_positive_step and y_smooth[i] <= y_target):
            dy = y_smooth[i] - y_smooth[i-1]
            dt = t[i] - t[i-1]
            t2 = t[i-1] + (y_target - y_smooth[i-1]) * (dt / dy)
            break
            
    tau = max(0.001, t2 - theta)

    return {
        'K': K,
        'tau': tau,
        'theta': theta,
        'y0': y0,
        'y_final': y_final,
        'm': m,
        't_inf': t_inf,
        'y_inf': y_inf,
        'y_target': y_target,
        't2': t2,
        'y_smooth': y_smooth
    }

def plot_results(t, y, results, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 7))

    K, tau, theta = results['K'], results['tau'], results['theta']
    y0, y_final, y_target = results['y0'], results['y_final'], results['y_target']
    t_inf, y_inf, m = results['t_inf'], results['y_inf'], results['m']
    t2 = results['t2']

    ax.plot(t, y, 'b-', linewidth=2, label='Real Data', alpha=0.6)
    
    y_sim = simulate_fopdt(t, K, tau, theta, y0)
    ax.plot(t, y_sim, 'g-', linewidth=2.5, label=f'FOPDT Model\n$G(s) = {K:.2f} / ({tau:.3f}s + 1) e^{{-{theta:.3f}s}}$')

    t_tangent = np.linspace(max(t[0], theta - tau*0.5), min(t[-1], t_inf + tau*1.5), 100)
    y_tangent = m * (t_tangent - t_inf) + y_inf
    ax.plot(t_tangent, y_tangent, 'r--', linewidth=2, label='Tangent Line')

    ax.axhline(y=y0, color='gray', linestyle=':', alpha=0.7)
    ax.axhline(y=y_final, color='gray', linestyle=':', alpha=0.7)
    ax.axhline(y=y_target, color='orange', linestyle=':', alpha=0.7, label='63.2% of Final Value')

    ax.plot(t_inf, y_inf, 'ko', markersize=6, label='Inflection Point')
    ax.plot(theta, y0, 'ro', markersize=8, label=f'Dead Time ($\\theta$ = {theta:.3f}s)')
    ax.plot(t2, y_target, 'go', markersize=8, label=f'63.2% Time ($t_2$ = {t2:.3f}s)')

    ax.set_title("System Identification - Hägglund's Method", fontsize=14, fontweight='bold')
    ax.set_xlabel('Time [s]', fontsize=12)
    ax.set_ylabel('Amplitude (Output)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)

    margin = (np.max(y) - np.min(y)) * 0.1
    ax.set_ylim(np.min(y) - margin, np.max(y) + margin)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to: {save_path}")

    plt.show()

def main():
    parser = argparse.ArgumentParser(description="FOPDT Identification using Hägglund's Method (63.2%)")
    parser.add_argument('--data', required=True, help='Path to CSV/TXT file (columns: y, t)')
    parser.add_argument('--delta_u', type=float, default=1.0, help='Input step amplitude (default: 1.0)')
    parser.add_argument('--save', help='Path to save the plot')
    args = parser.parse_args()

    filepath = Path(args.data)
    if not filepath.exists():
        print(f"Error: File not found - {filepath}")
        return 1

    t, y = load_data(filepath)
    results = hagglund_identification(t, y, args.delta_u)

    print("\n" + "=" * 50)
    print(" FOPDT IDENTIFICATION RESULTS (HÄGGLUND)")
    print("=" * 50)
    print(f"Static Gain (K):      {results['K']:.4f}")
    print(f"Time Constant (tau):  {results['tau']:.4f} s")
    print(f"Dead Time (theta):    {results['theta']:.4f} s")
    print("-" * 50)
    print(f"Initial Value (y0):   {results['y0']:.4f}")
    print(f"Final Value (y_final):{results['y_final']:.4f}")
    print("=" * 50 + "\n")

    plot_results(t, y, results, args.save)
    return 0

if __name__ == '__main__':
    exit(main())