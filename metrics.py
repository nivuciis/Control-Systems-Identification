import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.integrate import simpson

import ZN_method as zn
import hagglund_method as hm
import smith_method as sm
import sundaresan_method as sun
import mollenkamp_method as mol

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

def simulate_sopdt(t, K, tau1, tau2, theta, y0, delta_u=1.0):
    y_sim = np.ones_like(t) * y0
    for i, tp in enumerate(t):
        if tp > theta:
            t_eff = tp - theta
            if abs(tau1 - tau2) < 1e-4:
                y_sim[i] = y0 + K * delta_u * (1 - (1 + t_eff/tau1) * np.exp(-t_eff/tau1))
            else:
                term1 = tau1 * np.exp(-t_eff / tau1)
                term2 = tau2 * np.exp(-t_eff / tau2)
                y_sim[i] = y0 + K * delta_u * (1 - (term1 - term2) / (tau1 - tau2))
    return y_sim

def evaluate_metrics(t, y_real, y_sim):
    e = y_real - y_sim
    mse = np.mean(e**2)
    iae = simpson(np.abs(e), x=t)
    ise = simpson(e**2, x=t)
    itae = simpson(t * np.abs(e), x=t)
    return mse, iae, ise, itae

def print_metrics_table(metrics_dict):
    print("\n" + "="*85)
    print(f"{'GLOBAL MODEL PERFORMANCE COMPARISON':^85}")
    print("="*85)
    print(f"{'METHOD':<25} | {'MSE':<10} | {'IAE':<12} | {'ISE':<10} | {'ITAE':<12}")
    print("-" * 85)
    
    for name, m in metrics_dict.items():
        print(f"{name:<25} | {m['MSE']:<10.2e} | {m['IAE']:<12.6f} | {m['ISE']:<10.2e} | {m['ITAE']:<12.6f}")
    
    print("="*85 + "\n")

def plot_comparative(t, y_real, sims, metrics_dict, save_path=None):
    fig, ax = plt.subplots(figsize=(14, 8))
    
    styles = {
        'Ziegler-Nichols': ('#FF00FF', ':'),
        'Hagglund':        ('#0000FF', '--'),
        'Smith (1st Ord)': ('#FF8C00', '-.'),
        'Smith (2nd Ord)': ('#008000', '-'),
        'Sundaresan':      ('#FF0000', '--'),
        'Mollenkamp':      ('#8B4513', '-.')
    }
    
    ax.plot(t, y_real, 'k-', linewidth=4, label='Experimental Data', alpha=0.3)
    
    sorted_models = sorted(sims.keys(), key=lambda k: metrics_dict[k]['ISE'])
    
    for model in sorted_models:
        color, l_style = styles[model]
        ax.plot(t, sims[model], color=color, linestyle=l_style, linewidth=2, 
                label=f"{model} (ITAE: {metrics_dict[model]['ITAE']:.2e})")
    
    ax.set_title('Global System Identification Comparison - Step Response', fontsize=15, fontweight='bold')
    ax.set_xlabel('Time [s]', fontsize=13)
    ax.set_ylabel('Process Variable Amplitude', fontsize=13)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='lower right', fontsize=11, shadow=True)
    
    margin = (np.max(y_real) - np.min(y_real)) * 0.1
    ax.set_ylim(np.min(y_real) - margin, np.max(y_real) + margin)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Compares all system identification methods.")
    parser.add_argument('--data', required=True, help='Path to data file')
    parser.add_argument('--delta_u', type=float, default=1.0, help='Step amplitude')
    parser.add_argument('--save', help='Path to save plot')
    args = parser.parse_args()

    filepath = Path(args.data)
    if not filepath.exists():
        print(f"Error: File {filepath} not found.")
        return 1

    t, y_real = load_data(filepath)

    res_zn_raw = zn.zn_identification(t, y_real, setpoint_change_percent=args.delta_u*100)
    K_zn = res_zn_raw['B'] / args.delta_u
    tau_zn = res_zn_raw['A']
    theta_zn = res_zn_raw['step_time'] + res_zn_raw['L']
    y0_zn = res_zn_raw['initial_value']

    res_hagglund = hm.hagglund_identification(t, y_real, args.delta_u)
    res_smith1 = sm.smith_identification(t, y_real, order=1, delta_u=args.delta_u)
    res_smith2 = sm.smith_identification(t, y_real, order=2, delta_u=args.delta_u)
    res_sun = sun.sundaresan_identification(t, y_real, args.delta_u)
    res_mol = mol.mollenkamp_identification(t, y_real, args.delta_u)

    sims = {}
    y0 = res_hagglund['y0'] 

    sims['Ziegler-Nichols'] = simulate_fopdt(t, K_zn, tau_zn, theta_zn, y0_zn, args.delta_u)
    sims['Hagglund'] = simulate_fopdt(t, res_hagglund['K'], res_hagglund['tau'], res_hagglund['theta'], y0, args.delta_u)
    sims['Smith (1st Ord)'] = simulate_fopdt(t, res_smith1['K'], res_smith1['tau'], res_smith1['theta'], y0, args.delta_u)
    sims['Smith (2nd Ord)'] = simulate_sopdt(t, res_smith2['K'], res_smith2['tau1'], res_smith2['tau2'], res_smith2['theta'], y0, args.delta_u)
    sims['Sundaresan'] = simulate_fopdt(t, res_sun['K'], res_sun['tau'], res_sun['theta'], y0, args.delta_u)
    sims['Mollenkamp'] = simulate_fopdt(t, res_mol['K'], res_mol['tau'], res_mol['theta'], y0, args.delta_u)

    metrics = {}
    for model_name, y_sim in sims.items():
        mse, iae, ise, itae = evaluate_metrics(t, y_real, y_sim)
        metrics[model_name] = {'MSE': mse, 'IAE': iae, 'ISE': ise, 'ITAE': itae}

    print_metrics_table(metrics)
    plot_comparative(t, y_real, sims, metrics, args.save)

    return 0

if __name__ == '__main__':
    exit(main())