import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
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

def smith_identification(t, y, order, delta_u=1.0):
    y0 = np.mean(y[:5])
    y_final = np.mean(y[-5:])
    dy = y_final - y0
    K = dy / delta_u
    is_pos = dy > 0

    # Base de Smith (Necessária para ambas as ordens)
    y_28 = y0 + 0.283 * dy
    y_63 = y0 + 0.632 * dy
    t_28 = find_crossing_time(t, y, y_28, is_pos)
    t_63 = find_crossing_time(t, y, y_63, is_pos)

    tau_1st = 1.5 * (t_63 - t_28)
    theta_1st = max(0.0, t_63 - tau_1st)

    result = {
        'K': K, 'y0': y0, 'y_final': y_final, 'dy': dy, 'order': order,
        't_28': t_28, 't_63': t_63, 'y_28': y_28, 'y_63': y_63
    }

    if order == 1:
        result['tau'] = tau_1st
        result['theta'] = theta_1st
        return result

    if order == 2:
        def sopdt_wrapper(t_array, tau1, tau2, theta):
            return simulate_sopdt(t_array, K, tau1, tau2, theta, y0, delta_u)

        p0 = [tau_1st / 2, tau_1st / 2, theta_1st]
        bounds = ([0.001, 0.001, 0.0], [np.inf, np.inf, t[-1]])

        try:
            popt, _ = curve_fit(sopdt_wrapper, t, y, p0=p0, bounds=bounds)
            tau1_2nd, tau2_2nd, theta_2nd = popt
        except Exception as e:
            print(f"Erro no ajuste de 2ª Ordem: {e}. Usando estimativa padrão.")
            tau1_2nd, tau2_2nd, theta_2nd = tau_1st / 2, tau_1st / 2, theta_1st

        result['tau1'] = tau1_2nd
        result['tau2'] = tau2_2nd
        result['theta'] = theta_2nd
        return result

def plot_smith_results(t, y, res, delta_u=1.0, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 7))

    K, y0, y_final = res['K'], res['y0'], res['y_final']
    order = res['order']

    ax.plot(t, y, 'b-', linewidth=2, label='Dados Reais', alpha=0.5)

    if order == 1:
        y_sim = simulate_fopdt(t, K, res['tau'], res['theta'], y0, delta_u)
        ax.plot(t, y_sim, 'r--', linewidth=2.5, 
                label=f"1ª Ordem (FOPDT)\n$\\tau={res['tau']:.3f}s, \\theta={res['theta']:.3f}s$")
    else:
        y_sim = simulate_sopdt(t, K, res['tau1'], res['tau2'], res['theta'], y0, delta_u)
        ax.plot(t, y_sim, 'g-', linewidth=2.5, 
                label=f"2ª Ordem (SOPDT)\n$\\tau_1={res['tau1']:.3f}s, \\tau_2={res['tau2']:.3f}s, \\theta={res['theta']:.3f}s$")

    # Marcações clássicas do Método de Smith
    ax.plot(res['t_28'], res['y_28'], 'ko', markersize=6, label='28.3% (Smith)')
    ax.plot(res['t_63'], res['y_63'], 'ko', markersize=6, label='63.2% (Smith)')

    ax.axhline(y_final, color='gray', linestyle=':', alpha=0.7)
    ax.axhline(y0, color='gray', linestyle=':', alpha=0.7)

    ax.set_title(f"Identificação de Sistemas - Método de Smith ({order}ª Ordem)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Tempo [s]", fontsize=12)
    ax.set_ylabel("Amplitude (Saída)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Identificação pelo Método de Smith (1ª ou 2ª Ordem)")
    parser.add_argument('--data', required=True, help='Caminho para o arquivo CSV/TXT (colunas: y, t)')
    parser.add_argument('--delta_u', type=float, default=1.0, help='Amplitude do degrau de entrada (default: 1.0)')
    parser.add_argument('--save', help='Caminho para salvar o gráfico')
    
    # Grupo mutuamente exclusivo para forçar a escolha da ordem
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--1', dest='order1', action='store_true', help='Usa o modelo de 1ª Ordem (FOPDT)')
    group.add_argument('--2', dest='order2', action='store_true', help='Usa o modelo de 2ª Ordem (SOPDT)')
    
    args = parser.parse_args()

    filepath = Path(args.data)
    if not filepath.exists():
        print(f"Erro: Arquivo não encontrado - {filepath}")
        return 1

    t, y = load_data(filepath)
    
    # Define a ordem baseada na flag selecionada
    order = 1 if args.order1 else 2
    
    res = smith_identification(t, y, order, args.delta_u)

    print("\n" + "=" * 50)
    print(f" RESULTADOS - MÉTODO DE SMITH ({order}ª ORDEM) ")
    print("=" * 50)
    print(f"Ganho Estático (K):  {res['K']:.4f}")
    
    if order == 1:
        print(f"Constante (tau):     {res['tau']:.4f} s")
        print(f"Atraso (theta):      {res['theta']:.4f} s")
    else:
        print(f"Constante (tau1):    {res['tau1']:.4f} s")
        print(f"Constante (tau2):    {res['tau2']:.4f} s")
        print(f"Atraso (theta):      {res['theta']:.4f} s")
        
    print("-" * 50)
    print(f"Tempo em 28.3%:      {res['t_28']:.4f} s")
    print(f"Tempo em 63.2%:      {res['t_63']:.4f} s")
    print("=" * 50 + "\n")

    plot_smith_results(t, y, res, args.delta_u, args.save)
    return 0

if __name__ == '__main__':
    exit(main())