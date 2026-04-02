#!/usr/bin/env python3
"""
Ziegler-Nichols Process Identification Method

This script performs process identification using the Ziegler-Nichols reaction curve method.
It calculates:
- Process reaction rate (R = B/A) from the tangent at the point of maximum slope
- Unit reaction rate (R1 = R/X) where X is the set point change percentage
- Effective delay (L) - time from step change to tangent crossing initial value
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_data(filepath):
    """Load process data from file. Expects two columns: value, time"""
    data = np.loadtxt(filepath, delimiter=',')
    value = data[:, 0]  
    time = data[:, 1]   

    # Sort by time (in case data is not monotonic)
    sort_idx = np.argsort(time)
    time = time[sort_idx]
    value = value[sort_idx]

    return time, value


def find_step_change_info(time, value, threshold=0.01):
    """
    Find the step change point by detecting significant change from initial value.
    Returns the index where step change begins and the initial value.
    """
    initial_value = value[0]

    # Find where value starts changing significantly
    for i in range(1, len(value)):
        if abs(value[i] - initial_value) > threshold:
            # Step change detected
            step_idx = i - 1
            return step_idx, initial_value

    return 0, initial_value


def smooth_data(value, window_size=5):
    """Apply moving average smoothing to reduce noise."""
    smoothed = np.convolve(value, np.ones(window_size)/window_size, mode='same')
    return smoothed


def find_steepest_slope(time, value, window_size=7):
    """
    Find the point of maximum slope (steepest point) using a sliding window.
    Uses smoothed data for more reliable slope detection.
    Handles both positive (increasing) and negative (decreasing) processes.
    Returns the index of the steepest point and the slope at that point.
    """
    # Smooth the data first
    smoothed = smooth_data(value, window_size)

    max_magnitude = -np.inf
    steepest_idx = 0
    steepest_slope = 0

    # Calculate slopes using a moving window on smoothed data
    # Avoid the very beginning (startup) and very end (noise)
    start_idx = window_size * 2
    end_idx = len(value) - window_size * 2  # Leave margin at end

    min_dt = 0.001  # Minimum time difference to consider valid

    for i in range(start_idx, end_idx):
        # Find next valid point with sufficient time difference
        j = i + window_size
        while j < len(time) - 1 and (time[j] - time[i]) < min_dt:
            j += 1

        dt = time[j] - time[i]
        dv = smoothed[j] - smoothed[i]

        if dt >= min_dt:  # Valid time difference
            slope = dv / dt
            # Track the slope with largest magnitude (steepest, regardless of direction)
            if abs(slope) > max_magnitude:
                max_magnitude = abs(slope)
                steepest_idx = (i + j) // 2
                steepest_slope = slope

    return steepest_idx, steepest_slope


def calculate_tangent_line(time, value, steepest_idx, slope):
    """
    Calculate the tangent line at the point of maximum slope.
    Returns the tangent line values for all time points.
    """
    t0 = time[steepest_idx]
    y0 = value[steepest_idx]

    # Tangent line: y - y0 = slope * (t - t0)
    # => y = slope * (t - t0) + y0
    tangent = slope * (time - t0) + y0

    return tangent


def find_effective_delay(time, value, tangent, initial_value, step_time):
    """
    Find the effective delay (L): time from step to where tangent crosses initial value.

    The effective delay L is the time interval between the step change and the point
    where the tangent line (drawn at maximum slope) crosses the initial process value line.

    Returns (L, effective_delay_time).
    """
    # Find where tangent crosses initial value
    t_cross = None
    for i in range(len(tangent) - 1):
        y1, y2 = tangent[i], tangent[i + 1]
        # Check if initial_value is between y1 and y2
        if (y1 <= initial_value <= y2) or (y2 <= initial_value <= y1):
            if y2 != y1:
                t1, t2 = time[i], time[i + 1]
                frac = (initial_value - y1) / (y2 - y1)
                t_cross = t1 + frac * (t2 - t1)
                break

    if t_cross is None:
        # Extrapolate backwards if not found in data range
        return 0, step_time

    # L is the time from step to where tangent crosses initial value
    L = t_cross - step_time
    return max(0, L), t_cross


def find_tangent_crossing_point(time, tangent, target_value):
    """
    Find where the tangent line crosses a target value.
    Returns the time where crossing occurs.
    """
    for i in range(len(tangent) - 1):
        y1, y2 = tangent[i], tangent[i + 1]
        # Check if target is between y1 and y2
        if (y1 <= target_value <= y2) or (y2 <= target_value <= y1):
            if y2 != y1:
                t1, t2 = time[i], time[i + 1]
                frac = (target_value - y1) / (y2 - y1)
                return t1 + frac * (t2 - t1)
    return None


def calculate_reaction_rate(time, value, tangent, initial_value, steepest_idx, max_slope):
    """
    Calculate the process reaction rate R = B/A.

    For Ziegler-Nichols:
    - Tangent line is drawn at point of maximum slope (inflection point)
    - Line A: horizontal from where tangent crosses initial value (t_L) to end of time range
    - Line B: vertical rise from end of Line A up to tangent line at that time
    - R = B/A = slope

    Actually, the slope R is B/A where:
    - A is the time it takes for the tangent to go from initial value to final steady state
    - B is the total change in PV (final - initial)
    """
    # Find steady state (average of last 15% of data to avoid end noise)
    steady_start = int(len(value) * 0.85)
    final_value = np.mean(value[steady_start:])

    # Find where tangent crosses initial value (this is the "delay" point on tangent)
    t_at_initial = find_tangent_crossing_point(time, tangent, initial_value)
    if t_at_initial is None:
        # If tangent never crosses initial, extrapolate backwards
        t_at_steepest = time[steepest_idx]
        y_at_steepest = value[steepest_idx]
        # t = t0 - (y0 - initial)/slope
        if max_slope != 0:
            t_at_initial = t_at_steepest - (y_at_steepest - initial_value) / max_slope
        else:
            t_at_initial = time[0]

    # Find where tangent would cross final value
    t_at_final = find_tangent_crossing_point(time, tangent, final_value)
    if t_at_final is None:
        # Extrapolate forward
        t_at_steepest = time[steepest_idx]
        y_at_steepest = value[steepest_idx]
        if max_slope != 0:
            t_at_final = t_at_steepest + (final_value - y_at_steepest) / max_slope
        else:
            t_at_final = time[-1]

    # Line A: time interval from tangent start (at initial) to tangent end (at final)
    A = t_at_final - t_at_initial

    # Line B: the change in controlled variable (final - initial)
    B = final_value - initial_value

    # Process reaction rate R = B/A (equals the tangent slope)
    if abs(A) > 1e-9:
        R = B / A
    else:
        R = max_slope

    return R, A, B, t_at_initial, t_at_final, final_value


def calculate_unit_reaction_rate(R, setpoint_change_percent):
    """
    Calculate the unit reaction rate R1 = R / X
    where X is the percentage of set point change.
    """
    if setpoint_change_percent > 0:
        return R / setpoint_change_percent
    return R


def zn_identification(time, value, setpoint_change_percent=100):
    """
    Perform complete Ziegler-Nichols identification.

    Returns:
        dict with all calculated parameters and indices
    """
    # Find step change info
    step_idx, initial_value = find_step_change_info(time, value)
    step_time = time[step_idx]

    # Smooth data for better slope detection
    smoothed = smooth_data(value)

    # Find steepest slope point
    steepest_idx, max_slope = find_steepest_slope(time, smoothed)

    # Calculate tangent line using smoothed value at steepest point
    tangent = calculate_tangent_line(time, smoothed, steepest_idx, max_slope)

    # Calculate reaction rate parameters
    R, A, B, t_start, t_end, final_value = calculate_reaction_rate(time, value, tangent, initial_value, steepest_idx, max_slope)

    # Calculate effective delay
    L, effective_delay = find_effective_delay(time, value, tangent, initial_value, step_time)

    # Calculate unit reaction rate
    R1 = calculate_unit_reaction_rate(R, setpoint_change_percent)

    return {
        'initial_value': initial_value,
        'final_value': final_value,
        'step_time': step_time,
        'step_idx': step_idx,
        'steepest_idx': steepest_idx,
        'steepest_time': time[steepest_idx],
        'steepest_value': value[steepest_idx],
        'steepest_smoothed': smoothed[steepest_idx],
        'max_slope': max_slope,
        'tangent': tangent,
        'R': R,
        'A': A,
        'B': B,
        't_start': t_start,
        't_end': t_end,
        'effective_delay': effective_delay,
        'L': L,
        'R1': R1,
        'setpoint_change_percent': setpoint_change_percent
    }


def plot_reaction_curve(time, value, results, save_path=None):
    """
    Plot the reaction curve with tangent line and construction lines.

    Shows:
    - Process reaction curve (blue)
    - Tangent at max slope (red dashed)
    - Line A: time interval on horizontal axis (green)
    - Line B: vertical rise showing PV change (magenta)
    """
    fig, ax = plt.subplots(figsize=(14, 9))

    # Get key values
    initial_value = results['initial_value']
    final_value = results['final_value']
    t_start = results['t_start']  # Where tangent crosses initial value
    t_end = results['t_end']      # Where tangent would reach final value
    B = results['B']
    step_time = results['step_time']
    effective_delay = results['effective_delay']
    steepest_idx = results['steepest_idx']
    max_slope = results['max_slope']

    # Plot the process reaction curve
    ax.plot(time, value, 'b-', linewidth=2.5, label='Process Reaction Curve', zorder=3)

    # Extend tangent line for visualization (from before t_start to after t_end)
    t_extended = np.linspace(min(time[0], t_start - 0.1), max(time[-1], t_end + 0.1), 500)
    y_at_steepest = value[steepest_idx]
    t_at_steepest = time[steepest_idx]
    tangent_extended = max_slope * (t_extended - t_at_steepest) + y_at_steepest
    ax.plot(t_extended, tangent_extended, 'r--', linewidth=2, label='Tangent at Max Slope', zorder=2)

    # Draw horizontal line at initial value (the baseline)
    ax.axhline(y=initial_value, color='gray', linestyle=':', linewidth=1, alpha=0.5)

    # Draw Line A: horizontal arrow from t_start to t_end at initial value level
    ax.annotate('', xy=(t_end, initial_value), xytext=(t_start, initial_value),
                arrowprops=dict(arrowstyle='<->', color='green', lw=2))
    ax.text((t_start + t_end) / 2, initial_value - 0.08, f'A = {results["A"]:.4f}',
            ha='center', fontsize=11, color='green', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='green', alpha=0.8))

    # Draw Line B: vertical arrow from initial value up to final value at t_end
    ax.annotate('', xy=(t_end, final_value), xytext=(t_end, initial_value),
                arrowprops=dict(arrowstyle='<->', color='magenta', lw=2))
    ax.text(t_end + 0.02, (initial_value + final_value) / 2, f'B = {B:.4f}',
            ha='left', fontsize=11, color='magenta', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='magenta', alpha=0.8))

    # Mark the inflection point (steepest slope)
    ax.plot(time[steepest_idx], value[steepest_idx], 'ro', markersize=12, zorder=5,
            label=f'Inflection Point ({time[steepest_idx]:.3f}, {value[steepest_idx]:.4f})')

    # Mark step change point
    ax.axvline(x=step_time, color='orange', linestyle='-.', linewidth=1.5, alpha=0.7)
    ax.scatter([step_time], [initial_value], color='orange', s=100, marker='v', zorder=5,
               label=f'Step Change (t={step_time:.4f})')

    # Mark effective delay point (where tangent crosses initial value)
    ax.axvline(x=effective_delay, color='purple', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.scatter([effective_delay], [initial_value], color='purple', s=100, marker='^', zorder=5,
               label=f'Tangent Crosses Initial (L={results["L"]:.4f})')

    # Mark tangent start and end points
    ax.scatter([t_start], [initial_value], color='green', s=150, marker='s', zorder=5)
    ax.scatter([t_end], [final_value], color='green', s=150, marker='s', zorder=5)

    # Labels and title
    ax.set_xlabel('Time', fontsize=13)
    ax.set_ylabel('Process Variable', fontsize=13)
    ax.set_title('Ziegler-Nichols Process Identification\n' +
                 f'Reaction Rate R = B/A = {results["R"]:.4f},  ' +
                 f'Unit Rate R1 = {results["R1"]:.4f},  Delay L = {results["L"]:.4f}',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Set axis limits to show all relevant parts
    ax.set_xlim(min(time[0], t_start - 0.05), max(time[-1], t_end + 0.1))
    y_margin = (final_value - initial_value) * 0.2
    ax.set_ylim(initial_value - y_margin, final_value + y_margin)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.show()

    return fig, ax


def print_results(results):
    """Print the ZN identification results in a formatted way."""
    print("\n" + "=" * 60)
    print("ZIEGLER-NICHOLS PROCESS IDENTIFICATION RESULTS")
    print("=" * 60)
    print(f"\nInitial Process Value:        {results['initial_value']:.6f}")
    print(f"Final Steady-State Value:     {results['final_value']:.6f}")
    print(f"Total Process Change:         {results['final_value'] - results['initial_value']:.6f}")
    print(f"\nStep Change Time:             {results['step_time']:.6f}")
    print(f"Effective Delay (L):          {results['L']:.6f}")
    print(f"  (time from step to tangent crossing initial value)")
    print(f"\nPoint of Maximum Slope (Inflection Point):")
    print(f"  Time:                       {results['steepest_time']:.6f}")
    print(f"  Process Value (raw):        {results['steepest_value']:.6f}")
    print(f"  Process Value (smoothed):   {results['steepest_smoothed']:.6f}")
    print(f"  Maximum Slope:              {results['max_slope']:.6f}")
    print(f"\nTangent Line Construction:")
    print(f"  Tangent crosses initial value at:  t = {results['t_start']:.6f}")
    print(f"  Tangent crosses final value at:    t = {results['t_end']:.6f}")
    print(f"\n  Line A (time interval):     {results['A']:.6f}  (from {results['t_start']:.4f} to {results['t_end']:.4f})")
    print(f"  Line B (PV change):         {results['B']:.6f}  (from {results['initial_value']:.4f} to {results['final_value']:.4f})")
    print(f"\nProcess Reaction Rate:")
    print(f"  R = B/A = {results['B']:.6f}/{results['A']:.6f} = {results['R']:.6f}")
    print(f"\nUnit Reaction Rate:")
    print(f"  R1 = R/X = {results['R']:.6f}/{results['setpoint_change_percent']} = {results['R1']:.6f}")
    print(f"\nSet Point Change (X):         {results['setpoint_change_percent']:.1f}%")
    print(f"  (assumed unit step input, X = {results['setpoint_change_percent']:.1f}%)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Ziegler-Nichols Process Identification Method',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ZN_method.py --data data/conjunto1.txt
  python ZN_method.py --data data/conjunto1.txt --setpoint 50
  python ZN_method.py --data data/conjunto1.txt --save plot.png
        """
    )
    parser.add_argument('--data', required=True, help='Path to data file (two columns: time,value)')
    parser.add_argument('--setpoint', type=float, default=100,
                        help='Set point change percentage (default: 100)')
    parser.add_argument('--save', help='Save plot to file instead of displaying')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting, only print results')

    args = parser.parse_args()

    # Load data
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: File not found: {data_path}")
        return 1

    print(f"Loading data from: {data_path}")
    time, value = load_data(data_path)
    print(f"Loaded {len(time)} data points")

    # Perform ZN identification
    results = zn_identification(time, value, args.setpoint)

    # Print results
    print_results(results)

    # Plot if requested
    if not args.no_plot:
        plot_reaction_curve(time, value, results, args.save)

    return 0


if __name__ == '__main__':
    exit(main())
