import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import minimize
from scipy.stats import norm

def load_and_process_data(filename):
    """Load and process the CSV data"""
    data = pd.read_csv(filename)
agent    data.columns = ['lat', 'lon', 'range_m']
    
    # Convert to local coordinates (meters)
    lat_center = data['lat'].mean()
    lon_center = data['lon'].mean()
    
    data['x_m'] = (data['lon'] - lon_center) * 111000 * np.cos(np.radians(lat_center))
    data['y_m'] = (data['lat'] - lat_center) * 111000
    
    return data, lat_center, lon_center

def calculate_fuzzy_confidence(target_pos, boat_positions, ranges, uncertainties):
    """
    Calculate confidence using fuzzy hemispheres that account for measurement uncertainties.
    Each hemisphere has a "fuzzy" surface with uncertainty bands.
    """
    x, y, z = target_pos
    
    # Ensure target is underwater
    if z > 0:
        return 0.0, [0.0] * len(boat_positions)
    
    total_confidence = 0
    individual_confidences = []
    
    for i, (boat_x, boat_y) in enumerate(boat_positions):
        range_m = ranges[i]
        uncertainty = uncertainties[i]
        
        # Calculate 3D distance from boat to target
        distance_3d = np.sqrt((x - boat_x)**2 + (y - boat_y)**2 + z**2)
        
        # Calculate confidence based on how well target fits within the fuzzy hemisphere
        # Use a Gaussian-like confidence function
        error = abs(distance_3d - range_m)
        
        if error <= uncertainty:
            # Target is within the uncertainty band - high confidence
            confidence = np.exp(-(error / uncertainty)**2)
        else:
            # Target is outside uncertainty band - exponentially decreasing confidence
            confidence = np.exp(-(error / uncertainty)**2)
        
        individual_confidences.append(confidence)
        total_confidence += confidence
    
    # Normalize by number of measurements
    average_confidence = total_confidence / len(boat_positions)
    
    return average_confidence, individual_confidences

def estimate_measurement_uncertainties(data):
    """Estimate uncertainties for each measurement based on various error sources"""
    uncertainties = []
    
    for i, (_, row) in enumerate(data.iterrows()):
        range_m = row['range_m']
        
        # Base acoustic uncertainty (typically 1-5% of range)
        acoustic_uncertainty = range_m * 0.03  # 3% of range
        
        # GPS uncertainty (typically 3-10m for boat GPS)
        gps_uncertainty = 5.0  # meters
        
        # Drift uncertainty (boat movement during measurement)
        drift_uncertainty = 10.0  # meters
        
        # Averaging uncertainty (multiple measurements averaged)
        averaging_uncertainty = range_m * 0.01  # 1% of range
        
        # Environmental uncertainty (sound speed variations, etc.)
        environmental_uncertainty = range_m * 0.02  # 2% of range
        
        # Total uncertainty (root sum of squares)
        total_uncertainty = np.sqrt(
            acoustic_uncertainty**2 + 
            gps_uncertainty**2 + 
            drift_uncertainty**2 + 
            averaging_uncertainty**2 + 
            environmental_uncertainty**2
        )
        
        uncertainties.append(total_uncertainty)
        
        print(f"Point {i+1}: Range={range_m:.1f}m, Uncertainty={total_uncertainty:.1f}m "
              f"(Acoustic: {acoustic_uncertainty:.1f}m, GPS: {gps_uncertainty:.1f}m, "
              f"Drift: {drift_uncertainty:.1f}m, Avg: {averaging_uncertainty:.1f}m, "
              f"Env: {environmental_uncertainty:.1f}m)")
    
    return uncertainties

def find_fuzzy_intersection(data, uncertainties):
    """Find the best intersection point considering fuzzy hemispheres"""
    boat_positions = list(zip(data['x_m'], data['y_m']))
    ranges = data['range_m'].values
    
    def fuzzy_objective(target_pos):
        confidence, _ = calculate_fuzzy_confidence(target_pos, boat_positions, ranges, uncertainties)
        # Minimize negative confidence (maximize confidence)
        return -confidence
    
    # Try multiple starting points
    best_result = None
    best_confidence = -1
    
    # Try different depth estimates and positions
    depth_estimates = [-25, -50, -75, -100, -150, -200]
    
    for depth in depth_estimates:
        for x_offset in [-100, -50, 0, 50, 100]:
            for y_offset in [-100, -50, 0, 50, 100]:
                initial_guess = [data['x_m'].mean() + x_offset, 
                               data['y_m'].mean() + y_offset, 
                               depth]
                
                result = minimize(
                    fuzzy_objective, 
                    initial_guess, 
                    method='Nelder-Mead',
                    options={'maxiter': 2000}
                )
                
                if result.success:
                    confidence, _ = calculate_fuzzy_confidence(result.x, boat_positions, ranges, uncertainties)
                    if confidence > best_confidence:
                        best_result = result
                        best_confidence = confidence
    
    if best_result is None:
        # Fallback to simple approach
        return [data['x_m'].mean(), data['y_m'].mean(), -50], 0.0
    
    return best_result.x, best_confidence

def create_fuzzy_visualization(data, target_pos, uncertainties, confidence_score, individual_confidences):
    """Create visualization showing fuzzy hemisphere analysis"""
    fig = plt.figure(figsize=(18, 12))
    
    # 2D top view with fuzzy range circles
    ax1 = fig.add_subplot(231)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Fuzzy Range Circles (2D Projection)')
    ax1.grid(True, alpha=0.3)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(data)))
    
    for i, (_, row) in enumerate(data.iterrows()):
        range_m = row['range_m']
        uncertainty = uncertainties[i]
        confidence = individual_confidences[i]
        
        # Color based on confidence (red for low, green for high)
        if confidence > 0.8:
            color = 'green'
            alpha = 0.8
        elif confidence > 0.5:
            color = 'orange'
            alpha = 0.6
        else:
            color = 'red'
            alpha = 0.4
        
        ax1.scatter(row['x_m'], row['y_m'], 
                   c=[color], s=100, marker='o', alpha=alpha)
        
        # Draw fuzzy range circle (inner and outer bounds)
        inner_circle = patches.Circle((row['x_m'], row['y_m']), 
                                    range_m - uncertainty, 
                                    fill=False, 
                                    edgecolor=color, 
                                    alpha=alpha, 
                                    linewidth=1,
                                    linestyle=':')
        outer_circle = patches.Circle((row['x_m'], row['y_m']), 
                                    range_m + uncertainty, 
                                    fill=False, 
                                    edgecolor=color, 
                                    alpha=alpha, 
                                    linewidth=1,
                                    linestyle=':')
        
        ax1.add_patch(inner_circle)
        ax1.add_patch(outer_circle)
        
        # Draw nominal range circle
        nominal_circle = patches.Circle((row['x_m'], row['y_m']), 
                                      range_m, 
                                      fill=False, 
                                      edgecolor=color, 
                                      alpha=alpha, 
                                      linewidth=2)
        ax1.add_patch(nominal_circle)
    
    # Plot target location
    ax1.scatter(target_pos[0], target_pos[1], 
               c='red', s=300, marker='*', label='Target Estimate')
    ax1.legend()
    ax1.set_aspect('equal')
    
    # Confidence map
    ax2 = fig.add_subplot(232)
    x_min, x_max = data['x_m'].min() - 100, data['x_m'].max() + 100
    y_min, y_max = data['y_m'].min() - 100, data['y_m'].max() + 100
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                        np.linspace(y_min, y_max, 50))
    
    confidence_map = np.zeros_like(xx)
    boat_positions = list(zip(data['x_m'], data['y_m']))
    ranges = data['range_m'].values
    
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            point_x, point_y = xx[i,j], yy[i,j]
            point_z = target_pos[2]  # Same depth as target
            
            test_pos = [point_x, point_y, point_z]
            confidence, _ = calculate_fuzzy_confidence(test_pos, boat_positions, ranges, uncertainties)
            confidence_map[i,j] = confidence
    
    contour = ax2.contourf(xx, yy, confidence_map, levels=15, alpha=0.6, cmap='RdYlGn')
    ax2.contour(xx, yy, confidence_map, levels=[0.3, 0.5, 0.7, 0.9], colors='black', linewidths=1)
    ax2.scatter(target_pos[0], target_pos[1], c='red', s=200, marker='*')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title(f'Fuzzy Confidence Map (Depth: {target_pos[2]:.1f}m)')
    ax2.set_aspect('equal')
    
    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax2, shrink=0.8)
    cbar.set_label('Fuzzy Confidence')
    
    # Individual confidence scores
    ax3 = fig.add_subplot(233)
    measurement_points = range(len(data))
    
    # Color bars based on confidence
    bar_colors = []
    for conf in individual_confidences:
        if conf > 0.8:
            bar_colors.append('green')
        elif conf > 0.5:
            bar_colors.append('orange')
        else:
            bar_colors.append('red')
    
    bars = ax3.bar(measurement_points, individual_confidences, color=bar_colors, alpha=0.7)
    
    # Add threshold lines
    ax3.axhline(y=0.8, color='green', linestyle='--', label='High Confidence (0.8)')
    ax3.axhline(y=0.5, color='orange', linestyle='--', label='Moderate Confidence (0.5)')
    ax3.axhline(y=0.3, color='red', linestyle='--', label='Low Confidence (0.3)')
    
    ax3.set_xlabel('Measurement Point')
    ax3.set_ylabel('Individual Confidence')
    ax3.set_title('Fuzzy Confidence Scores')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    # Uncertainty vs Range analysis
    ax4 = fig.add_subplot(234)
    ranges = data['range_m'].values
    
    ax4.scatter(ranges, uncertainties, c=colors, s=100, alpha=0.7)
    ax4.set_xlabel('Range (m)')
    ax4.set_ylabel('Uncertainty (m)')
    ax4.set_title('Uncertainty vs Range')
    ax4.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(ranges, uncertainties, 1)
    p = np.poly1d(z)
    ax4.plot(ranges, p(ranges), "r--", alpha=0.8, label=f'Trend: {z[0]:.3f}x + {z[1]:.1f}')
    ax4.legend()
    
    # Error distribution
    ax5 = fig.add_subplot(235)
    boat_positions = list(zip(data['x_m'], data['y_m']))
    ranges = data['range_m'].values
    
    errors = []
    for i, (boat_x, boat_y) in enumerate(boat_positions):
        range_m = ranges[i]
        distance_3d = np.sqrt((target_pos[0] - boat_x)**2 + 
                             (target_pos[1] - boat_y)**2 + 
                             target_pos[2]**2)
        error = abs(distance_3d - range_m)
        errors.append(error)
    
    ax5.hist(errors, bins=10, alpha=0.7, color='blue', edgecolor='black')
    ax5.axvline(np.mean(errors), color='red', linestyle='--', 
                label=f'Mean: {np.mean(errors):.1f}m')
    ax5.set_xlabel('Range Error (m)')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Error Distribution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Summary statistics
    ax6 = fig.add_subplot(236)
    ax6.axis('off')
    
    # Convert target position to GPS coordinates
    lat_center = data['lat'].mean()
    lon_center = data['lon'].mean()
    target_lat = lat_center + target_pos[1] / 111000
    target_lon = lon_center + target_pos[0] / (111000 * np.cos(np.radians(lat_center)))
    
    high_conf_count = sum(1 for c in individual_confidences if c > 0.8)
    moderate_conf_count = sum(1 for c in individual_confidences if 0.5 < c <= 0.8)
    low_conf_count = sum(1 for c in individual_confidences if c <= 0.5)
    
    summary_text = f'FUZZY HEMISPHERE ANALYSIS RESULTS\n\n' \
                   f'Target Location:\n' \
                   f'• Latitude: {target_lat:.6f}°\n' \
                   f'• Longitude: {target_lon:.6f}°\n' \
                   f'• Depth: {target_pos[2]:.1f}m\n' \
                   f'• Horizontal distance: {np.sqrt(target_pos[0]**2 + target_pos[1]**2):.1f}m\n\n' \
                   f'Fuzzy Confidence:\n' \
                   f'• Overall confidence: {confidence_score:.1%}\n' \
                   f'• High confidence points: {high_conf_count}/{len(data)}\n' \
                   f'• Moderate confidence points: {moderate_conf_count}/{len(data)}\n' \
                   f'• Low confidence points: {low_conf_count}/{len(data)}\n\n' \
                   f'Uncertainty Analysis:\n' \
                   f'• Mean uncertainty: {np.mean(uncertainties):.1f}m\n' \
                   f'• Range: {np.min(uncertainties):.1f}m - {np.max(uncertainties):.1f}m\n' \
                   f'• Mean error: {np.mean(errors):.1f}m\n\n' \
                   f'Conclusion:\n' \
                   f'• {"High confidence" if confidence_score > 0.7 else "Moderate confidence" if confidence_score > 0.5 else "Low confidence"} in target location'
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, 
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9))
    
    plt.tight_layout()
    return fig

def print_fuzzy_analysis(data, target_pos, uncertainties, confidence_score, individual_confidences):
    """Print comprehensive fuzzy hemisphere analysis"""
    print("=" * 80)
    print("FUZZY HEMISPHERE ANALYSIS - REALISTIC UNCERTAINTY MODELING")
    print("=" * 80)
    
    print(f"Method: Each range measurement defines a 'fuzzy' hemisphere with")
    print(f"        uncertainty bands accounting for acoustic, GPS, drift, and")
    print(f"        environmental errors. Target location confidence is calculated")
    print(f"        based on how well it fits within these fuzzy hemispheres.\n")
    
    print(f"Data Summary:")
    print(f"  • Number of measurement points: {len(data)}")
    print(f"  • Range measurements: {data['range_m'].min():.1f}m to {data['range_m'].max():.1f}m")
    print(f"  • Average range: {data['range_m'].mean():.1f}m")
    
    # Convert target position to GPS coordinates
    lat_center = data['lat'].mean()
    lon_center = data['lon'].mean()
    target_lat = lat_center + target_pos[1] / 111000
    target_lon = lon_center + target_pos[0] / (111000 * np.cos(np.radians(lat_center)))
    
    print(f"\nTarget Location (Fuzzy Analysis):")
    print(f"  • Latitude:  {target_lat:.6f}°")
    print(f"  • Longitude: {target_lon:.6f}°")
    print(f"  • Depth: {target_pos[2]:.1f}m (below surface)")
    print(f"  • Horizontal distance from center: {np.sqrt(target_pos[0]**2 + target_pos[1]**2):.1f}m")
    
    print(f"\nUncertainty Analysis:")
    print(f"  • Mean uncertainty: {np.mean(uncertainties):.1f}m")
    print(f"  • Uncertainty range: {np.min(uncertainties):.1f}m to {np.max(uncertainties):.1f}m")
    print(f"  • Uncertainty as % of range: {np.mean(uncertainties) / data['range_m'].mean() * 100:.1f}%")
    
    print(f"\nFuzzy Confidence Results:")
    print(f"  • Overall confidence: {confidence_score:.1%}")
    
    # Categorize confidence levels
    high_conf = [i+1 for i, c in enumerate(individual_confidences) if c > 0.8]
    moderate_conf = [i+1 for i, c in enumerate(individual_confidences) if 0.5 < c <= 0.8]
    low_conf = [i+1 for i, c in enumerate(individual_confidences) if c <= 0.5]
    
    print(f"  • High confidence points (>80%): {high_conf}")
    print(f"  • Moderate confidence points (50-80%): {moderate_conf}")
    print(f"  • Low confidence points (<50%): {low_conf}")
    
    print(f"\nIndividual Point Analysis:")
    for i, (confidence, uncertainty) in enumerate(zip(individual_confidences, uncertainties)):
        range_m = data.iloc[i]['range_m']
        status = "HIGH" if confidence > 0.8 else "MODERATE" if confidence > 0.5 else "LOW"
        print(f"  Point {i+1}: {confidence:.1%} confidence, {uncertainty:.1f}m uncertainty, {range_m:.1f}m range ({status})")
    
    # Confidence assessment
    if confidence_score > 0.7:
        assessment = "HIGH - Target location is well-constrained by fuzzy hemispheres"
    elif confidence_score > 0.5:
        assessment = "MODERATE - Target location has reasonable constraint"
    elif confidence_score > 0.3:
        assessment = "LOW - Target location has significant uncertainty"
    else:
        assessment = "VERY LOW - Target location is poorly constrained"
    
    print(f"\nConfidence Assessment:")
    print(f"  • {assessment}")
    print(f"  • The fuzzy analysis accounts for realistic measurement uncertainties")
    print(f"  • This provides a more practical confidence estimate than rigid hemisphere surfaces")
    
    print("=" * 80)

def main():
    """Main function for fuzzy hemisphere analysis"""
    try:
        print("Loading data and estimating measurement uncertainties...")
        data, lat_center, lon_center = load_and_process_data('rawdata.csv')
        
        print("\nEstimating measurement uncertainties...")
        uncertainties = estimate_measurement_uncertainties(data)
        
        print("\nFinding fuzzy hemisphere intersection...")
        target_pos, confidence_score = find_fuzzy_intersection(data, uncertainties)
        
        print("\nCalculating individual confidence scores...")
        _, individual_confidences = calculate_fuzzy_confidence(target_pos, 
                                                             list(zip(data['x_m'], data['y_m'])), 
                                                             data['range_m'].values, 
                                                             uncertainties)
        
        print_fuzzy_analysis(data, target_pos, uncertainties, confidence_score, individual_confidences)
        
        print("\nCreating fuzzy hemisphere visualization...")
        fig = create_fuzzy_visualization(data, target_pos, uncertainties, confidence_score, individual_confidences)
        
        plt.show()
        print("\nFuzzy hemisphere analysis complete!")
        
    except FileNotFoundError:
        print("Error: rawdata.csv not found in current directory")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 