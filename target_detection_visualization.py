import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

class TargetDetectionVisualizer:
    def __init__(self, data_file):
        """Initialize the visualizer with data file"""
        self.data = pd.read_csv(data_file)
        self.data.columns = ['lat', 'lon', 'range_m']
        
        # Convert lat/lon to approximate meters for calculations
        # Rough conversion: 1 degree lat ≈ 111,000m, 1 degree lon ≈ 111,000m * cos(lat)
        self.lat_center = self.data['lat'].mean()
        self.lon_center = self.data['lon'].mean()
        
        # Convert to local coordinates (meters)
        self.data['x_m'] = (self.data['lon'] - self.lon_center) * 111000 * np.cos(np.radians(self.lat_center))
        self.data['y_m'] = (self.data['lat'] - self.lat_center) * 111000
        
        # Estimate target location using triangulation
        self.target_location = self.estimate_target_location()
        
        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        
    def estimate_target_location(self):
        """Estimate target location using least squares triangulation"""
        def objective_function(target_pos):
            x, y = target_pos
            predicted_ranges = np.sqrt((self.data['x_m'] - x)**2 + (self.data['y_m'] - y)**2)
            return np.sum((predicted_ranges - self.data['range_m'])**2)
        
        # Initial guess: centroid of boat positions
        initial_guess = [self.data['x_m'].mean(), self.data['y_m'].mean()]
        
        # Optimize to find target location
        result = minimize(objective_function, initial_guess, method='Nelder-Mead')
        
        if result.success:
            return result.x
        else:
            # Fallback to centroid if optimization fails
            return initial_guess
    
    def create_animation(self):
        """Create animated visualization"""
        self.ax.clear()
        
        # Set up the plot
        self.ax.set_xlabel('Longitude (relative to center)')
        self.ax.set_ylabel('Latitude (relative to center)')
        self.ax.set_title('Underwater Target Detection Visualization')
        self.ax.grid(True, alpha=0.3)
        
        # Plot boat positions with different markers and colors
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.data)))
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', '|']
        
        for i, (_, row) in enumerate(self.data.iterrows()):
            # Plot boat position
            self.ax.scatter(row['x_m'], row['y_m'], 
                          c=[colors[i]], 
                          marker=markers[i % len(markers)],
                          s=100, 
                          edgecolors='black',
                          linewidth=1,
                          label=f'Boat {i+1} (Range: {row["range_m"]:.1f}m)')
            
            # Create range circle
            circle = patches.Circle((row['x_m'], row['y_m']), 
                                  row['range_m'], 
                                  fill=False, 
                                  edgecolor=colors[i], 
                                  alpha=0.3, 
                                  linewidth=2,
                                  linestyle='--')
            self.ax.add_patch(circle)
        
        # Plot estimated target location
        target_x, target_y = self.target_location
        self.ax.scatter(target_x, target_y, 
                       c='red', 
                       marker='*', 
                       s=300, 
                       edgecolors='black',
                       linewidth=2,
                       label=f'Estimated Target\n({target_x:.1f}m, {target_y:.1f}m)')
        
        # Create confidence region (area where circles overlap)
        self.plot_confidence_region()
        
        # Add legend
        self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Set equal aspect ratio
        self.ax.set_aspect('equal')
        
        # Add text box with target information
        target_lat = self.lat_center + target_y / 111000
        target_lon = self.lon_center + target_x / (111000 * np.cos(np.radians(self.lat_center)))
        
        info_text = f'Target Location:\nLat: {target_lat:.6f}°\nLon: {target_lon:.6f}°\nRange from center: {np.sqrt(target_x**2 + target_y**2):.1f}m'
        self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        return self.ax,
    
    def plot_confidence_region(self):
        """Plot the confidence region where target is likely located"""
        # Create a grid of points
        x_min, x_max = self.data['x_m'].min() - 100, self.data['x_m'].max() + 100
        y_min, y_max = self.data['y_m'].min() - 100, self.data['y_m'].max() + 100
        
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        
        # Calculate how many range circles each point falls within
        confidence = np.zeros_like(xx)
        for _, row in self.data.iterrows():
            distance = np.sqrt((xx - row['x_m'])**2 + (yy - row['y_m'])**2)
            # Points within range get +1 confidence
            confidence += (distance <= row['range_m']).astype(float)
        
        # Normalize confidence (0 to 1)
        confidence = confidence / len(self.data)
        
        # Plot confidence region
        contour = self.ax.contourf(xx, yy, confidence, levels=10, alpha=0.3, cmap='Reds')
        self.ax.contour(xx, yy, confidence, levels=[0.5], colors='red', linewidths=2, alpha=0.7)
        
        # Add colorbar
        cbar = plt.colorbar(contour, ax=self.ax, shrink=0.8)
        cbar.set_label('Confidence Level (fraction of circles containing point)')
    
    def animate(self, frame):
        """Animation function for progressive reveal"""
        self.ax.clear()
        
        # Set up the plot
        self.ax.set_xlabel('Longitude (relative to center)')
        self.ax.set_ylabel('Latitude (relative to center)')
        self.ax.set_title(f'Underwater Target Detection Visualization - Step {frame+1}/{len(self.data)}')
        self.ax.grid(True, alpha=0.3)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.data)))
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', '|']
        
        # Plot only up to current frame
        for i in range(min(frame + 1, len(self.data))):
            row = self.data.iloc[i]
            
            # Plot boat position
            self.ax.scatter(row['x_m'], row['y_m'], 
                          c=[colors[i]], 
                          marker=markers[i % len(markers)],
                          s=100, 
                          edgecolors='black',
                          linewidth=1,
                          label=f'Boat {i+1} (Range: {row["range_m"]:.1f}m)')
            
            # Create range circle
            circle = patches.Circle((row['x_m'], row['y_m']), 
                                  row['range_m'], 
                                  fill=False, 
                                  edgecolor=colors[i], 
                                  alpha=0.3, 
                                  linewidth=2,
                                  linestyle='--')
            self.ax.add_patch(circle)
        
        # Show target estimate only after all points are plotted
        if frame >= len(self.data) - 1:
            target_x, target_y = self.target_location
            self.ax.scatter(target_x, target_y, 
                           c='red', 
                           marker='*', 
                           s=300, 
                           edgecolors='black',
                           linewidth=2,
                           label=f'Estimated Target\n({target_x:.1f}m, {target_y:.1f}m)')
            
            # Plot confidence region
            self.plot_confidence_region()
            
            # Add target information
            target_lat = self.lat_center + target_y / 111000
            target_lon = self.lon_center + target_x / (111000 * np.cos(np.radians(self.lat_center)))
            
            info_text = f'Target Location:\nLat: {target_lat:.6f}°\nLon: {target_lon:.6f}°\nRange from center: {np.sqrt(target_x**2 + target_y**2):.1f}m'
            self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        self.ax.set_aspect('equal')
        
        return self.ax,
    
    def create_static_plot(self):
        """Create static visualization"""
        self.create_animation()
        plt.show()
    
    def create_animated_plot(self, save_path=None):
        """Create animated visualization"""
        anim = FuncAnimation(self.fig, self.animate, frames=len(self.data), 
                           interval=1000, blit=False, repeat=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=1)
        
        plt.show()
        return anim

def main():
    """Main function to run the visualization"""
    # Create visualizer
    visualizer = TargetDetectionVisualizer('rawdata.csv')
    
    print("Target Detection Visualization")
    print("=" * 40)
    print(f"Number of data points: {len(visualizer.data)}")
    print(f"Data range: {visualizer.data['range_m'].min():.1f}m to {visualizer.data['range_m'].max():.1f}m")
    
    # Show target location estimate
    target_x, target_y = visualizer.target_location
    target_lat = visualizer.lat_center + target_y / 111000
    target_lon = visualizer.lon_center + target_x / (111000 * np.cos(np.radians(visualizer.lat_center)))
    
    print(f"\nEstimated Target Location:")
    print(f"Latitude: {target_lat:.6f}°")
    print(f"Longitude: {target_lon:.6f}°")
    print(f"Distance from center: {np.sqrt(target_x**2 + target_y**2):.1f}m")
    
    # Create static plot
    print("\nCreating static visualization...")
    visualizer.create_static_plot()
    
    # Create animated plot
    print("\nCreating animated visualization...")
    visualizer.create_animated_plot()

if __name__ == "__main__":
    main() 