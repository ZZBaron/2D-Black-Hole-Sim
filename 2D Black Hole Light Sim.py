import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider, RadioButtons
from scipy.integrate import solve_ivp
import tkinter as tk
from tkinter import messagebox


class BlackHoleSimulation:
    def __init__(self, mass=1.0):
        self.G = 1.0  # Gravitational constant (normalized units)
        self.c = 1.0  # Speed of light (normalized units)
        self.M = mass  # Black hole mass
        self.rs = 2 * self.G * self.M / (self.c ** 2)  # Schwarzschild radius

    def cartesian_to_polar(self, x, y, vx, vy):
        """Convert Cartesian to polar coordinates with proper metric components"""
        r = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)

        # Calculate intermediate velocities
        vphi = (x * vy - y * vx) / (r ** 2)
        vr = (x * vx + y * vy) / r

        # Include proper metric components for null geodesics
        f = 1 - self.rs / r
        pr = vr / f
        pphi = r ** 2 * vphi

        return r, phi, pr, pphi

    def geodesic_equations(self, t, state):
        """
        Implements the geodesic equations in Schwarzschild geometry
        state = [r, phi, pr, pphi]
        """
        r, phi, pr, pphi = state

        # Safety check for numerical stability
        if r <= self.rs * 1.1:  # Increased safety margin
            return [0, 0, 0, 0]

        f = 1 - self.rs / r

        # Improved geodesic equations with better numerical stability
        r_dot = f * pr
        phi_dot = pphi / (r ** 2)
        pr_dot = (pphi ** 2 / r ** 3) * f - (self.rs / (2 * r ** 2)) * f
        pphi_dot = 0  # Angular momentum conservation

        return [r_dot, phi_dot, pr_dot, pphi_dot]

    def simulate_ray(self, x0, y0, vx0, vy0, t_span):
        """Simulate a single light ray with improved numerical integration"""
        # Normalize initial velocity to speed of light
        v_magnitude = np.sqrt(vx0 ** 2 + vy0 ** 2)
        if v_magnitude > 0:
            vx0 = vx0 * self.c / v_magnitude
            vy0 = vy0 * self.c / v_magnitude

        # Convert to polar coordinates
        r0, phi0, pr0, pphi0 = self.cartesian_to_polar(x0, y0, vx0, vy0)

        # Set up initial conditions
        initial_state = [r0, phi0, pr0, pphi0]

        # Use solve_ivp with better error control
        solution = solve_ivp(
            self.geodesic_equations,
            (0, t_span),
            initial_state,
            method='RK45',
            rtol=1e-8,
            atol=1e-8,
            max_step=0.1,
            dense_output=True
        )

        # Get evenly spaced points for smooth visualization
        t_eval = np.linspace(0, t_span, 1000)
        solution_dense = solution.sol(t_eval)

        return solution_dense.T


# interactive plt figure wrapper
class BlackHoleAnimation:
    def __init__(self):
        self.bh = BlackHoleSimulation(mass=1.0)

        # Animation parameters
        self.light_mode = 'source'  # 'source' or 'directional'
        self.source_x = -15
        self.source_y = 0
        self.source_angle = 0  # degrees
        self.sweep_angle = 180  # degrees
        self.direction_angle = 0  # degrees for directional mode
        self.start_distance = 15
        self.resolution = 2.0  # rays per unit
        self.is_animating = False

        self.setup_figure()
        self.setup_controls()
        # Don't auto-generate rays on startup - wait for user interaction

    def setup_figure(self):
        """Set up the main figure and axis"""
        self.fig = plt.figure(figsize=(15, 10))

        # Main plot area
        self.ax = plt.subplot2grid((4, 4), (0, 0), colspan=3, rowspan=4)
        self.ax.set_xlim(-20, 20)
        self.ax.set_ylim(-20, 20)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title('Interactive Black Hole Light Ray Simulation')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')

        # Draw black hole
        self.black_hole_circle = plt.Circle((0, 0), self.bh.rs, color='black', zorder=10)
        self.ax.add_artist(self.black_hole_circle)

        # Draw event horizon (slightly larger for visibility)
        self.event_horizon = plt.Circle((0, 0), self.bh.rs * 1.1,
                                        color='red', fill=False,
                                        linestyle='--', alpha=0.5, zorder=9)
        self.ax.add_artist(self.event_horizon)

    def setup_controls(self):
        """Set up interactive controls"""
        # Control panel area
        control_ax = plt.subplot2grid((4, 4), (0, 3), rowspan=4)
        control_ax.set_xlim(0, 1)
        control_ax.set_ylim(0, 1)
        control_ax.axis('off')

        # Mode selection
        mode_ax = plt.axes([0.77, 0.85, 0.15, 0.1])
        self.mode_radio = RadioButtons(mode_ax, ('source', 'directional'))
        self.mode_radio.on_clicked(self.set_mode)

        # Source position sliders
        self.source_x_slider = Slider(
            plt.axes([0.77, 0.75, 0.15, 0.03]),
            'Source X', -20, 20, valinit=self.source_x
        )
        self.source_x_slider.on_changed(self.update_source_x)

        self.source_y_slider = Slider(
            plt.axes([0.77, 0.70, 0.15, 0.03]),
            'Source Y', -20, 20, valinit=self.source_y
        )
        self.source_y_slider.on_changed(self.update_source_y)

        # Angle sliders
        self.source_angle_slider = Slider(
            plt.axes([0.77, 0.65, 0.15, 0.03]),
            'Source Angle', 0, 360, valinit=self.source_angle
        )
        self.source_angle_slider.on_changed(self.update_source_angle)

        self.sweep_angle_slider = Slider(
            plt.axes([0.77, 0.60, 0.15, 0.03]),
            'Sweep Angle', 10, 360, valinit=self.sweep_angle
        )
        self.sweep_angle_slider.on_changed(self.update_sweep_angle)

        self.direction_angle_slider = Slider(
            plt.axes([0.77, 0.55, 0.15, 0.03]),
            'Direction', 0, 360, valinit=self.direction_angle
        )
        self.direction_angle_slider.on_changed(self.update_direction_angle)

        # Resolution slider
        self.resolution_slider = Slider(
            plt.axes([0.77, 0.50, 0.15, 0.03]),
            'Resolution', 0.5, 5.0, valinit=self.resolution
        )
        self.resolution_slider.on_changed(self.update_resolution)

        # Distance slider
        self.distance_slider = Slider(
            plt.axes([0.77, 0.45, 0.15, 0.03]),
            'Start Distance', 5, 30, valinit=self.start_distance
        )
        self.distance_slider.on_changed(self.update_distance)

        # Control buttons
        self.regenerate_button = Button(plt.axes([0.77, 0.35, 0.15, 0.05]), 'Generate')
        self.regenerate_button.on_clicked(self.regenerate_rays)

        self.animate_button = Button(plt.axes([0.77, 0.30, 0.15, 0.05]), 'Animate')
        self.animate_button.on_clicked(self.toggle_animation)

        self.save_animation_button = Button(plt.axes([0.77, 0.25, 0.15, 0.05]), 'Save Animation')
        self.save_animation_button.on_clicked(self.save_animation)

        self.reset_button = Button(plt.axes([0.77, 0.20, 0.15, 0.05]), 'Reset')
        self.reset_button.on_clicked(self.reset_view)

    def set_mode(self, label):
        """Set the light generation mode"""
        self.light_mode = label
        # Don't auto-generate rays

    def update_source_x(self, val):
        self.source_x = val
        # Don't auto-generate rays

    def update_source_y(self, val):
        self.source_y = val
        # Don't auto-generate rays

    def update_source_angle(self, val):
        self.source_angle = val
        # Don't auto-generate rays

    def update_sweep_angle(self, val):
        self.sweep_angle = val
        # Don't auto-generate rays

    def update_direction_angle(self, val):
        self.direction_angle = val
        # Don't auto-generate rays

    def update_resolution(self, val):
        self.resolution = val
        # Don't auto-generate rays

    def update_distance(self, val):
        self.start_distance = val
        # Don't auto-generate rays

    def generate_rays(self):
        """Generate light rays based on current settings"""
        # Clear existing rays
        for line in getattr(self, 'lines', []):
            line.remove()

        self.lines = []
        self.ray_data = []

        if self.light_mode == 'source':
            self.generate_source_rays()
        else:
            self.generate_directional_rays()

        self.fig.canvas.draw()

    def generate_source_rays(self):
        """Generate rays from a point source with sweep angle"""
        # Calculate number of rays based on sweep angle and resolution
        n_rays = max(1, int(self.sweep_angle * self.resolution / 10))

        # Generate angles for the sweep
        start_angle = self.source_angle - self.sweep_angle / 2
        end_angle = self.source_angle + self.sweep_angle / 2
        angles = np.linspace(start_angle, end_angle, n_rays)

        for angle in angles:
            # Convert angle to radians
            angle_rad = np.radians(angle)

            # Initial velocity components
            vx0 = np.cos(angle_rad)
            vy0 = np.sin(angle_rad)

            try:
                ray = self.bh.simulate_ray(self.source_x, self.source_y, vx0, vy0, 50.0)

                # Convert solution back to Cartesian coordinates
                x = ray[:, 0] * np.cos(ray[:, 1])
                y = ray[:, 0] * np.sin(ray[:, 1])

                # Filter out any NaN values
                mask = ~(np.isnan(x) | np.isnan(y))
                x = x[mask]
                y = y[mask]

                if len(x) > 0:
                    line, = self.ax.plot(x, y, 'y-', lw=1, alpha=0.7)
                    self.lines.append(line)
                    self.ray_data.append((x, y))
            except:
                continue

        # Draw source point
        if hasattr(self, 'source_point'):
            self.source_point.remove()
        self.source_point = self.ax.plot(self.source_x, self.source_y, 'ro', markersize=8, zorder=15)[0]

    def generate_directional_rays(self):
        """Generate parallel rays all traveling in the same direction"""
        # Direction vector
        direction_rad = np.radians(self.direction_angle)
        vx0 = np.cos(direction_rad)
        vy0 = np.sin(direction_rad)

        # Create perpendicular vector for spacing rays
        perp_x = -np.sin(direction_rad)
        perp_y = np.cos(direction_rad)

        # Calculate starting line perpendicular to direction
        center_x = -self.start_distance * vx0
        center_y = -self.start_distance * vy0

        # Calculate number of rays based on resolution
        ray_spacing = 1.0 / self.resolution
        n_rays = int(40 / ray_spacing)  # Cover 40 units width

        for i in range(n_rays):
            offset = (i - n_rays // 2) * ray_spacing

            # Starting position
            x0 = center_x + offset * perp_x
            y0 = center_y + offset * perp_y

            # Skip if too close to black hole
            if np.sqrt(x0 ** 2 + y0 ** 2) < self.bh.rs * 2:
                continue

            try:
                ray = self.bh.simulate_ray(x0, y0, vx0, vy0, 50.0)

                # Convert solution back to Cartesian coordinates
                x = ray[:, 0] * np.cos(ray[:, 1])
                y = ray[:, 0] * np.sin(ray[:, 1])

                # Filter out any NaN values
                mask = ~(np.isnan(x) | np.isnan(y))
                x = x[mask]
                y = y[mask]

                if len(x) > 0:
                    line, = self.ax.plot(x, y, 'c-', lw=1, alpha=0.7)
                    self.lines.append(line)
                    self.ray_data.append((x, y))
            except:
                continue

        # Remove source point if it exists
        if hasattr(self, 'source_point'):
            self.source_point.remove()
            del self.source_point

    def toggle_animation(self, event):
        """Toggle animation on/off"""
        if not self.is_animating:
            self.start_animation()
        else:
            self.stop_animation()

    def start_animation(self):
        """Start the animation"""
        self.is_animating = True
        self.animate_button.label.set_text('Stop')

        # Clear existing rays for animation
        for line in self.lines:
            line.set_data([], [])

        # Create animation
        self.anim = FuncAnimation(
            self.fig,
            self.animate_frame,
            frames=200,
            interval=50,
            blit=False,
            repeat=True
        )

        self.fig.canvas.draw()

    def stop_animation(self):
        """Stop the animation"""
        self.is_animating = False
        self.animate_button.label.set_text('Animate')

        if hasattr(self, 'anim'):
            self.anim.event_source.stop()

        # Redraw all rays
        self.generate_rays()

    def animate_frame(self, frame):
        """Animation frame function"""
        frame_length = int((frame + 1) * 1000 / 200)

        for line, (x, y) in zip(self.lines, self.ray_data):
            if len(x) > 0:
                end_idx = min(frame_length, len(x))
                line.set_data(x[:end_idx], y[:end_idx])

        return self.lines

    def reset_view(self, event):
        """Reset view to default"""
        self.ax.set_xlim(-20, 20)
        self.ax.set_ylim(-20, 20)
        self.fig.canvas.draw()

    def regenerate_rays(self, event):
        """Regenerate rays with current settings"""
        self.generate_rays()

    def save_animation(self, event):
        """Save animation with warning and plot-only capture"""
        if not hasattr(self, 'anim'):
            # Show warning dialog
            root = tk.Tk()
            root.withdraw()  # Hide the main window
            messagebox.showwarning("No Animation",
                                   "No animation is currently running. Please start an animation first.")
            root.destroy()
            return

        # Show warning dialog about slow saving
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        result = messagebox.askyesno(
            "Save Animation Warning",
            "Warning: Saving the animation may take several minutes and will use significant disk space. "
            "The program may appear frozen during this process. Continue?"
        )
        root.destroy()

        if not result:
            return

        try:
            print("Saving animation... This may take several minutes. Please wait.")

            # Create a new figure with just the plot for saving
            save_fig = plt.figure(figsize=(10, 10))
            save_ax = save_fig.add_subplot(111)

            # Copy the plot settings
            save_ax.set_xlim(self.ax.get_xlim())
            save_ax.set_ylim(self.ax.get_ylim())
            save_ax.set_aspect('equal')
            save_ax.grid(True, alpha=0.3)
            save_ax.set_title('Black Hole Light Ray Simulation')
            save_ax.set_xlabel('x')
            save_ax.set_ylabel('y')

            # Draw black hole and event horizon
            black_hole_circle = plt.Circle((0, 0), self.bh.rs, color='black', zorder=10)
            save_ax.add_artist(black_hole_circle)

            event_horizon = plt.Circle((0, 0), self.bh.rs * 1.1,
                                       color='red', fill=False,
                                       linestyle='--', alpha=0.5, zorder=9)
            save_ax.add_artist(event_horizon)

            # Create lines for animation
            save_lines = []
            for x, y in self.ray_data:
                if self.light_mode == 'source':
                    line, = save_ax.plot([], [], 'y-', lw=1, alpha=0.7)
                else:
                    line, = save_ax.plot([], [], 'c-', lw=1, alpha=0.7)
                save_lines.append(line)

            # Add source point if in source mode
            if self.light_mode == 'source':
                save_ax.plot(self.source_x, self.source_y, 'ro', markersize=8, zorder=15)

            def animate_save_frame(frame):
                frame_length = int((frame + 1) * 1000 / 200)
                for line, (x, y) in zip(save_lines, self.ray_data):
                    if len(x) > 0:
                        end_idx = min(frame_length, len(x))
                        line.set_data(x[:end_idx], y[:end_idx])
                return save_lines

            # Create and save animation
            save_anim = FuncAnimation(
                save_fig,
                animate_save_frame,
                frames=200,
                interval=50,
                blit=False,
                repeat=True
            )

            # Save with high quality settings
            save_anim.save('blackhole_animation.gif', fps=20, dpi=150, writer='pillow')

            # Clean up
            plt.close(save_fig)
            del save_anim

            print("Animation saved successfully as 'blackhole_animation.gif'")

            # Show success dialog
            root = tk.Tk()
            root.withdraw()
            messagebox.showinfo("Success", "Animation saved successfully as 'blackhole_animation.gif'")
            root.destroy()

        except Exception as e:
            print(f"Error saving animation: {e}")
            # Show error dialog
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Error", f"Failed to save animation: {str(e)}")
            root.destroy()

    def show(self):
        """Display the interactive simulation"""
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Create and show the interactive simulation
    sim = BlackHoleAnimation()
    sim.show()