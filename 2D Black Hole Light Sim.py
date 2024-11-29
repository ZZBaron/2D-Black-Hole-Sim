import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp


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


class BlackHoleAnimation:
    def __init__(self):
        self.bh = BlackHoleSimulation(mass=1.0)

        # Set up the figure
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_xlim(-20, 20)
        self.ax.set_ylim(-20, 20)
        self.ax.set_aspect('equal')
        self.ax.grid(False)

        # Draw black hole
        circle = plt.Circle((0, 0), self.bh.rs, color='black')
        self.ax.add_artist(circle)

        # Initialize light rays
        self.n_rays = 100
        self.rays = []
        self.lines = []
        self.ray_data = []

        # Improved initial conditions for light rays
        x0 = -15  # Start further out
        min_height = -5
        max_height = 15

        for i in range(self.n_rays):
            y0 = min_height + ((max_height - min_height) * i) / (self.n_rays - 1)

            # Initial velocity (pointing right)
            vx0 = 1.0
            vy0 = 0.0

            try:
                ray = self.bh.simulate_ray(x0, y0, vx0, vy0, 50.0)  # Shorter simulation time

                # Convert solution back to Cartesian coordinates
                x = ray[:, 0] * np.cos(ray[:, 1])
                y = ray[:, 0] * np.sin(ray[:, 1])

                # Filter out any NaN values
                mask = ~(np.isnan(x) | np.isnan(y))
                x = x[mask]
                y = y[mask]

                line, = self.ax.plot([], [], 'y-', lw=1, alpha=0.5)
                self.lines.append(line)
                self.ray_data.append((x, y))
            except:
                continue

        plt.title('Light Rays Around a Black Hole')
        plt.xlabel('x')
        plt.ylabel('y')

    def init(self):
        for line in self.lines:
            line.set_data([], [])
        return self.lines

    def animate(self, frame):
        frame_length = int((frame + 1) * 1000 / 200)  # Slower animation
        for line, (x, y) in zip(self.lines, self.ray_data):
            line.set_data(x[:frame_length], y[:frame_length])
        return self.lines

    def create_animation(self):
        anim = FuncAnimation(
            self.fig,
            self.animate,
            init_func=self.init,
            frames=200,
            interval=20,
            blit=True
        )
        return anim


if __name__ == "__main__":
    bh_anim = BlackHoleAnimation()
    anim = bh_anim.create_animation()
    plt.show()