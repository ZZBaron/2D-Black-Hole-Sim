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
        self.G = 1.0
        self.c = 1.0
        self.M = mass
        self.rs = 2 * self.G * self.M / (self.c ** 2)

    def cartesian_to_polar(self, x, y, vx, vy):
        r = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)
        vphi = (x * vy - y * vx) / (r ** 2)
        vr = (x * vx + y * vy) / r
        f = 1 - self.rs / r
        pr = vr / f
        pphi = r ** 2 * vphi
        return r, phi, pr, pphi

    def geodesic_equations(self, t, state):
        r, phi, pr, pphi = state
        f = 1 - self.rs / r
        r_dot = f * pr
        phi_dot = pphi / (r ** 2)
        pr_dot = (pphi ** 2 / r ** 3) * f - (self.rs / (2 * r ** 2)) * f
        pphi_dot = 0
        return [r_dot, phi_dot, pr_dot, pphi_dot]

    def simulate_ray(self, x0, y0, vx0, vy0, t_span):
        v_magnitude = np.sqrt(vx0 ** 2 + vy0 ** 2)
        if v_magnitude > 0:
            vx0 = vx0 * self.c / v_magnitude
            vy0 = vy0 * self.c / v_magnitude
        r0, phi0, pr0, pphi0 = self.cartesian_to_polar(x0, y0, vx0, vy0)
        solution = solve_ivp(
            self.geodesic_equations,
            (0, t_span),
            [r0, phi0, pr0, pphi0],
            method='RK45',
            rtol=1e-8,
            atol=1e-8,
            max_step=0.1,
            dense_output=True
        )
        t_eval = np.linspace(0, t_span, 1000)
        return solution.sol(t_eval).T


class BlackHoleAnimation:
    def __init__(self):
        self.bh = BlackHoleSimulation(mass=1.0)
        self.light_mode = 'source'
        self.source_x = -15
        self.source_y = 0
        self.source_angle = 0
        self.sweep_angle = 180
        self.direction_angle = 0
        self.start_distance = 15
        self.resolution = 2.0
        self.is_animating = False
        self.lines = []
        self.ray_data = []
        self.setup_figure()
        self.setup_controls()

    def setup_figure(self):
        self.fig = plt.figure(figsize=(15, 10))
        self.ax = plt.subplot2grid((4, 4), (0, 0), colspan=3, rowspan=4)
        self.ax.set_xlim(-20, 20)
        self.ax.set_ylim(-20, 20)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title('Interactive Black Hole Light Ray Simulation')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.add_artist(plt.Circle((0, 0), self.bh.rs, color='black', zorder=10))

    def _make_setter(self, attr):
        def setter(val):
            setattr(self, attr, val)
        return setter

    def _clear_rays(self):
        for line in self.lines:
            line.remove()
        self.lines = []
        self.ray_data = []
        if hasattr(self, 'source_point'):
            self.source_point.remove()
            del self.source_point
        if self.is_animating:
            self.stop_animation()

    def _plot_ray(self, ray, color):
        x = ray[:, 0] * np.cos(ray[:, 1])
        y = ray[:, 0] * np.sin(ray[:, 1])
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]
        absorbed = np.where(np.sqrt(x ** 2 + y ** 2) <= self.bh.rs)[0]
        if len(absorbed):
            x, y = x[:absorbed[0]], y[:absorbed[0]]
        if len(x) > 0:
            line, = self.ax.plot(x, y, color, lw=1, alpha=0.7)
            self.lines.append(line)
            self.ray_data.append((x, y))

    def _tk_dialog(self, kind, title, message):
        root = tk.Tk()
        root.withdraw()
        result = getattr(messagebox, kind)(title, message)
        root.destroy()
        return result

    def setup_controls(self):
        control_ax = plt.subplot2grid((4, 4), (0, 3), rowspan=4)
        control_ax.axis('off')

        mode_ax = plt.axes([0.77, 0.85, 0.15, 0.1])
        self.mode_radio = RadioButtons(mode_ax, ('Source', 'Directional'))
        self.mode_radio.on_clicked(self.set_mode)


        self.ax_src_x = plt.axes([0.77, 0.75, 0.15, 0.03])
        self.source_x_slider = Slider(self.ax_src_x, 'Source X', -20, 20, valinit=self.source_x)
        self.source_x_slider.on_changed(self._make_setter('source_x'))

        self.ax_src_y = plt.axes([0.77, 0.70, 0.15, 0.03])
        self.source_y_slider = Slider(self.ax_src_y, 'Source Y', -20, 20, valinit=self.source_y)
        self.source_y_slider.on_changed(self._make_setter('source_y'))

        self.ax_src_angle = plt.axes([0.77, 0.65, 0.15, 0.03])
        self.source_angle_slider = Slider(self.ax_src_angle, 'Source Angle', 0, 360, valinit=self.source_angle)
        self.source_angle_slider.on_changed(self._make_setter('source_angle'))

        self.ax_sweep = plt.axes([0.77, 0.60, 0.15, 0.03])
        self.sweep_angle_slider = Slider(self.ax_sweep, 'Sweep Angle', 10, 360, valinit=self.sweep_angle)
        self.sweep_angle_slider.on_changed(self._make_setter('sweep_angle'))

        self.ax_dir = plt.axes([0.77, 0.60, 0.15, 0.03])
        self.direction_angle_slider = Slider(self.ax_dir, 'Direction', 0, 360, valinit=self.direction_angle)
        self.direction_angle_slider.on_changed(self._make_setter('direction_angle'))
        self.ax_dir.set_visible(False)

        self.ax_res = plt.axes([0.77, 0.55, 0.15, 0.03])
        self.resolution_slider = Slider(self.ax_res, 'Resolution', 0.5, 5.0, valinit=self.resolution)
        self.resolution_slider.on_changed(self._make_setter('resolution'))

        self.ax_dist = plt.axes([0.77, 0.50, 0.15, 0.03])
        self.distance_slider = Slider(self.ax_dist, 'Start Distance', 5, 30, valinit=self.start_distance)
        self.distance_slider.on_changed(self._make_setter('start_distance'))

        self._source_axes = [self.ax_src_x, self.ax_src_y, self.ax_src_angle, self.ax_sweep]
        self._dir_axes = [self.ax_dir]

        self.regenerate_button = Button(plt.axes([0.77, 0.40, 0.15, 0.05]), 'Generate')
        self.regenerate_button.on_clicked(self.generate_rays)

        self.animate_button = Button(plt.axes([0.77, 0.34, 0.15, 0.05]), 'Animate')
        self.animate_button.on_clicked(
            lambda e: self.stop_animation() if self.is_animating else self.start_animation()
        )

        self.save_animation_button = Button(plt.axes([0.77, 0.28, 0.15, 0.05]), 'Save Animation')
        self.save_animation_button.on_clicked(self.save_animation)

        self.reset_button = Button(plt.axes([0.77, 0.22, 0.15, 0.05]), 'Reset')
        self.reset_button.on_clicked(self.reset_view)

    def set_mode(self, label):
        self.light_mode = label.lower()
        self._clear_rays()
        is_source = self.light_mode == 'source'
        for ax in self._source_axes:
            ax.set_visible(is_source)
        for ax in self._dir_axes:
            ax.set_visible(not is_source)
        self.fig.canvas.draw()

    def generate_rays(self, event=None):
        self._clear_rays()
        if self.light_mode == 'source':
            self.generate_source_rays()
        else:
            self.generate_directional_rays()
        self.fig.canvas.draw()

    def generate_source_rays(self):
        n_rays = max(1, int(self.sweep_angle * self.resolution / 10))
        start_angle = self.source_angle - self.sweep_angle / 2
        end_angle = self.source_angle + self.sweep_angle / 2
        for angle in np.linspace(start_angle, end_angle, n_rays):
            angle_rad = np.radians(angle)
            vx0, vy0 = np.cos(angle_rad), np.sin(angle_rad)
            try:
                ray = self.bh.simulate_ray(self.source_x, self.source_y, vx0, vy0, 50.0)
                self._plot_ray(ray, 'y-')
            except Exception:
                continue
        if hasattr(self, 'source_point'):
            self.source_point.remove()
        self.source_point = self.ax.plot(self.source_x, self.source_y, 'ro', markersize=8, zorder=15)[0]

    def generate_directional_rays(self):
        direction_rad = np.radians(self.direction_angle)
        vx0, vy0 = np.cos(direction_rad), np.sin(direction_rad)
        perp_x, perp_y = -np.sin(direction_rad), np.cos(direction_rad)
        center_x = -self.start_distance * vx0
        center_y = -self.start_distance * vy0
        ray_spacing = 1.0 / self.resolution
        n_rays = int(40 / ray_spacing)
        for i in range(n_rays):
            offset = (i - n_rays // 2) * ray_spacing
            x0 = center_x + offset * perp_x
            y0 = center_y + offset * perp_y
            if np.sqrt(x0 ** 2 + y0 ** 2) < self.bh.rs * 2:
                continue
            try:
                ray = self.bh.simulate_ray(x0, y0, vx0, vy0, 50.0)
                self._plot_ray(ray, 'c-')
            except Exception:
                continue

    def start_animation(self):
        self.is_animating = True
        self.animate_button.label.set_text('Stop')
        for line in self.lines:
            line.set_data([], [])
        self.anim = FuncAnimation(
            self.fig, self.animate_frame,
            frames=200, interval=50, blit=False, repeat=True
        )
        self.fig.canvas.draw()

    def stop_animation(self):
        self.is_animating = False
        self.animate_button.label.set_text('Animate')
        if hasattr(self, 'anim'):
            self.anim.event_source.stop()

    def animate_frame(self, frame):
        frame_length = int((frame + 1) * 1000 / 200)
        for line, (x, y) in zip(self.lines, self.ray_data):
            if len(x) > 0:
                end_idx = min(frame_length, len(x))
                line.set_data(x[:end_idx], y[:end_idx])
        return self.lines

    def reset_view(self, event=None):
        self._clear_rays()
        self.ax.set_xlim(-20, 20)
        self.ax.set_ylim(-20, 20)
        self.fig.canvas.draw()

    def save_animation(self, event=None):
        if not hasattr(self, 'anim'):
            self._tk_dialog('showwarning', 'No Animation',
                            'No animation is currently running. Please start an animation first.')
            return

        if not self._tk_dialog('askyesno', 'Save Animation Warning',
                               'Warning: Saving the animation may take a minute. '
                               'The program may appear frozen during this process. Continue?'):
            return

        try:
            print("Saving animation... This may take several minutes. Please wait.")
            save_fig, save_ax = plt.subplots(figsize=(10, 10))
            save_ax.set_xlim(self.ax.get_xlim())
            save_ax.set_ylim(self.ax.get_ylim())
            save_ax.set_aspect('equal')
            save_ax.grid(True, alpha=0.3)
            save_ax.set_title('Black Hole Light Ray Simulation')
            save_ax.set_xlabel('x')
            save_ax.set_ylabel('y')
            save_ax.add_artist(plt.Circle((0, 0), self.bh.rs, color='black', zorder=10))
            color = 'y-' if self.light_mode == 'source' else 'c-'
            save_lines = [save_ax.plot([], [], color, lw=1, alpha=0.7)[0] for _ in self.ray_data]
            if self.light_mode == 'source':
                save_ax.plot(self.source_x, self.source_y, 'ro', markersize=8, zorder=15)

            def animate_save_frame(frame):
                frame_length = int((frame + 1) * 1000 / 200)
                for line, (x, y) in zip(save_lines, self.ray_data):
                    if len(x) > 0:
                        line.set_data(x[:min(frame_length, len(x))], y[:min(frame_length, len(y))])
                return save_lines

            save_anim = FuncAnimation(save_fig, animate_save_frame,
                                      frames=200, interval=50, blit=False, repeat=True)
            save_anim.save('blackhole_animation.gif', fps=20, dpi=150, writer='pillow')
            plt.close(save_fig)
            del save_anim
            print("Animation saved successfully as 'blackhole_animation.gif'")
            self._tk_dialog('showinfo', 'Success',
                            "Animation saved successfully as 'blackhole_animation.gif'")
        except Exception as e:
            print(f"Error saving animation: {e}")
            self._tk_dialog('showerror', 'Error', f'Failed to save animation: {str(e)}')

    def show(self):
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    sim = BlackHoleAnimation()
    sim.show()