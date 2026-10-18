import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle


def pedir_parametros():
    """Permite al usuario introducir los parámetros del sistema de forma interactiva."""
    print("==================================================")
    print("  SIMULACIÓN: PÉNDULO ARTICULADO SOBRE BASE MÓVIL ")
    print("==================================================")
    print("Introduce los valores o pulsa [Enter] para usar los predeterminados:\n")
    
    try:
        m1 = float(input("Masa de la base m1 (kg) [por defecto 2.0]: ") or 2.0)
        m2 = float(input("Masa del péndulo m2 (kg) [por defecto 1.0]: ") or 1.0)
        k  = float(input("Constante del resorte k (N/m) [por defecto 20.0]: ") or 20.0)
        l0 = float(input("Longitud natural del resorte l0 (m) [por defecto 1.0]: ") or 1.0)
        l  = float(input("Longitud del péndulo l (m) [por defecto 0.8]: ") or 0.8)
        g  = float(input("Aceleración de la gravedad g (m/s^2) [por defecto 9.81]: ") or 9.81)
    except ValueError:
        print("\n[Aviso] Entrada no válida detectada. Se usarán los valores por defecto.")
        m1, m2, k, l0, l, g = 2.0, 1.0, 20.0, 1.0, 0.8, 9.81
        
    return {'m1': m1, 'm2': m2, 'k': k, 'l0': l0, 'l': l, 'g': g}


def edom_sistema(t, y, p):
    """Define las ecuaciones de movimiento acopladas del sistema."""
    x_val, dx_val, th_val, dth_val = y

    # Matriz de masa M * [ddx; ddtheta] = F
    M = np.array([
        [p['m1'] + p['m2'], p['m2'] * p['l'] * np.cos(th_val)],
        [p['m2'] * p['l'] * np.cos(th_val), p['m2'] * p['l']**2]
    ])

    F = np.array([
        p['m2'] * p['l'] * (dth_val**2) * np.sin(th_val) - p['k'] * (x_val - p['l0']),
        -p['m2'] * p['g'] * p['l'] * np.sin(th_val)
    ])

    accel = np.linalg.solve(M, F)
    return [dx_val, accel[0], dth_val, accel[1]]


def draw_spring(x0, x1, y=0.0, coils=12, amp=0.06):
    """Genera puntos para dibujar un muelle entre x0 (pared) y x1 (base)."""
    dist = x1 - x0
    if abs(dist) < 1e-6:
        return np.array([x0, x1]), np.array([y, y])
    n = max(coils * 6, 50)
    xs = np.linspace(x0, x1, n)
    t = np.linspace(0, np.pi * coils, n)
    ys = y + amp * np.sin(t) * (dist / (abs(dist) + 1e-6))
    return xs, ys


def animate_system(sol, params, save=False, filename='animacion.mp4'):
    """Dibuja y anima el sistema: pared fija en x=0, muelle, base móvil (rectángulo) y péndulo.

    Añade etiquetas como en el esquema: l0, k sobre el muelle; m1 sobre la base; l y m2 en el péndulo.
    """
    t = sol.t
    x = sol.y[0]
    theta = sol.y[2]

    wall_x = 0.0

    margin = 0.7 + params['l']
    x_min = min(np.min(x), 0.0) - margin
    x_max = max(np.max(x), 0.0) + margin
    y_min = -params['l'] - margin
    y_max = 0.6

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal', 'box')
    ax.set_title('Animación: muelle + base + péndulo')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')

    # Pared vertical
    wall_line, = ax.plot([wall_x, wall_x], [y_min, y_max], color='k', linewidth=2)

    # Superficie horizontal (suelo)
    ax.plot([x_min, x_max], [0, 0], color='k', linewidth=1)

    # Elementos dinámicos
    spring_line, = ax.plot([], [], color='tab:orange', linewidth=2)
    base_width = 0.4
    base_height = 0.12
    base_patch = Rectangle((0, -base_height/2), base_width, base_height, fc='tab:blue', ec='k')
    ax.add_patch(base_patch)

    rod_line, = ax.plot([], [], color='k', linewidth=2)
    bob_radius = 0.06 + 0.02 * (params['m2']**(1/3))
    bob_patch = Circle((0, 0), bob_radius, fc='tab:red', ec='k')
    ax.add_patch(bob_patch)

    # Etiquetas estáticas aproximadas (se colocan relativas a la pared/muelle)
    label_k = ax.text(0.05 * (x_max - x_min) + wall_x, 0.08, f"l0, k", fontsize=10, va='bottom')
    label_m1 = ax.text(0.5 * (x_min + x_max), 0.12, f"m1", fontsize=10, ha='center')
    # Nota: las etiquetas se actualizarán si es necesario en update

    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    def init():
        spring_line.set_data([], [])
        base_patch.set_xy((x[0] - base_width/2, -base_height/2))
        rod_line.set_data([], [])
        bob_patch.center = (x[0], -params['l'])
        time_text.set_text('')
        return spring_line, base_patch, rod_line, bob_patch, time_text

    def update(i):
        xb = x[i]
        th = theta[i]
        bob_x = xb + params['l'] * np.sin(th)
        bob_y = - params['l'] * np.cos(th)

        xs_s, ys_s = draw_spring(wall_x, xb, y=0.0, coils=18, amp=0.06)
        spring_line.set_data(xs_s, ys_s)

        base_patch.set_xy((xb - base_width/2, -base_height/2))

        rod_line.set_data([xb, bob_x], [0.0, bob_y])

        bob_patch.center = (bob_x, bob_y)

        # Actualizar etiquetas de posición para mayor claridad
        label_k.set_position((wall_x + 0.1 * (xb - wall_x), 0.08))
        label_m1.set_position((xb, 0.16))

        # Etiqueta longitud l con una flecha aproximada
        # eliminamos la anotación anterior para evitar acumulación; en su lugar reponemos un texto temporal
        ax.texts = [tx for tx in ax.texts if tx is time_text or tx is label_k or tx is label_m1]
        ax.annotate(f"l", xy=(xb + 0.5 * params['l'] * np.sin(th), -0.5 * params['l'] * np.cos(th)),
                    xytext=(xb + 0.5 * params['l'] * np.sin(th) + 0.05, -0.5 * params['l'] * np.cos(th) + 0.05),
                    arrowprops=dict(arrowstyle='-|>', color='gray'), fontsize=9)
        # Bob label
        ax.text(bob_x, bob_y - 0.06, 'm2', fontsize=9, ha='center')

        time_text.set_text(f't = {t[i]:.2f} s')
        return spring_line, base_patch, rod_line, bob_patch, time_text

    if len(t) > 1:
        interval = max(1.0, 1000.0 * (t[1] - t[0]))
    else:
        interval = 40

    anim = FuncAnimation(fig, update, frames=len(t), init_func=init,
                         interval=interval, blit=False, repeat=False)

    if save:
        try:
            anim.save(filename, fps=int(np.round(1000.0/interval)))
            print(f"Animación guardada en {filename}")
        except Exception as e:
            print("Error guardando animación:", e)
            print("Asegúrate de tener ffmpeg instalado si quieres guardar MP4.")
    else:
        plt.show()


def main():
    # 1. Obtener parámetros interactivos
    params = pedir_parametros()

    # 2. Condiciones iniciales y tiempo de simulación
    y0 = [1.2, 0.0, np.deg2rad(45), 0.0]  # [x, dx/dt, theta, dtheta/dt]
    tspan = (0, 10)
    t_eval = np.linspace(tspan[0], tspan[1], 1000)

    print("\nEjecutando integración numérica con SciPy...")
    sol = solve_ivp(lambda t, y: edom_sistema(t, y, params), tspan, y0, 
                    t_eval=t_eval, rtol=1e-8, atol=1e-8)

    t = sol.t
    x = sol.y[0]
    dx = sol.y[1]
    theta = sol.y[2]
    dtheta = sol.y[3]

    # 3. Post-procesamiento (Energía y Fuerza Normal)
    n_pts = len(t)
    E = np.zeros(n_pts)
    N = np.zeros(n_pts)

    for i in range(n_pts):
        _, d2x, _, d2th = edom_sistema(t[i], sol.y[:, i], params)
        
        # Energía Mecánica Total (T + V)
        T = 0.5 * (params['m1'] + params['m2']) * dx[i]**2 + \
            0.5 * params['m2'] * params['l']**2 * dtheta[i]**2 + \
            params['m2'] * params['l'] * dx[i] * dtheta[i] * np.cos(theta[i])
            
        V = 0.5 * params['k'] * (x[i] - params['l0'])**2 - \
            params['m2'] * params['g'] * params['l'] * np.cos(theta[i])
            
        E[i] = T + V
        
        # Fuerza Normal del suelo sobre m1
        N[i] = (params['m1'] + params['m2']) * params['g'] + \
               params['m2'] * params['l'] * (d2th * np.sin(theta[i]) + dtheta[i]**2 * np.cos(theta[i]))

    print("¡Simulación finalizada con éxito! Generando gráficos...\n")

    # 4. Representación Gráfica (series temporales)
    plt.figure(figsize=(10, 9))

    plt.subplot(3, 1, 1)
    plt.plot(t, x, 'b-', label='x (m)', linewidth=1.5)
    plt.plot(t, np.rad2deg(theta), 'r--', label='theta (deg)', linewidth=1.5)
    plt.title('Evolución temporal de las coordenadas generalizadas')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Magnitud')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(t, E, 'k-', linewidth=1.5)
    plt.title('Energía Mecánica Total (Constante del movimiento)')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Energía (J)')
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(t, N, 'm-', linewidth=1.5)
    plt.title('Fuerza Normal del suelo sobre m1')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Normal N (N)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Preguntar al usuario si desea ver la animación
    try:
        ans = input("¿Quieres ver animación del sistema? [y/N]: ").strip().lower()
    except Exception:
        ans = 'n'
    if ans == 'y':
        try:
            animate_system(sol, params, save=False)
        except Exception as e:
            print("No ha sido posible mostrar la animación en pantalla (posible entorno sin GUI).")
            print("Error:", e)
            save_ans = input("¿Quieres guardar la animación como MP4 en lugar de mostrarla? [y/N]: ").strip().lower()
            if save_ans == 'y':
                fname = input("Nombre de archivo (por defecto animacion.mp4): ").strip() or 'animacion.mp4'
                animate_system(sol, params, save=True, filename=fname)
            else:
                print("Animación omitida.")
    else:
        print("Animación omitida. Si después la quieres, ejecuta el script y responde 'y' cuando te pregunte.")


if __name__ == '__main__':
    main()
