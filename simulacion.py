import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import gridspec

def pedir_parametros():
    """Permite introducir los parámetros del sistema de forma interactiva.
    Si no hay entrada disponible (por ejemplo código ejecutado en entorno no interactivo)
    o hay error en la conversión, se usan valores por defecto.
    """
    print("==================================================")
    print("  SIMULACIÓN: PÉNDULO ARTICULADO SOBRE BASE MÓVIL ")
    print("==================================================")
    print("Introduce los valores o pulsa [Enter] para usar los predeterminados:\n")

    defaults = {'m1': 2.0, 'm2': 1.0, 'k': 20.0, 'l0': 1.0, 'l': 0.8, 'g': 9.81}
    try:
        # Proteger contra EOFError cuando no hay stdin disponible
        m1 = float(input("Masa de la base m1 (kg): ") or defaults['m1'])
        m2 = float(input("Masa del péndulo m2 (kg): ") or defaults['m2'])
        k  = float(input("Constante del resorte k (N/m): ") or defaults['k'])
        l0 = float(input("Longitud natural del resorte l0 (m): ") or defaults['l0'])
        l  = float(input("Longitud del péndulo l (m): ") or defaults['l'])
        g  = float(input("Aceleración de la gravedad g (m/s^2): ") or defaults['g'])
    except (ValueError, EOFError):
        print("\n[Aviso] Entrada no válida o no interactiva. Se usarán los valores por defecto.")
        m1, m2, k, l0, l, g = defaults['m1'], defaults['m2'], defaults['k'], defaults['l0'], defaults['l'], defaults['g']

    return {'m1': m1, 'm2': m2, 'k': k, 'l0': l0, 'l': l, 'g': g}


def edom_sistema(t, y, p):
    """Ecuaciones de movimiento acopladas del sistema.

    y = [x, dx, theta, dtheta]
    Devuelve [dx, ddx, dtheta, ddtheta].
    """
    x_val, dx_val, th_val, dth_val = y

    # Matriz de masa M * [ddx; ddtheta] = F
    M = np.array([
        [p['m1'] + p['m2'], p['m2'] * p['l'] * np.cos(th_val)],
        [p['m2'] * p['l'] * np.cos(th_val), p['m2'] * p['l']**2]
    ])

    # Fuerzas / términos del lado derecho
    F = np.array([
        p['m2'] * p['l'] * (dth_val**2) * np.sin(th_val) - p['k'] * (x_val - p['l0']),
        -p['m2'] * p['g'] * p['l'] * np.sin(th_val)
    ])

    # Resolver aceleraciones
    accel = np.linalg.solve(M, F)
    return [dx_val, accel[0], dth_val, accel[1]]


def generar_coordenadas_resorte(x_bloque, num_vueltas=10, ancho=0.08, n_puntos=100):
    """Genera la geometría del resorte en zigzag entre x=0 (pared) y x_bloque (bloque).
    Soporta x_bloque negativo y longitudes pequeñas.
    Devuelve arrays xr (x-coords) e yr (y-coords).
    """
    # Si el bloque está muy cerca de la pared, dibujamos una pequeña longitud mínima
    x_end = float(x_bloque)
    min_len = 0.1
    if np.isclose(x_end, 0.0) or abs(x_end) < min_len:
        # Dibujar un resorte corto de longitud min_len en la dirección de x_end
        sign = 1.0 if x_end >= 0 else -1.0
        xr = np.linspace(0.0, sign * min_len, n_puntos)
    else:
        xr = np.linspace(0.0, x_end, n_puntos)

    # Generar perfil sinusoidal centrado entre los extremos, con ceros en extremos
    yr = np.zeros_like(xr)
    # Solo introducir zigzag si la longitud real es suficientemente grande
    if len(xr) > 20 and abs(xr[-1] - xr[0]) > 2 * min_len / 3:
        # colocamos el zigzag en la región interior (evitando extremos)
        inner = slice(10, -10)
        phases = np.linspace(0, num_vueltas * 2 * np.pi, xr[inner].size)
        # amplitud escalada por la longitud para que resulte proporcional
        amp = ancho * max(0.5, abs(xr[-1] - xr[0])) 
        yr[inner] = amp * np.sin(phases)
    return xr, yr


def main():
    # 1. Obtener parámetros
    params = pedir_parametros()

    # 2. Condiciones iniciales y resolución numérica
    y0 = [1.2, 0.0, np.deg2rad(45), 0.0]  # [x, dx/dt, theta, dtheta/dt]
    tspan = (0, 10)
    t_eval = np.linspace(tspan[0], tspan[1], 400)

    print("\nCalculando trayectoria con SciPy (solve_ivp)...")
    sol = solve_ivp(lambda t, y: edom_sistema(t, y, params), tspan, y0,
                    t_eval=t_eval, rtol=1e-8, atol=1e-8)

    if not sol.success:
        print("[Error] La integración numérica falló:", sol.message)
        return

    t = sol.t
    x = sol.y[0]
    dx = sol.y[1]
    theta = sol.y[2]
    dtheta = sol.y[3]

    # Posiciones cartesianas del diagrama
    x1 = x
    y1 = np.zeros_like(x1)
    x2 = x1 + params['l'] * np.sin(theta)
    y2 = -params['l'] * np.cos(theta)

    # Calcular energías (para gráficos auxiliares)
    # Velocidad del péndulo m2
    v2x = dx + params['l'] * np.cos(theta) * dtheta
    v2y = params['l'] * np.sin(theta) * dtheta
    v2sq = v2x**2 + v2y**2

    T = 0.5 * params['m1'] * dx**2 + 0.5 * params['m2'] * v2sq
    U_spring = 0.5 * params['k'] * (x - params['l0'])**2
    U_grav = params['m2'] * params['g'] * y2
    E_total = T + U_spring + U_grav

    # 3. Configuración de la ventana gráfica: animación + gráficos auxiliares
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1])

    ax_anim = fig.add_subplot(gs[:, 0])  # ocupa dos filas, primera columna
    ax_theta = fig.add_subplot(gs[0, 1])
    ax_energy = fig.add_subplot(gs[1, 1])

    ax_anim.set_aspect('equal')
    ax_anim.grid(True)

    # Calcular límites con un pequeño margen
    xmin = min(np.min(x1), np.min(x2)) - 0.5
    xmax = max(np.max(x1), np.max(x2)) + 0.5
    ymin = min(np.min(y1), np.min(y2)) - 0.5
    ymax = max(np.max(y1), np.max(y2)) + 0.5

    ax_anim.set_xlim(xmin, xmax)
    ax_anim.set_ylim(ymin, ymax)
    ax_anim.set_title('Simulación Dinámica: Péndulo Articulado sobre Base con Resorte')
    ax_anim.set_xlabel('Posición X (m)')
    ax_anim.set_ylabel('Posición Y (m)')

    # Elementos visuales del diagrama
    ax_anim.axhline(0, color='saddlebrown', lw=3, label='Suelo')
    ax_anim.axvline(0, color='dimgray', lw=3, label='Pared')

    linea_resorte, = ax_anim.plot([], [], 'k-', lw=1.5, label='Resorte (k, l_0)')
    bloque_m1, = ax_anim.plot([], [], marker='s', markerfacecolor='lightgray',
                              markeredgecolor='black', markersize=22, linestyle='None', label='Masa m_1')
    linea_varilla, = ax_anim.plot([], [], 'k-', lw=2, label='Péndulo (l)')
    masa_m2, = ax_anim.plot([], [], marker='o', markerfacecolor='lightpink',
                           markeredgecolor='red', markersize=12, linestyle='None', label='Masa m_2')
    traza, = ax_anim.plot([], [], 'g--', lw=1, alpha=0.6, label='Trayectoria m_2')

    ax_anim.legend(loc='upper right', fontsize=8)

    # Gráficos auxiliares: ángulo y energía
    ax_theta.set_title('Ángulo θ (grados) vs Tiempo')
    ax_theta.set_xlabel('t (s)')
    ax_theta.set_ylabel('θ (°)')
    line_theta, = ax_theta.plot([], [], color='C1', lw=1.5)
    point_theta, = ax_theta.plot([], [], 'o', color='C1')
    ax_theta.set_xlim(t[0], t[-1])
    th_deg = np.rad2deg(theta)
    th_margin = (np.max(th_deg) - np.min(th_deg)) * 0.1 if np.ptp(th_deg) > 0 else 10
    ax_theta.set_ylim(np.min(th_deg) - th_margin, np.max(th_deg) + th_margin)

    ax_energy.set_title('Energía (J) vs Tiempo')
    ax_energy.set_xlabel('t (s)')
    ax_energy.set_ylabel('E (J)')
    line_E, = ax_energy.plot([], [], color='C2', lw=1.5, label='E total')
    line_T, = ax_energy.plot([], [], color='C3', lw=1.0, linestyle='--', label='T (cinética)')
    line_U, = ax_energy.plot([], [], color='C4', lw=1.0, linestyle='-.', label='U (potencial)')
    ax_energy.legend(fontsize=8)
    ax_energy.set_xlim(t[0], t[-1])
    Emin = np.min(E_total);
    Emax = np.max(E_total);
    E_margin = (Emax - Emin) * 0.1 if (Emax - Emin) > 0 else 1.0
    ax_energy.set_ylim(Emin - E_margin, Emax + E_margin)

    def init():
        linea_resorte.set_data([], [])
        bloque_m1.set_data([], [])
        linea_varilla.set_data([], [])
        masa_m2.set_data([], [])
        traza.set_data([], [])

        line_theta.set_data([], [])
        point_theta.set_data([], [])

        line_E.set_data([], [])
        line_T.set_data([], [])
        line_U.set_data([], [])

        return (linea_resorte, bloque_m1, linea_varilla, masa_m2, traza,
                line_theta, point_theta, line_E, line_T, line_U)

    def update(frame):
        # Actualizar geometría del resorte (entre x=0 y x1[frame])
        xr, yr = generar_coordenadas_resorte(x1[frame])
        linea_resorte.set_data(xr, yr)

        # Actualizar bloque m1
        bloque_m1.set_data([x1[frame]], [y1[frame]])

        # Actualizar péndulo (varilla y masa m2)
        linea_varilla.set_data([x1[frame], x2[frame]], [y1[frame], y2[frame]])
        masa_m2.set_data([x2[frame]], [y2[frame]])

        # Actualizar traza de la trayectoria
        traza.set_data(x2[:frame+1], y2[:frame+1])

        # Actualizar gráfico de ángulo
        line_theta.set_data(t[:frame+1], th_deg[:frame+1])
        point_theta.set_data(t[frame], th_deg[frame])

        # Actualizar gráfico de energía: dividir en T y U_total (spring+grav)
        U_total = U_spring + U_grav
        line_E.set_data(t[:frame+1], E_total[:frame+1])
        line_T.set_data(t[:frame+1], T[:frame+1])
        line_U.set_data(t[:frame+1], U_total[:frame+1])

        return (linea_resorte, bloque_m1, linea_varilla, masa_m2, traza,
                line_theta, point_theta, line_E, line_T, line_U)

    print("¡Listo! Abriendo ventana de animación y gráficos auxiliares...\n")
    ani = animation.FuncAnimation(fig, update, frames=len(t), init_func=init,
                                  interval=25, blit=True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
