import argparse
import sys
import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import gridspec


DEFAULTS = {
    'm1': 2.0,
    'm2': 1.0,
    'k': 20.0,
    'l0': 1.0,
    'l': 0.8,
    'g': 9.81,
    'mu': 0.0,  # Sin rozamiento por defecto
}


def float_es(valor):
    """Convierte números escritos con coma o punto decimal."""
    try:
        return float(str(valor).replace(",", "."))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"'{valor}' no es un número válido."
        )


def pedir_parametros(defaults=None):
    """Pide parámetros al usuario. Enter conserva el valor por defecto."""
    if defaults is None:
        defaults = DEFAULTS

    def leer_float(mensaje, defecto, permitir_cero=True):
        try:
            texto = input(f"{mensaje} [{defecto}]: ").strip().replace(",", ".")
        except (EOFError, OSError):
            return defecto

        if texto == "":
            return defecto

        try:
            valor = float(texto)

            if valor < 0 or (not permitir_cero and valor == 0):
                print(f"[Aviso] El valor debe ser positivo. Se usa {defecto}.")
                return defecto

            return valor

        except ValueError:
            print(f"[Aviso] '{texto}' no es válido. Se usa {defecto}.")
            return defecto

    def quiere_rozamiento():
        try:
            respuesta = input(
                "\n¿Quieres añadir rozamiento con el suelo? (s/N) [N]: "
            ).strip().lower()
        except (EOFError, OSError):
            return False

        return respuesta in ("s", "si", "sí", "y", "yes")

    print("==================================================")
    print("  SIMULACIÓN: PÉNDULO ARTICULADO SOBRE BASE MÓVIL ")
    print("==================================================")
    print("Introduce valores o pulsa Enter para usar los predeterminados:\n")

    parametros = {
        'm1': leer_float("Masa de la base m1 (kg)", defaults['m1'], False),
        'm2': leer_float("Masa del péndulo m2 (kg)", defaults['m2'], False),
        'k': leer_float("Constante del resorte k (N/m)", defaults['k'], False),
        'l0': leer_float("Longitud natural del resorte l0 (m)", defaults['l0'], False),
        'l': leer_float("Longitud del péndulo l (m)", defaults['l'], False),
        'g': leer_float("Aceleración de la gravedad g (m/s²)", defaults['g'], False),
        'mu': 0.0,
    }

    if quiere_rozamiento():
        parametros['mu'] = leer_float(
            "Coeficiente de rozamiento μ",
            0.0
        )
        print(f"[Info] Rozamiento activado con μ = {parametros['mu']}")
    else:
        print("[Info] Rozamiento desactivado (μ = 0).")

    return parametros


def edom_sistema(t, y, p):
    """Ecuaciones de movimiento del sistema.

    y = [x, dx, theta, dtheta]

    El rozamiento con el suelo usa una aproximación suave de Coulomb:
        F_rozamiento = -mu * N * tanh(dx / velocidad_suavizado)

    Esto evita errores numéricos cuando la velocidad cambia de signo.
    """
    x_val, dx_val, th_val, dth_val = y

    matriz_masas = np.array([
        [
            p['m1'] + p['m2'],
            p['m2'] * p['l'] * np.cos(th_val)
        ],
        [
            p['m2'] * p['l'] * np.cos(th_val),
            p['m2'] * p['l'] ** 2
        ]
    ])

    # Rozamiento entre la base y el suelo.
    # Si mu = 0, la fuerza es exactamente cero.
    mu = max(0.0, p.get('mu', 0.0))
    normal = (p['m1'] + p['m2']) * p['g']
    velocidad_suavizado = 0.001

    fuerza_rozamiento = (
        -mu * normal * np.tanh(dx_val / velocidad_suavizado)
    )

    fuerzas = np.array([
        p['m2'] * p['l'] * dth_val ** 2 * np.sin(th_val)
        - p['k'] * (x_val - p['l0'])
        + fuerza_rozamiento,

        -p['m2'] * p['g'] * p['l'] * np.sin(th_val)
    ])

    aceleraciones = np.linalg.solve(matriz_masas, fuerzas)

    return [
        dx_val,
        aceleraciones[0],
        dth_val,
        aceleraciones[1]
    ]


def generar_coordenadas_resorte(x_bloque, num_vueltas=10, ancho=0.08, n_puntos=100):
    """Genera un resorte en zigzag entre la pared y el bloque."""
    x_end = float(x_bloque)
    longitud_minima = 0.1

    if np.isclose(x_end, 0.0) or abs(x_end) < longitud_minima:
        signo = 1.0 if x_end >= 0 else -1.0
        xr = np.linspace(0.0, signo * longitud_minima, n_puntos)
    else:
        xr = np.linspace(0.0, x_end, n_puntos)

    yr = np.zeros_like(xr)

    if len(xr) > 20 and abs(xr[-1] - xr[0]) > 2 * longitud_minima / 3:
        interior = slice(10, -10)
        fases = np.linspace(0, num_vueltas * 2 * np.pi, xr[interior].size)
        amplitud = ancho * max(0.5, abs(xr[-1] - xr[0]))
        yr[interior] = amplitud * np.sin(fases)

    return xr, yr


def run_simulation(params, y0=None, tspan=(0, 10), n_steps=400, save_animation=None):
    """Ejecuta y muestra o guarda la simulación."""
    if y0 is None:
        y0 = [1.2, 0.0, np.deg2rad(45), 0.0]

    t_eval = np.linspace(tspan[0], tspan[1], n_steps)

    sol = solve_ivp(
        lambda t, y: edom_sistema(t, y, params),
        tspan,
        y0,
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-8
    )

    if not sol.success:
        raise RuntimeError(f"La integración numérica falló: {sol.message}")

    t = sol.t
    x = sol.y[0]
    dx = sol.y[1]
    theta = sol.y[2]
    dtheta = sol.y[3]

    x1 = x
    y1 = np.zeros_like(x1)
    x2 = x1 + params['l'] * np.sin(theta)
    y2 = -params['l'] * np.cos(theta)

    v2x = dx + params['l'] * np.cos(theta) * dtheta
    v2y = params['l'] * np.sin(theta) * dtheta
    v2sq = v2x ** 2 + v2y ** 2

    energia_cinetica = (
        0.5 * params['m1'] * dx ** 2
        + 0.5 * params['m2'] * v2sq
    )

    energia_resorte = 0.5 * params['k'] * (x - params['l0']) ** 2
    energia_gravedad = params['m2'] * params['g'] * y2
    energia_potencial = energia_resorte + energia_gravedad
    energia_total = energia_cinetica + energia_potencial

    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1])

    ax_anim = fig.add_subplot(gs[:, 0])
    ax_theta = fig.add_subplot(gs[0, 1])
    ax_energy = fig.add_subplot(gs[1, 1])

    ax_anim.set_aspect('equal')
    ax_anim.grid(True)

    xmin = min(np.min(x1), np.min(x2)) - 0.5
    xmax = max(np.max(x1), np.max(x2)) + 0.5
    ymin = min(np.min(y1), np.min(y2)) - 0.5
    ymax = max(np.max(y1), np.max(y2)) + 0.5

    ax_anim.set_xlim(xmin, xmax)
    ax_anim.set_ylim(ymin, ymax)
    ax_anim.set_title(
        f"Péndulo sobre base con resorte — μ = {params['mu']}"
    )
    ax_anim.set_xlabel("Posición X (m)")
    ax_anim.set_ylabel("Posición Y (m)")

    ax_anim.axhline(0, color='saddlebrown', lw=3, label='Suelo')
    ax_anim.axvline(0, color='dimgray', lw=3, label='Pared')

    linea_resorte, = ax_anim.plot([], [], 'k-', lw=1.5, label='Resorte')

    bloque_m1, = ax_anim.plot(
        [],
        [],
        marker='s',
        markerfacecolor='lightgray',
        markeredgecolor='black',
        markersize=22,
        linestyle='None',
        label='Masa m1'
    )

    linea_varilla, = ax_anim.plot([], [], 'k-', lw=2, label='Péndulo')

    masa_m2, = ax_anim.plot(
        [],
        [],
        marker='o',
        markerfacecolor='lightpink',
        markeredgecolor='red',
        markersize=12,
        linestyle='None',
        label='Masa m2'
    )

    traza, = ax_anim.plot(
        [],
        [],
        'g--',
        lw=1,
        alpha=0.6,
        label='Trayectoria m2'
    )

    ax_anim.legend(loc='upper right', fontsize=8)

    th_deg = np.rad2deg(theta)

    ax_theta.set_title("Ángulo θ (grados) vs Tiempo")
    ax_theta.set_xlabel("t (s)")
    ax_theta.set_ylabel("θ (°)")

    line_theta, = ax_theta.plot([], [], color='C1', lw=1.5)
    point_theta, = ax_theta.plot([], [], 'o', color='C1')

    ax_theta.set_xlim(t[0], t[-1])

    margen_theta = (
        (np.max(th_deg) - np.min(th_deg)) * 0.1
        if np.ptp(th_deg) > 0
        else 10
    )

    ax_theta.set_ylim(
        np.min(th_deg) - margen_theta,
        np.max(th_deg) + margen_theta
    )

    ax_energy.set_title("Energía (J) vs Tiempo")
    ax_energy.set_xlabel("t (s)")
    ax_energy.set_ylabel("E (J)")

    line_E, = ax_energy.plot([], [], color='C2', lw=1.5, label='E total')
    line_T, = ax_energy.plot(
        [],
        [],
        color='C3',
        lw=1.0,
        linestyle='--',
        label='Cinética'
    )
    line_U, = ax_energy.plot(
        [],
        [],
        color='C4',
        lw=1.0,
        linestyle='-.',
        label='Potencial'
    )

    ax_energy.legend(fontsize=8)
    ax_energy.set_xlim(t[0], t[-1])

    energia_minima = np.min(energia_total)
    energia_maxima = np.max(energia_total)

    margen_energia = (
        (energia_maxima - energia_minima) * 0.1
        if energia_maxima > energia_minima
        else 1.0
    )

    ax_energy.set_ylim(
        energia_minima - margen_energia,
        energia_maxima + margen_energia
    )

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

        return (
            linea_resorte, bloque_m1, linea_varilla, masa_m2, traza,
            line_theta, point_theta, line_E, line_T, line_U
        )

    def update(frame):
        xr, yr = generar_coordenadas_resorte(x1[frame])
        linea_resorte.set_data(xr, yr)

        bloque_m1.set_data([x1[frame]], [y1[frame]])

        linea_varilla.set_data(
            [x1[frame], x2[frame]],
            [y1[frame], y2[frame]]
        )

        masa_m2.set_data([x2[frame]], [y2[frame]])
        traza.set_data(x2[:frame + 1], y2[:frame + 1])

        line_theta.set_data(t[:frame + 1], th_deg[:frame + 1])
        point_theta.set_data([t[frame]], [th_deg[frame]])

        line_E.set_data(t[:frame + 1], energia_total[:frame + 1])
        line_T.set_data(t[:frame + 1], energia_cinetica[:frame + 1])
        line_U.set_data(t[:frame + 1], energia_potencial[:frame + 1])

        return (
            linea_resorte, bloque_m1, linea_varilla, masa_m2, traza,
            line_theta, point_theta, line_E, line_T, line_U
        )

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(t),
        init_func=init,
        interval=25,
        blit=False
    )

    if save_animation:
        print(f"Guardando animación en {save_animation}...")
        ani.save(save_animation, writer='ffmpeg', fps=30)
        print("Guardado completado.")
    else:
        print("¡Listo! Abriendo ventana de animación...")
        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Simulación de péndulo articulado sobre base móvil"
    )

    parser.add_argument('--m1', type=float_es, help='Masa de la base m1 (kg)')
    parser.add_argument('--m2', type=float_es, help='Masa del péndulo m2 (kg)')
    parser.add_argument('--k', type=float_es, help='Constante del resorte k (N/m)')
    parser.add_argument('--l0', type=float_es, help='Longitud natural del resorte l0 (m)')
    parser.add_argument('--l', type=float_es, help='Longitud del péndulo l (m)')
    parser.add_argument('--g', type=float_es, help='Aceleración de la gravedad g (m/s²)')
    parser.add_argument('--mu', type=float_es, help='Coeficiente de rozamiento')
    parser.add_argument(
        '--no-input',
        action='store_true',
        help='No preguntar valores; usar los predeterminados'
    )
    parser.add_argument(
        '--save',
        type=str,
        help='Guardar la animación en un archivo MP4'
    )

    args = parser.parse_args()

    valores_por_argumento = {
        clave: getattr(args, clave)
        for clave in DEFAULTS
        if getattr(args, clave) is not None
    }

    params = DEFAULTS.copy()
    params.update(valores_por_argumento)

    if params['mu'] < 0:
        print("[Aviso] μ no puede ser negativo. Se usará μ = 0.")
        params['mu'] = 0.0

    if not valores_por_argumento and not args.no_input and sys.stdin.isatty():
        params = pedir_parametros(DEFAULTS)
    elif args.no_input or not sys.stdin.isatty():
        print("[Info] Usando valores:", params)

    headless = False

    if sys.platform != 'win32' and os.environ.get('DISPLAY', '') == '':
        headless = True

    save_target = args.save

    if headless and save_target is None:
        save_target = 'simulacion_auto.mp4'
        print(f"[Info] Sin pantalla gráfica. Se guardará en {save_target}")

    run_simulation(params, save_animation=save_target)


if __name__ == '__main__':
    main()
