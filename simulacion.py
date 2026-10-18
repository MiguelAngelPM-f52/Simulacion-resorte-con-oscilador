import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

def generar_coordenadas_resorte(x_bloque, num_vueltas=10, ancho=0.08):
    """Genera los puntos (x, y) para dibujar un resorte en zigzag desde x=0 hasta x_bloque."""
    x = np.linspace(0, x_bloque, 100)
    y = np.zeros(100)
    # Zigzag central
    y[10:-10] = ancho * np.sin(np.linspace(0, num_vueltas * 2 * np.pi, 80))
    return x, y

def main():
    # 1. Parámetros
    params = pedir_parametros()

    # 2. Condiciones iniciales
    y0 = [1.2, 0.0, np.deg2rad(45), 0.0]  # [x, dx/dt, theta, dtheta/dt]
    tspan = (0, 10)
    t_eval = np.linspace(tspan[0], tspan[1], 500)

    print("\nEjecutando integración numérica...")
    sol = solve_ivp(lambda t, y: edom_sistema(t, y, params), tspan, y0, 
                    t_eval=t_eval, rtol=1e-8, atol=1e-8)

    t = sol.t
    x = sol.y[0]
    theta = sol.y[2]

    # Posiciones cartesianas de las masas
    x1 = x
    y1 = np.zeros_like(x1)

    x2 = x1 + params['l'] * np.sin(theta)
    y2 = -params['l'] * np.cos(theta)

    # 3. Animación gráfica del sistema físico
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlim(-0.5, np.max(x1) + params['l'] + 0.5)
    ax.set_ylim(-params['l'] - 0.5, 0.5)
    ax.set_title('Animación: Sistema Resorte - Péndulo Acoplado')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')

    # Elementos visuales
    linea_suelo = ax.axhline(0, color='brown', lw=2)
    linea_pared = ax.axvline(0, color='gray', lw=4)
    linea_resorte, = ax.plot([], [], 'k-', lw=1.5, label='Resorte')
    bloque_m1, = ax.plot([], [], 'bs', ms=20, label='Masa m1')
    linea_varilla, = ax.plot([], [], 'r-', lw=2, label='Varilla péndulo')
    masa_m2, = ax.plot([], [], 'ro', ms=12, label='Masa m2')
    traza_pulo, = ax.plot([], [], 'g--', lw=1, alpha=0.5, label='Trayectoria m2')

    ax.legend(loc='upper right')

    def init():
        linea_resorte.set_data([], [])
        bloque_m1.set_data([], [])
        linea_varilla.set_data([], [])
        masa_m2.set_data([], [])
        traza_pulo.set_data([], [])
        return linea_resorte, bloque_m1, linea_varilla, masa_m2, traza_pulo

    def update(frame):
        # Dibujar resorte
        xr, yr = generar_coordenadas_resorte(x1[frame])
        linea_resorte.set_data(xr, yr)

        # Dibujar m1
        bloque_m1.set_data([x1[frame]], [y1[frame]])

        # Dibujar péndulo (varilla y masa m2)
        linea_varilla.set_data([x1[frame], x2[frame]], [y1[frame], y2[frame]])
        masa_m2.set_data([x2[frame]], [y2[frame]])

        # Trayectoria histórica de m2
        traza_pulo.set_data(x2[:frame], y2[:frame])

        return linea_resorte, bloque_m1, linea_varilla, masa_m2, traza_pulo

    ani = animation.FuncAnimation(fig, update, frames=len(t), init_func=init,
                                  interval=20, blit=True)

    plt.show()

if __name__ == '__main__':
    main()
