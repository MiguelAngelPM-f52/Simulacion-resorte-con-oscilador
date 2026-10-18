import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

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

    # 4. Representación Gráfica
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

if __name__ == '__main__':
    main()
