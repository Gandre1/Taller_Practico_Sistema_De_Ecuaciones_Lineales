import numpy as np
import scipy.linalg as sla
import time
import matplotlib.pyplot as plt
from typing import Tuple, Dict
import os

np.random.seed(0)  # reproducibilidad

# UTILIDADES

def make_well_conditioned(n: int, diag_boost: float = None) -> Tuple[np.ndarray, np.ndarray]:
    """Genera una matriz A (invertible y razonablemente condicionada) y vector b.
    Para favorecer convergencia de Gauss-Seidel hacemos la matriz diagonalmente dominante
    al sumar diag_boost * I (si diag_boost es None, se usa n).
    """
    A = np.random.randn(n, n)
    if diag_boost is None:
        diag_boost = float(n)
    A = A + diag_boost * np.eye(n)
    b = np.random.randn(n)
    return A, b


def residual_norm(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    r = A.dot(x) - b
    return np.linalg.norm(r, 2)


# MÉTODOS DIRECTOS

def solve_numpy(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Resolver usando np.linalg.solve (recomendado para eficiencia y estabilidad).
    """
    return np.linalg.solve(A, b)


def solve_inverse(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Resolver usando la matriz inversa: x = A^{-1} b (menos recomendado).
    Mostrar para comparación de precisión/tiempo.
    """
    Ainv = np.linalg.inv(A)
    return Ainv.dot(b)


def solve_lu(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Resolver usando factorización LU (scipy.linalg.lu_factor + lu_solve)
    """
    lu, piv = sla.lu_factor(A)
    x = sla.lu_solve((lu, piv), b)
    return x


# ELIMINACIÓN GAUSSIANA (propia)

def gaussian_elimination(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Implementación de eliminación gaussiana con pivoteo parcial.
    Devuelve la solución x del sistema A x = b. No modifica A,b originales."""
    A = A.copy().astype(float)
    b = b.copy().astype(float)
    n = A.shape[0]

    # Eliminación hacia adelante
    for k in range(n - 1):
        # pivoteo parcial
        max_row = np.argmax(np.abs(A[k:, k])) + k
        if np.abs(A[max_row, k]) < 1e-15:
            raise np.linalg.LinAlgError("Matriz singular o casi singular detectada en pivoteo.")
        if max_row != k:
            A[[k, max_row]] = A[[max_row, k]]
            b[[k, max_row]] = b[[max_row, k]]
        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] = A[i, k:] - factor * A[k, k:]
            b[i] = b[i] - factor * b[k]

    # Sustitución regresiva
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if abs(A[i, i]) < 1e-15:
            raise np.linalg.LinAlgError("Matriz singular detectada en sustitución regresiva.")
        x[i] = (b[i] - A[i, i + 1 :].dot(x[i + 1 :])) / A[i, i]
    return x


# GAUSS-SEIDEL (iterativo)

def gauss_seidel(A: np.ndarray, b: np.ndarray, x0: np.ndarray = None, tol: float = 1e-8, max_iter: int = 10000) -> Tuple[np.ndarray, int]:
    """Implementación del método de Gauss-Seidel.
    Requiere preferiblemente que A sea diagonalmente dominante para convergencia garantizada.
    Devuelve (x, iters)."""
    n = A.shape[0]
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy().astype(float)

    # Pre-computar para velocidad
    D = np.diag(A)
    if np.any(np.abs(D) < 1e-15):
        raise np.linalg.LinAlgError("La matriz tiene ceros en la diagonal; Gauss-Seidel no es aplicable directamente.")

    for it in range(1, max_iter + 1):
        x_old = x.copy()
        for i in range(n):
            s1 = A[i, :i].dot(x[:i])
            s2 = A[i, i + 1 :].dot(x_old[i + 1 :])
            x[i] = (b[i] - s1 - s2) / A[i, i]
        # criterio de parada
        if np.linalg.norm(x - x_old, ord=np.inf) < tol:
            return x, it
    return x, max_iter


# EXPERIMENTO Y COMPARACIÓN

def run_experiment(A: np.ndarray, b: np.ndarray, methods: Dict[str, callable]) -> Dict[str, dict]:
    """Ejecuta cada método de 'methods' sobre (A,b), mide tiempo, calcula residual y retorna resultados.
    methods: dict con nombre -> función(A,b) -> x"""
    results = {}
    for name, fn in methods.items():
        t0 = time.time()
        if name == 'Gauss-Seidel':
            x, iters = fn(A, b)
            t1 = time.time()
            res = residual_norm(A, x, b)
            results[name] = {'x': x, 'time': t1 - t0, 'residual': res, 'iters': iters}
        else:
            x = fn(A, b)
            t1 = time.time()
            res = residual_norm(A, x, b)
            results[name] = {'x': x, 'time': t1 - t0, 'residual': res}
    return results


# VISUALIZACIÓN

def plot_comparison(results: Dict[str, dict], title_prefix: str = '', save_prefix: str = None):
    names = list(results.keys())
    times = [results[n]['time'] for n in names]
    residuals = [results[n]['residual'] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Tiempo
    axes[0].bar(names, times)
    axes[0].set_ylabel('Tiempo (s)')
    axes[0].set_title(f'{title_prefix} - Tiempo de ejecución')
    axes[0].tick_params(axis='x', rotation=45)

    # Residual (log scale porque pueden variar mucho)
    axes[1].bar(names, residuals)
    axes[1].set_yscale('log')
    axes[1].set_ylabel('Residual ||Ax-b||_2 (escala log)')
    axes[1].set_title(f'{title_prefix} - Precisión (residual)')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    if save_prefix:
        fn = f'{save_prefix}_comparison.png'
        plt.savefig(fn, bbox_inches='tight')
        print(f'Gráfico guardado en: {fn}')
    plt.show()


# PRUEBAS AUTOMATIZADAS

def automated_tests(sizes=(3, 10, 100)):
    """Corre los experimentos para tamaños indicados y guarda gráficos,y, además, 
    guarda las matrices generadas (A,b) en la carpeta `datasets/` como .txt"""
    summary = {}
    os.makedirs('datasets', exist_ok=True)

    for n in sizes:
        print('' + '=' * 60)
        print(f'Prueba para n = {n}')
        A, b = make_well_conditioned(n)

        # Guardar dataset (A,b) generado para esta prueba
        combined = f'datasets/matrix_n{n}.txt'
        # Guardar UN solo archivo de texto por matriz: primera línea = n, luego A (n filas), línea '# b', luego b en UNA sola línea
        with open(combined, 'w') as fh:
            fh.write(f'{n}\n')
            np.savetxt(fh, A, fmt='%.18e')
            fh.write('# b\n')
            # b como una sola línea separada por espacios
            fh.write(' '.join([f'{val:.18e}' for val in b]) + '\n')
        print(f'Dataset combinado guardado en: {combined}')

        methods = {
            'NumPy.solve': solve_numpy,
            'Inverse (np.inv)': solve_inverse,
            'LU (scipy)': solve_lu,
            'Gaussian elimination': gaussian_elimination,
            'Gauss-Seidel': lambda A, b: gauss_seidel(A, b, tol=1e-8, max_iter=10000)[0:2]
        }

        # Ejecutar y medir
        results = run_experiment(A, b, methods)

        # Mostrar resumen en consola
        for name, r in results.items():
            if 'iters' in r:
                print(f"{name}: time={r['time']:.6f}s, residual={r['residual']:.2e}, iters={r['iters']}")
            else:
                print(f"{name}: time={r['time']:.6f}s, residual={r['residual']:.2e}")

        # Guardar figura comparativa
        save_prefix = f'comparison_n{n}'
        plot_comparison(results, title_prefix=f'n={n}', save_prefix=save_prefix)

        # Guardar resumen numérico
        summary[n] = {name: {'time': r['time'], 'residual': r['residual'], 'iters': r.get('iters', None)} for name, r in results.items()}


# MAIN
if __name__ == '__main__':
    print('Ejecutando pruebas automáticas (3x3, 10x10, 100x100). Esto puede tardar unos segundos).')
    automated_tests(sizes=(3, 10, 100))
    print('\nHecho.')
    