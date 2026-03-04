import numpy as np
from scipy import constants
from typing import Tuple, Optional


def factorial(n: int) -> float:
    """Compute factorial of n."""
    result = 1.0
    for i in range(2, n + 1):
        result *= i
    return result


def legendre_coefficients(n: int) -> np.ndarray:
    """Compute coefficients of Legendre polynomial Pn(x).

    Args:
        n: Polynomial order.

    Returns:
        Array of coefficients a[i] where P_n(x) = sum(a[i] * x^i).
    """
    a = np.zeros(n + 1)
    m = n % 2
    N = n // 2 if m == 0 else (n - 1) // 2

    for i in range(N + 1):
        idx = n - 2 * i
        a[idx] = (
            ((-1) ** i) * factorial(2 * n - 2 * i)
            / (2**n * factorial(i) * factorial(n - i) * factorial(n - 2 * i))
        )

    return a


def evaluate_legendre(a: np.ndarray, n: int, m: int, x: float) -> float:
    """Evaluate Legendre polynomial at x.

    Args:
        a: Coefficients array.
        n: Polynomial order.
        m: n % 2 (parity).
        x: Evaluation point.

    Returns:
        P_n(x) value.
    """
    p = 0.0
    if m == 0:
        for i in range(0, n + 1, 2):
            if x == 0:
                break
            p += a[i] * (x ** i)
    else:
        for i in range(1, n + 1, 2):
            p += a[i] * (x ** i)
    return p


def evaluate_legendre_derivative(a: np.ndarray, n: int, m: int, x: float) -> float:
    """Evaluate derivative of Legendre polynomial at x.

    Args:
        a: Coefficients array.
        n: Polynomial order.
        m: n % 2 (parity).
        x: Evaluation point.

    Returns:
        P'_n(x) value.
    """
    p = 0.0
    if m == 0:
        for i in range(0, n + 1, 2):
            if x == 0:
                break
            if i > 0:
                p += i * a[i] * (x ** (i - 1))
    else:
        for i in range(1, n + 1, 2):
            p += i * a[i] * (x ** (i - 1))
    return p


def gauss_legendre_roots_weights(
    n: int, tolerance: float = 1e-16, damping: float = 0.5, max_iter: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Gauss-Legendre quadrature roots and weights.

    Uses Newton-Raphson iteration to find roots of Legendre polynomial.

    Args:
        n: Number of quadrature points (polynomial order).
        tolerance: Convergence tolerance for Newton-Raphson.
        damping: Damping factor for Newton-Raphson (0.5 for N1, 1.0 for N2).
        max_iter: Maximum Newton-Raphson iterations.

    Returns:
        Tuple of (roots, weights) arrays, each of length n.
    """
    if n <= 0:
        return np.array([]), np.array([])

    a = legendre_coefficients(n)
    m = n % 2

    roots = np.zeros(n)
    weights = np.zeros(n)

    for i in range(n):
        # Initial guess using Chebyshev approximation
        z = np.cos(np.pi * (i + 0.75) / (n + 0.5))
        x = z

        # Newton-Raphson iteration
        for _ in range(max_iter):
            px = evaluate_legendre(a, n, m, x)
            dpx = evaluate_legendre_derivative(a, n, m, x)

            if abs(dpx) < 1e-30:
                break

            correction = px / dpx
            x_new = x - damping * correction

            if abs(x_new - x) < tolerance:
                x = x_new
                break
            x = x_new

        roots[i] = x
        dpx = evaluate_legendre_derivative(a, n, m, x)
        weights[i] = 2.0 / ((1 - x**2) * dpx**2)

    return roots, weights


class Integrand:
    """Integrand functions for Biot-Savart field computation.

    The parameters t1, t2, t3, t4 come from the four-corner decomposition
    of the thick solenoid field calculation.
    """

    def __init__(self, t1: float, t2: float, t3: float, t4: float):
        """Initialize integrand with parameters.

        Args:
            t1, t2, t3, t4: Integration parameters from field computation.
        """
        self.t1 = t1
        self.t2 = t2
        self.t3 = t3
        self.t4 = t4

    def f1(self, x: float) -> float:
        """First integrand function (for Bz component).

        Args:
            x: Angle in radians.

        Returns:
            Integrand value.
        """
        cos_x = np.cos(x)
        sin_x = np.sin(x)

        g1 = self.t1 - self.t3 * cos_x
        g2 = self.t3**2 * sin_x**2 + (self.t2 - self.t4)**2
        g3 = self.t3 * cos_x
        gf = np.sqrt(g1**2 + g2)

        # Avoid log of zero or negative
        arg = g1 + gf
        if arg <= 0:
            arg = 1e-30

        return cos_x * (gf + g3 * np.log(arg))

    def f2(self, x: float) -> float:
        """Second integrand function (for Br component).

        Args:
            x: Angle in radians.

        Returns:
            Integrand value.
        """
        cos_x = np.cos(x)
        sin_x = np.sin(x)

        g1 = self.t1 - self.t3 * cos_x
        g2 = self.t3**2 * sin_x**2 + (self.t2 - self.t4)**2
        g3 = self.t3 * cos_x
        g4 = self.t3 * sin_x
        g5 = self.t2 - self.t4
        gf = np.sqrt(g1**2 + g2)

        # Avoid division by zero and log of negative
        eps = 1e-30

        if abs(g4) < eps:
            g4 = eps if g4 >= 0 else -eps
        if abs(g5) < eps:
            g5 = eps if g5 >= 0 else -eps
        if abs(gf + g5) < eps:
            gf_plus_g5 = eps
        else:
            gf_plus_g5 = gf + g5
        if abs(gf - g5) < eps:
            gf_minus_g5 = eps
        else:
            gf_minus_g5 = gf - g5

        log_arg = g1 + gf
        if log_arg <= 0:
            log_arg = eps

        ratio = gf_minus_g5 / gf_plus_g5
        if ratio <= 0:
            ratio = eps

        M1 = (
            2 * g5 * g3 * np.log(log_arg)
            + 0.25 * (3 * (g3**2 - g4**2) - (g1 + g3)**2) * np.log(ratio)
        )

        atan_arg1 = (g1 * (g1 + gf) + g4**2) / (g4 * g5)
        atan_arg2 = (g5 + g1 + gf) / g4
        atan_arg3 = (-g5 + g1 + gf) / g4

        M2 = -2 * np.arctan(atan_arg1) + np.arctan(atan_arg2) - np.arctan(atan_arg3)

        return cos_x * (M1 + g3 * g4 * M2 + 0.5 * g5 * gf)
    
    
def integrate_n1(
    t1: float, t2: float, t3: float, t4: float,
    lower: float, upper: float,
    order: int = 18, tolerance: float = 1e-16,
) -> float:
    """Compute N1 integral using Gaussian-Legendre quadrature.

    N1 is the integral of f1 from lower to upper, used for Bz computation.
    Uses damping factor 0.5 for Newton-Raphson root finding.

    Args:
        t1, t2, t3, t4: Integration parameters.
        lower: Lower integration limit.
        upper: Upper integration limit.
        order: Quadrature order (default 18).
        tolerance: Newton-Raphson tolerance.

    Returns:
        Integral value.
    """
    if order <= 0:
        return 0.0

    integrand = Integrand(t1, t2, t3, t4)
    roots, weights = gauss_legendre_roots_weights(order, tolerance, damping=0.5)

    # Transform from [-1, 1] to [lower, upper]
    scale = (upper - lower) / 2
    shift = (upper + lower) / 2

    result = 0.0
    for i in range(order):
        u = scale * roots[i] + shift
        result += weights[i] * integrand.f1(u)

    return result * scale


def integrate_n2(
    t1: float, t2: float, t3: float, t4: float, 
    lower: float, upper: float,
    order: int = 10, tolerance: float = 1e-16,
) -> float:
    """Compute N2 integral using Gaussian-Legendre quadrature.

    N2 is the integral of f2 from lower to upper, used for Br computation.
    Uses damping factor 1.0 for Newton-Raphson root finding.

    Args:
        t1, t2, t3, t4: Integration parameters.
        lower: Lower integration limit.
        upper: Upper integration limit.
        order: Quadrature order (default 10).
        tolerance: Newton-Raphson tolerance.

    Returns:
        Integral value.
    """
    if order <= 0:
        return 0.0

    integrand = Integrand(t1, t2, t3, t4)
    roots, weights = gauss_legendre_roots_weights(order, tolerance, damping=1.0)

    # Transform from [-1, 1] to [lower, upper]
    scale = (upper - lower) / 2
    shift = (upper + lower) / 2

    result = 0.0
    for i in range(order):
        u = scale * roots[i] + shift
        result += weights[i] * integrand.f2(u)

    return result * scale


def compute_bfield(
    coil_center: np.ndarray, coil_radius: float,
    cross_section_width: float, cross_section_height: float,
    j0: float, point: np.ndarray,
    order_n1: int = 18, order_n2: int = 10, tolerance: float = 1e-16,
) -> np.ndarray:
    """Compute magnetic field from a thick circular solenoid at a point [7].

    Args:
        coil_center: 3D coordinates of coil center [x, y, z].
        coil_radius: Radius of the coil (r0 in the paper).
        cross_section_width: Width of the coil cross-section.
        cross_section_height: Height of the coil cross-section.
        mu: Magnetic permeability (μ).
        j0: Current density (J0).
        point: 3D coordinates of the field evaluation point.
        order_n1: Quadrature order for N1 integration.
        order_n2: Quadrature order for N2 integration.
        tolerance: Integration tolerance.

    Returns:
        Array [Bx, By, Bz] of magnetic field components.
    """
    # r0 from the paper: coil radius
    r0 = coil_radius

    # Define b = dr/2 and a = dz/2
    b = cross_section_width / 2
    a = cross_section_height / 2

    # Relative position of point from coil center
    x = point[0] - coil_center[0]
    y = point[1] - coil_center[1]
    z = point[2] - coil_center[2]

    # Convert to cylindrical coordinates
    r = np.sqrt(x * x + y * y)

    # Off-axis case (r >= 0.000001)
    if r >= 1e-6:
        # Four integrals with different parameters
        # t1 = r0 ± b (coil radii), t2 = ±a (half heights), t3 = r, t4 = z
        n1_1 = integrate_n1(r0 + b, -a, r, z, 0.0, np.pi, order_n1, tolerance)
        n1_2 = integrate_n1(r0 - b, -a, r, z, 0.0, np.pi, order_n1, tolerance)
        n1_3 = integrate_n1(r0 + b, a, r, z, 0.0, np.pi, order_n1, tolerance)
        n1_4 = integrate_n1(r0 - b, a, r, z, 0.0, np.pi, order_n1, tolerance)

        # Equation (9)
        dAdz = (constants.mu_0 * j0) / (2 * np.pi) * (n1_1 - n1_2 - n1_3 + n1_4)

        # Note: N2 integral has different order of terms per C++ code
        n2_1 = integrate_n2(r0 + b, -a, r, z, 0.0, np.pi, order_n2, tolerance)
        n2_2 = integrate_n2(r0 - b, -a, r, z, 0.0, np.pi, order_n2, tolerance)
        n2_3 = integrate_n2(r0 + b, a, r, z, 0.0, np.pi, order_n2, tolerance)
        n2_4 = integrate_n2(r0 - b, a, r, z, 0.0, np.pi, order_n2, tolerance)

        # Equation (11) - different order than N1!
        dAdr = (constants.mu_0 * j0) / (2 * np.pi) * (n2_3 - n2_4 - n2_1 + n2_2)

        # Equation (8)
        Bx = -(x / r) * dAdz
        By = -(y / r) * dAdz
        Bz = (1 / r) * dAdr

    # On-axis case (r < 0.000001)
    else:
        Bx = 0.0
        By = 0.0

        # Equation (14b)
        term1_num = r0 + b + np.sqrt((r0 + b) ** 2 + (a - z) ** 2)
        term1_den = r0 - b + np.sqrt((r0 - b) ** 2 + (a - z) ** 2)
        term2_num = r0 + b + np.sqrt((r0 + b) ** 2 + (a + z) ** 2)
        term2_den = r0 - b + np.sqrt((r0 - b) ** 2 + (a + z) ** 2)

        Bz = ((constants.mu_0 * j0) / 2) * (
            (a - z) * np.log(term1_num / term1_den)
            + (a + z) * np.log(term2_num / term2_den)
        )

    return np.array([Bx, By, Bz])


class AMNSMatrix:
    """Container for AMNS precomputed field matrices.

    The AMNS matrix is a 4D tensor with dimensions:
    - c: Field component (0=Bx, 1=By, 2=Bz)
    - m: Optimization point index
    - n: Coil position index
    - w: Width index

    Attributes:
        Bx: Field x-component matrix [num_opt_points, num_coils, num_widths]
        By: Field y-component matrix [num_opt_points, num_coils, num_widths]
        Bz: Field z-component matrix [num_opt_points, num_coils, num_widths]
        num_opt_points: Number of optimization points
        num_coils: Number of coil positions
        num_widths: Number of width values
    """

    Bx: np.ndarray
    By: np.ndarray
    Bz: np.ndarray
    num_opt_points: int
    num_coils: int
    num_widths: int
    
    def __init__(self, Bx: np.ndarray, By: np.ndarray, Bz: np.ndarray,
                 num_opt_points: int, num_coils: int, num_widths: int):
        self.Bx = Bx
        self.By = By
        self.Bz = Bz
        self.num_opt_points = num_opt_points
        self.num_coils = num_coils
        self.num_widths = num_widths

    @property
    def shape(self):
        """Return the shape of the AMNS tensor."""
        return (3, self.num_opt_points, self.num_coils, self.num_widths)

    def __getitem__(self, idx):
        """Allow indexing AMNS[c, m, n, w] or AMNS[c] for component."""
        if isinstance(idx, int):
            if idx == 0:
                return self.Bx
            elif idx == 1:
                return self.By
            elif idx == 2:
                return self.Bz
            else:
                raise IndexError(f"Component index must be 0, 1, or 2, got {idx}")
        elif isinstance(idx, tuple):
            c = idx[0]
            rest = idx[1:]
            if c == 0:
                return self.Bx[rest]
            elif c == 1:
                return self.By[rest]
            elif c == 2:
                return self.Bz[rest]
            else:
                raise IndexError(f"Component index must be 0, 1, or 2, got {c}")


def compute_amns(
    opt_points: np.ndarray,
    coil_positions: np.ndarray,
    coil_radii: np.ndarray,
    widths: np.ndarray,
    cross_section_ratio: float,
    N: float,
    current: float = 1.0,
    order_n1: int = 18, order_n2: int = 10, tolerance: float = 1e-16,
    progress_callback: Optional[callable] = None,
) -> AMNSMatrix:
    """Compute the AMNS precomputed field matrix.

    For each coil configuration (position, radius, width) and each optimization
    point, computes the magnetic field contribution. This allows the optimization
    to use linear combinations of precomputed values.

    Args:
        opt_points: Array of optimization points, shape (num_opt_points, 3).
        coil_positions: Array of coil center coordinates, shape (num_coils, 3).
        coil_radii: Array of coil radii, shape (num_coils,).
        widths: Array of cross-section widths, shape (num_widths,).
        cross_section_ratio: Height/width ratio for cross-section.
        N: Number of wire turns (used for current density calculation).
        current: Current value (default 1.0 for unit current precomputation).
        order_n1: Quadrature order for N1 integration.
        order_n2: Quadrature order for N2 integration.
        tolerance: Integration tolerance.
        progress_callback: Optional callback(m, num_opt_points) for progress.

    Returns:
        AMNSMatrix containing the precomputed field values.
    """
    num_opt_points = len(opt_points)
    num_coils = len(coil_positions)
    num_widths = len(widths)

    # Initialize output arrays
    Bx = np.zeros((num_opt_points, num_coils, num_widths))
    By = np.zeros((num_opt_points, num_coils, num_widths))
    Bz = np.zeros((num_opt_points, num_coils, num_widths))

    for m in range(num_opt_points):
        point = opt_points[m]

        for n in range(num_coils):
            coil_center = coil_positions[n]
            radius = coil_radii[n]

            for w in range(num_widths):
                width = widths[w]
                height = width / cross_section_ratio  # C++ uses width / ratio

                # Current density: J0 = N * I / area
                area = width * height
                j0 = N * current / area

                # Compute field
                b_xyz = compute_bfield(
                    coil_center=coil_center,
                    coil_radius=radius,
                    cross_section_width=width,
                    cross_section_height=height,
                    j0=j0,
                    point=point,
                    order_n1=order_n1,
                    order_n2=order_n2,
                    tolerance=tolerance,
                )

                Bx[m, n, w] = b_xyz[0]
                By[m, n, w] = b_xyz[1]
                Bz[m, n, w] = b_xyz[2]

        if progress_callback:
            progress_callback(m + 1, num_opt_points)

    return AMNSMatrix(
        Bx=Bx,
        By=By,
        Bz=Bz,
        num_opt_points=num_opt_points,
        num_coils=num_coils,
        num_widths=num_widths,
    )
