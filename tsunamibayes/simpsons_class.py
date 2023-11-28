from __future__ import division  # Python 2 compatibility
from scipy.special import binom
from scipy.interpolate import RectBivariateSpline
import numpy as np
import sys
from concurrent.futures import ThreadPoolExecutor


class TimeModelSimpsons:
    """A class to approximate line integrals using Simpson's rule
    based on Bezier curves to model arrival time of a tsunami.

    This class facilitates the calculation of speed functions
    derived from Bezier curves representing geographical path of the tsunami.
    It employs Simpson's rule for numerical integration to
    determine the speed along a path defined by control points.

    Attributes:
        control_points (numpy.ndarray): An array containing control points
            defining the Bezier curve's trajectory.
        matrix (numpy.ndarray): Matrix representation of bathymetry data.
        base_x (numpy.ndarray): Array representing the X-axis values.
        base_y (numpy.ndarray): Array representing the Y-axis values.
        cached_coefficients (dict): Cached binomial coefficients for optimization.

    Methods:
        __init__(self, control_points): Initializes the TimeModelSimpsons instance.
        create_x(self): Generates the X-axis array based on specified parameters.
        create_y(self): Generates the Y-axis array based on specified parameters.
        get_depth(self, lon, lat): Retrieves depth values at given coordinates.
        binomial_coefficient(self, n, k): Computes binomial coefficients.
        bezier_curve(self, control_points, num_points=100): Calculates Bezier curves.
        derivative_bezier_curve(self, control_points, num_points=100): Computes the derivative of Bezier curves.
        get_derivative_at_t(self, control_points, t): Computes derivatives at a given point.
        speed_function(self, t): Computes the speed function at a given time.
        simpson(self, a, b, n, num_threads=24): Approximates definite integrals
            using Simpson's rule with multithreading support.
    """
    # Used to fetch cached values
    cached_coefficients = {}

    def __init__(self, control_points):
        """
        Initializes a TimeModelSimpsons instance.

        Args:
            control_points (numpy.ndarray): An array containing control points
                defining the Bezier curve's trajectory.
        Attributes:
            control_points (numpy.ndarray): An array containing control points defining the Bezier curve's trajectory.
            matrix (numpy.ndarray): Matrix representation of bathymetry data.
            base_x (numpy.ndarray): Array representing the X-axis values.
            base_y (numpy.ndarray): Array representing the Y-axis values.
            bezier_values (Tuple[numpy.ndarray, numpy.ndarray]): Tuple containing arrays
            representing the X and Y values of the Bezier curve.
            bezier_derivative_values (Tuple[numpy.ndarray, numpy.ndarray]): Tuple containing arrays
            representing the X and Y values of the derivative of the Bezier curve.
            cached_coefficients (dict): Cached binomial coefficients for optimization.
        """
        self.control_points = control_points
        self.matrix = self.read_file_into_matrix(sys.argv[1])
        self.base_x = self.create_x()
        self.base_y = self.create_y()
        self.bezier_values = self.bezier_curve(control_points)
        self.bezier_derivative_values = self.derivative_bezier_curve(control_points)

    def make_matrix(self, lines):
        # Convert a list of strings into a matrix of integers
        matrix = []
        for line in lines:
            line = line.split()
            new_line = [int(num) for num in line]
            matrix.append(new_line)
        return matrix[::-1]

    def readlines(self, filename):
        with open(filename) as file:
            return file.readlines()


    def read_file_into_matrix(self, infile):
        lines = self.readlines(infile)
        matrix = self.make_matrix(lines)
        return np.array(matrix).T

    def create_x(self):
        """
        Generates the X-axis array based on specified parameters.

        Returns:
            numpy.ndarray: Array representing the X-axis values.
        """
        start_value = 124.991666666667  # Define the starting value for X-axis
        step = 0.016666666667  # Define the step size for X-axis
        array_length = 571  # Define the length of the array

        # Generate the X-axis array using numpy's linspace function
        my_array = np.linspace(start_value, start_value + step * (array_length - 1), array_length)
        return my_array

    def create_y(self):
        """
        Generates the Y-axis array based on specified parameters.

        Returns:
            numpy.ndarray: Array representing the Y-axis values.
        """
        start_value = -9.508333333333  # Define the starting value for Y-axis
        step = 0.016666666667  # Define the step size for Y-axis
        array_length = 421  # Define the length of the array

        # Generate the Y-axis array using numpy's linspace function
        my_array = np.linspace(start_value, start_value + step * (array_length - 1), array_length)
        return my_array

    def get_depth(self, lon, lat):
        """
        Retrieves depth values at given coordinates (longitude and latitude).

        Args:
            lon (float): Longitude coordinate.
            lat (float): Latitude coordinate.

        Returns:
            float: Interpolated depth value at the specified coordinates.
        """

        # Extract the base X, Y, and Z values
        x = self.base_x
        y = self.base_y
        z = np.array(self.matrix)

        # Create an interpolation function using RectBivariateSpline from scipy
        interp_function = RectBivariateSpline(x, y, z)

        # Compute the interpolated depth value at the given coordinates
        interp_value = interp_function(lon, lat)

        return interp_value[0][0]

    def binomial_coefficient(self, n, k):
        """
        Computes binomial coefficients and caches results for optimization.

        Args:
            n (int): Total number of items.
            k (int): Number of items to choose.

        Returns:
            int: Binomial coefficient (n choose k).
        """

        # Check if the binomial coefficient for given n and k is already cached
        if (n, k) in self.cached_coefficients:
            return self.cached_coefficients[(n, k)]
        else:
            # Calculate the binomial coefficient using scipy's binom function
            res = binom(n, k)
            # Cache the computed binomial coefficient for future use
            self.cached_coefficients[(n, k)] = res
            return res

    def bezier_curve(self, control_points, num_points=100):
        """
        Calculates a Bezier curve based on control points.

        Args:
            control_points (numpy.ndarray): Array containing control points
                defining the Bezier curve's trajectory.
            num_points (int, optional): Number of points to generate on the curve.
                Defaults to 100.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: X and Y values of the Bezier curve.
        """

        # Determine the number of control points
        m = len(control_points) - 1

        # Generate an array of 't' values
        t_values = np.linspace(0, 1, num_points)

        # Initialize arrays to store X and Y values of the curve
        x_values = np.zeros(num_points, dtype=np.float64)
        y_values = np.zeros(num_points, dtype=np.float64)

        # Iterate through each 't' value
        for i in range(m + 1):
            # Calculate temporary values for X and Y components of the curve
            temp_x = (
                self.binomial_coefficient(m, i)
                * (1 - t_values) ** (m - i)
                * t_values**i
                * control_points[i, 0]
            )
            temp_y = (
                self.binomial_coefficient(m, i)
                * (1 - t_values) ** (m - i)
                * t_values**i
                * control_points[i, 1]
            )

            # Accumulate the temporary values to get the final curve
            x_values += temp_x
            y_values += temp_y
        return x_values, y_values

    def derivative_bezier_curve(self, control_points, num_points=100):
        """
        Calculates the derivative of a Bezier curve based on control points.

        Args:
            control_points (numpy.ndarray): Array containing control points
                defining the Bezier curve's trajectory.
            num_points (int, optional): Number of points to generate on the curve.
                Defaults to 100.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: X and Y values of the derivative of the Bezier curve.
        """

        # Determine the number of control points
        m = len(control_points) - 1

        # Generate an array of 't' values
        t_values = np.linspace(0, 1, num_points)

        # Initialize arrays to store X and Y derivative values of the curve
        x_derivatives = np.zeros(num_points, dtype=np.float64)
        y_derivatives = np.zeros(num_points, dtype=np.float64)

        # Iterate through each 't' value
        for i in range(m):
            # Calculate temporary values for X and Y components of the derivative curve
            temp_x = (
                m * (control_points[i + 1, 0] - control_points[i, 0])
                * self.binomial_coefficient(m - 1, i)
                * (1 - t_values) ** (m - i - 1)
                * t_values**i
            )
            temp_y = (
                m * (control_points[i + 1, 1] - control_points[i, 1])
                * self.binomial_coefficient(m - 1, i)
                * (1 - t_values) ** (m - i - 1)
                * t_values**i
            )
            # Accumulate the temporary values to get the final derivative curve values
            x_derivatives += temp_x
            y_derivatives += temp_y

        return x_derivatives, y_derivatives

    def get_derivative_at_t(self, control_points, t):
        """
        Calculates the derivative of the Bezier curve at a specific 't' value.

        Args:
            control_points (numpy.ndarray): Array containing control points
                defining the Bezier curve's trajectory.
            t (int): Value between 0 and 1 indicating the position on the curve.

        Returns:
            Tuple[float, float, float, float]: Tuple containing x-coordinate, y-coordinate,
                x-derivative, and y-derivative values at the given 't'.
        """
        # Number of points used in the curve calculation
        num_points = 100

        # Determine the index corresponding to the given 't' in the curve arrays
        t_index = int(t * (num_points - 1))

        # Extract the derivative values from pre-calculated arrays
        x_derivative, y_derivative = self.bezier_derivative_values
        x, y = self.bezier_values

        # Return the x-coordinate, y-coordinate, x-derivative, and y-derivative at the given 't'
        return x[t_index], y[t_index], x_derivative[t_index], y_derivative[t_index]

    def speed_function(self, t):
        """
        Calculates the speed at a specific 't' value along the Bezier curve.

        Args:
            t (int): Value between 0 and 1 indicating the position on the curve.

        Returns:
            float: Speed at the given 't' value on the Bezier curve.
        """
        # Get the x, y, x_derivative, and y_derivative at the specified 't'
        x, y, x_derivative, y_derivative = self.get_derivative_at_t(control_points, t)

        # Calculate the interpolated depth at the given x, y coordinates
        interp_depth = float(abs(self.get_depth(x, y)))

        # Calculate the speed using the derivatives and depth
        speed = (x_derivative**2 + y_derivative**2) ** 0.5 / ((9.8 * interp_depth) ** 0.5)

        return speed

    def simpson(self, a: float, b: float, n: int, num_threads: int = 24):
        """
        Approximates the definite integral of the speed function using the composite Simpson's rule.

        Args:
            a (float): Lower limit of integration.
            b (float): Upper limit of integration.
            n (int): Number of subintervals (must be even).
            num_threads (int): Number of threads for concurrent computation (default=24).

        Returns:
            float: Approximated definite integral of the speed function.
        """

        # Check if the number of subintervals is even
        if n % 2:
            raise ValueError("n must be even (received n=%d)" % n)

        # Calculate the subinterval width
        h = (b - a) / n

        # Function to compute the subinterval values in parallel
        def compute_subinterval(i):
            x = a + i * h

            # Conditions to apply Simpson's rule
            if i == 0 or i == n:
                return self.speed_function(x)
            elif i % 2 == 1:
                return 4 * self.speed_function(x)
            else:
                return 2 * self.speed_function(x)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            subinterval_values = list(executor.map(compute_subinterval, range(n + 1)))

        # Calculate the approximate integral using Simpson's rule
        s = sum(subinterval_values)
        return s * h / 3


if __name__ == "__main__":
    control_points = np.array([(132.125000000143, -4.674999999903001), (132.125000000143, -4.691666666570001), (132.125000000143, -4.708333333237), (132.125000000143, -4.724999999904001), (132.125000000143, -4.7416666665710006), (132.108333333476, -4.7416666665710006), (132.091666666809, -4.7416666665710006), (132.091666666809, -4.758333333238), (132.091666666809, -4.774999999905001), (132.091666666809, -4.7916666665720005), (132.075000000142, -4.7916666665720005), (132.075000000142, -4.808333333239), (132.058333333475, -4.808333333239), (132.041666666808, -4.808333333239), (132.025000000141, -4.808333333239), (132.008333333474, -4.808333333239), (131.991666666807, -4.808333333239), (131.97500000014, -4.808333333239), (131.958333333473, -4.808333333239), (131.94166666680601, -4.808333333239), (131.925000000139, -4.808333333239), (131.908333333472, -4.808333333239), (131.908333333472, -4.824999999906001), (131.891666666805, -4.824999999906001), (131.87500000013802, -4.824999999906001), (131.87500000013802, -4.841666666573), (131.858333333471, -4.841666666573), (131.841666666804, -4.841666666573), (131.825000000137, -4.841666666573), (131.80833333347, -4.841666666573), (131.791666666803, -4.841666666573), (131.775000000136, -4.841666666573), (131.758333333469, -4.841666666573), (131.741666666802, -4.841666666573), (131.725000000135, -4.841666666573), (131.708333333468, -4.841666666573), (131.691666666801, -4.841666666573), (131.675000000134, -4.841666666573), (131.658333333467, -4.841666666573), (131.6416666668, -4.841666666573), (131.625000000133, -4.841666666573), (131.608333333466, -4.841666666573), (131.591666666799, -4.841666666573), (131.575000000132, -4.841666666573), (131.55833333346501, -4.841666666573), (131.541666666798, -4.841666666573), (131.525000000131, -4.841666666573), (131.508333333464, -4.841666666573), (131.491666666797, -4.841666666573), (131.47500000013, -4.841666666573), (131.458333333463, -4.841666666573), (131.441666666796, -4.841666666573), (131.425000000129, -4.841666666573), (131.408333333462, -4.841666666573), (131.391666666795, -4.841666666573), (131.375000000128, -4.841666666573), (131.358333333461, -4.841666666573), (131.341666666794, -4.841666666573), (131.325000000127, -4.841666666573), (131.30833333346, -4.841666666573), (131.291666666793, -4.841666666573), (131.275000000126, -4.841666666573), (131.258333333459, -4.841666666573), (131.241666666792, -4.841666666573), (131.225000000125, -4.841666666573), (131.208333333458, -4.841666666573), (131.191666666791, -4.841666666573), (131.175000000124, -4.841666666573), (131.158333333457, -4.841666666573), (131.14166666679, -4.841666666573), (131.125000000123, -4.841666666573), (131.108333333456, -4.841666666573), (131.091666666789, -4.841666666573), (131.075000000122, -4.841666666573), (131.058333333455, -4.841666666573), (131.041666666788, -4.841666666573), (131.025000000121, -4.841666666573), (131.008333333454, -4.841666666573), (130.991666666787, -4.841666666573), (130.97500000012, -4.841666666573), (130.958333333453, -4.841666666573), (130.941666666786, -4.841666666573), (130.925000000119, -4.841666666573), (130.908333333452, -4.841666666573), (130.891666666785, -4.841666666573), (130.875000000118, -4.841666666573), (130.85833333345101, -4.841666666573), (130.841666666784, -4.841666666573), (130.825000000117, -4.841666666573), (130.80833333345, -4.841666666573), (130.791666666783, -4.841666666573), (130.775000000116, -4.841666666573), (130.758333333449, -4.841666666573), (130.741666666782, -4.841666666573), (130.725000000115, -4.841666666573), (130.708333333448, -4.841666666573), (130.691666666781, -4.841666666573), (130.675000000114, -4.841666666573), (130.658333333447, -4.841666666573), (130.64166666678, -4.841666666573), (130.625000000113, -4.841666666573), (130.608333333446, -4.841666666573), (130.591666666779, -4.841666666573), (130.575000000112, -4.841666666573), (130.558333333445, -4.841666666573), (130.541666666778, -4.841666666573), (130.525000000111, -4.841666666573), (130.508333333444, -4.841666666573), (130.491666666777, -4.841666666573), (130.47500000011001, -4.841666666573), (130.458333333443, -4.841666666573), (130.441666666776, -4.841666666573), (130.425000000109, -4.841666666573), (130.408333333442, -4.841666666573), (130.391666666775, -4.841666666573), (130.375000000108, -4.841666666573), (130.358333333441, -4.841666666573), (130.341666666774, -4.841666666573), (130.325000000107, -4.841666666573), (130.30833333344, -4.841666666573), (130.291666666773, -4.841666666573), (130.275000000106, -4.841666666573), (130.258333333439, -4.841666666573), (130.241666666772, -4.841666666573), (130.225000000105, -4.841666666573), (130.208333333438, -4.841666666573), (130.191666666771, -4.841666666573), (130.175000000104, -4.841666666573), (130.158333333437, -4.841666666573), (130.14166666677, -4.841666666573), (130.125000000103, -4.841666666573), (130.108333333436, -4.841666666573), (130.09166666676902, -4.841666666573), (130.075000000102, -4.841666666573), (130.058333333435, -4.841666666573), (130.041666666768, -4.841666666573), (130.025000000101, -4.841666666573), (130.008333333434, -4.841666666573), (129.991666666767, -4.841666666573), (129.9750000001, -4.841666666573), (129.958333333433, -4.841666666573), (129.941666666766, -4.841666666573), (129.925000000099, -4.841666666573), (129.908333333432, -4.841666666573), (129.891666666765, -4.841666666573), (129.875000000098, -4.841666666573), (129.858333333431, -4.841666666573), (129.841666666764, -4.841666666573), (129.825000000097, -4.841666666573), (129.80833333343, -4.841666666573), (129.791666666763, -4.841666666573), (129.791666666763, -4.824999999906001), (129.791666666763, -4.808333333239), (129.791666666763, -4.7916666665720005), (129.791666666763, -4.774999999905001), (129.791666666763, -4.758333333238), (129.791666666763, -4.7416666665710006), (129.791666666763, -4.724999999904001), (129.791666666763, -4.708333333237), (129.791666666763, -4.691666666570001), (129.791666666763, -4.674999999903001), (129.791666666763, -4.658333333236), (129.791666666763, -4.641666666569001), (129.791666666763, -4.624999999902), (129.791666666763, -4.6083333332350005), (129.791666666763, -4.591666666568001), (129.791666666763, -4.574999999901), (129.791666666763, -4.558333333234001), (129.791666666763, -4.541666666567001), (129.791666666763, -4.5249999999), (129.77500000009601, -4.5249999999)])
    runner = TimeModelSimpsons(control_points)
    result = runner.simpson(0.0, 1.0, 100000)
    print((result * 111120) / 60)