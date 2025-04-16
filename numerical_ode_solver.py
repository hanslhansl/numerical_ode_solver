import scipy, re, sympy, numpy as np

class NumericalODESolver:
    """
    A wrapper around scipy.integrate.solve_bvp() to solve a boundary value problem (BVP) for an ordinary differential equation (ODE).
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_bvp.html for details.
    It simplifies the process of defining the ODE, boundary conditions and initial guess.

    Method generate_scipy_string() generates a string containing python code which, if executed, solves the BVP.
    Either plug it into exec() or copy it to a new file and run it. This is the recommended way if performance is critical.

    Method run_scipy() runs the BVP solver directly and returns the solution.
    Because it relies on repeated calls to eval()/exec() it has inferior performance.
    """
    identifier_pattern = r"[a-zA-Z_]\w*"
    derivative_leibnitz_notation = rf"\bd({identifier_pattern})/d({identifier_pattern})\b"
    higher_derivative_leibnitz_notation_a = rf"\bd([1-9][0-9]*)({identifier_pattern})/d({identifier_pattern})\b"
    higher_derivative_leibnitz_notation_b = rf"\bd\^([1-9][0-9]*)({identifier_pattern})/d({identifier_pattern})\^([1-9][0-9]*)\b"
    derivative_prime_notation = rf"\b({identifier_pattern})('+)"
    object_access_pattern = rf'\b(?:{identifier_pattern}\.)+{identifier_pattern}\b'

    def __init__(self, ode : str,
                 interval : tuple[str | int | float, str | int | float],
                 bcs : str | tuple[str,...],
                 initial_guess : str | int | float | tuple[str | int | float,...] = 0):
        """
        Regarding syntax
        --------------------
        Several of the parameters are passed as string containg python-like code representing mathematical expressions.
        For the most part these strings are interpreted as normal python code and therefor have to obey python syntax rules, i.e.:
        - 'a' to the power of 'b' is 'a**b'
        - Mathematical brackets are '(' and ')'. '[]' and '{}' can be used but retain their 'python meaning'.
        - Names can contain digits but can't start with a digit.
        - Constants, variables, and function calls are supported, but they must be available in the current scope - typically as global definitions.

        The exception to the python syntax rules is the target function: It is denoted as a valid python identifier like 'y', 'v', 'v1', 'v_1', 'foo', 'bar' etc.
        Invalid examples: 'y(x)' (the parameter 'x' must not be specified with the function), '1y' (starts with digit).

        There are two ways to denote the derivative of the target function:
        - Leibnitz notation: The derivative of 'y' by parameter 'x' is denoted as 'dy/dx'. The second derivative is denoted either as 'd2y/dx2' or 'd^2y/dx^2' etc.
          Invalid examples: 'dy/dx(x)' (the parameter 'x' must not be specified with the derivative), 'd2y/d3x' (incorrect derivative order)
          The name of the parameter has to be a valid python identifier like 'x', 'x1', 'z' etc.
        - Prime notation: The derivatives of 'y' are denoted as y', y'', y''' etc.
          Invalid example: "y'(x)" (the parameter 'x' must not be specified with the derivative)
          If only prime notation is used 'x' is used as the default parameter name.
          
        Parameters
        ----------
        ode: A string containing an ordinary differential equation.
            See above for syntax rules.
            E.g.: "y' = 2 * y + 1", 'dy/dx = 2 * y + 1', "y + y' + d^2y/dx^2 = sin(x)"
            
        interval: The interval to solve the ode in.
            E.g.: (0, 1), ('a', 'b')

        bcs: The boundary condition(s) for the target function and all but its highest order derivative.
            See above for general syntax rules.
            A bc must be an equation containing the target function and/or any of its derivatives (using prime notation) evaluated at one of the interval boundaries.
            Valid examples for interval (0, 1): "y(1)=5", "K_p * v'(0)**2 = 2 * p / (r * U**2) - y(0)"
            Invalid examples: 'y=5' (no boundary specified), 'dy/dx(1)=0' (can't use Leibnitz notation for bcs)

        initial_guess: The initial guess for the target function and all but its highest order derivative.
            See above for general syntax rules.
            Can make use of the target function's parameter (e.g. 'x'). The parameter will have type 'numpy.ndarray'.
            Therefor, functions from module 'numpy' should be prefered over those from module 'math'.
            If no initial guess is provided it is set to 0 for the target and all derivatives.
            Valid examples for parameter 'x': 0, 17, 'x', 'x**2', 'U - U * x'
        """
        
        self.target_name : str = None
        self.parameter : str = None
        self.highest_derivative : int = None

        def set_fields(target, param, order, full_match):
            full_match = full_match.group()
            if self.target_name is None:
                self.target_name = target
            assert self.target_name == target, f"target name '{self.target_name}' does not match '{target}' in '{full_match}'"
            
            if param is not None:
                if self.parameter is None:
                    self.parameter = param
                assert self.parameter == param, f"parameter name '{self.parameter}' does not match '{param}' in '{full_match}'"

            if self.highest_derivative is None:
                self.highest_derivative = order
            self.highest_derivative = max(self.highest_derivative, order)
            
            nonlocal ode
            ode = ode.replace(full_match, self.target_derivative_python(order), 1)

        # replace derivatives with python syntax
        for match in re.finditer(self.derivative_leibnitz_notation, ode):
            target, param = match.groups()
            set_fields(target, param, 1, match)
        for match in re.finditer(self.higher_derivative_leibnitz_notation_a, ode):
            order, target, param_order = match.groups()
            assert param_order.endswith(order), f"could not parse '{match.group()}'"
            set_fields(target, param_order.removesuffix(order), int(order), match)
        for match in re.finditer(self.higher_derivative_leibnitz_notation_b, ode):
            order1, target, param, order2 = match.groups()
            assert order1 == order2, f"could not parse '{match.group()}'"
            set_fields(target, param, int(order1), match)
        if self.parameter is None:
            self.parameter = "x"
        for match in re.finditer(self.derivative_prime_notation, ode):
            target, apostrophes = match.groups()
            set_fields(target, None, len(apostrophes), match)
        assert isinstance(self.target_name, str), f"could not find a target function in the ode"
        assert isinstance(self.highest_derivative, int), f"could not find a derivative in the ode"

        # replace object access syntax
        object_access_dict = {}
        for match in re.finditer(self.object_access_pattern, ode):
            original = match.group()
            temp = f"__{original.replace('.', '_')}"
            object_access_dict[temp] = original
            ode = ode.replace(original, temp, 1)

        # use sympy to solve ode for highest order derivative
        lhs_str, rhs_str = ode.split('=')
        lhs = sympy.parse_expr(lhs_str.strip())
        rhs = sympy.parse_expr(rhs_str.strip())
        sympy_dequation = sympy.solve(lhs - rhs, self.target_derivative_python(self.highest_derivative))
        assert len(sympy_dequation) == 1, f"could not parse ode: '{ode}' ({len(sympy_dequation)})"
        ode = str(sympy_dequation[0])

        # reinsert object access syntax
        for temp, original in object_access_dict.items():
            ode = ode.replace(temp, original, 1)
        self.dequation = ode

        # interval
        assert len(interval) == 2, f"the interval '{interval}' must consist of 2 endpoints"
        self.interval = interval

        # boundary conditions
        bc_pattern = fr"\b{self.target_name}('*)\((\w+)\)"
        self.bcs : list[str] = []
        if isinstance(bcs, str):
            bcs = (bcs, )
        for bc in bcs:
            for match in re.findall(bc_pattern, bc):
                assert len(match) == 2, f"could not parse bc: '{bc}'"
                apostrophes, parameter = match
                derivative_order = len(apostrophes)
                assert derivative_order < self.highest_derivative, f"bc contains derivative of order {derivative_order} (max is order {self.highest_derivative-1})"
                if parameter == str(self.interval[0]):
                    postfix = "a"
                elif parameter == str(self.interval[1]):
                    postfix = "b"
                else:
                    raise ValueError(f"boundary condition parameter {parameter} is not an endpoint of interval {self.interval}")
                bc = bc.replace(f"{self.target_name}{'\''*derivative_order}({parameter})", f"{self.target_derivative_python(derivative_order)}_{postfix}", 1)
            lhs, rhs = bc.split("=")
            self.bcs.append(f"{lhs.strip()} - ({rhs.strip()})")

        # initial guess
        if not isinstance(initial_guess, tuple):
            initial_guess = (initial_guess, ) * self.highest_derivative
        assert len(initial_guess) == self.highest_derivative, f"wrong number of initial guesses ({len(initial_guess)}), should be {self.highest_derivative}"
        self.initial_guess = initial_guess

        pass

    def target_derivative_math(self, order : int):
        if order == 0:
            return f"{self.target_name}({self.parameter})"
        elif order == 1:
            return f"d{self.target_name}/d{self.parameter}"
        return f"d^{order}{self.target_name}/d{self.parameter}^{order}"
    def target_derivative_python(self, order : int):
        if order == 0:
            return self.target_name
        elif order == 1:
            return f"d{self.target_name}d{self.parameter}"
        return f"d{order}{self.target_name}d{self.parameter}{order}"

    def generate_scipy_string(self, plot = True, steps = 50, **kwargs):
        """
        Generates a string containing python code which, if executed, solves the BVP.

        Parameters
        ----------
        plot: If True, the solution is plotted after (and if) the calculation finished successfully.
        steps: The number of steps to use for the solver.
        kwargs: Additional keyword arguments to pass to scipy.integrate.solve_bvp().
        """

        wh = "    "

        res = "import numpy as np, scipy, matplotlib.pyplot as plt\n\n"

        res += f"def system({self.parameter}, y):\n"
        res += f"{wh}{", ".join(self.target_derivative_python(order) for order in range(self.highest_derivative))} = y\n"
        
        res += f"{wh}return [\n"
        for order in range(1, self.highest_derivative):
            res += f"{wh * 2}{self.target_derivative_python(order)},\n"
        res += f"{wh * 2}{self.dequation}\n"
        res += f"{wh}]\n\n"

        res += "def bc(ya, yb):\n"
        res += f"{wh}{", ".join(f"{self.target_derivative_python(order)}_a" for order in range(self.highest_derivative))} = ya\n"
        res += f"{wh}{", ".join(f"{self.target_derivative_python(order)}_b" for order in range(self.highest_derivative))} = yb\n"
        res += f"{wh}return [\n{wh * 2}"
        res += f",\n{wh * 2}".join(self.bcs)
        res += f"\n{wh}]\n\n"

        res += f"{self.parameter} = np.linspace({self.interval[0]}, {self.interval[1]}, {steps})    # from, to, steps\n"
        res += f"initial_guess = np.zeros(({self.highest_derivative}, {self.parameter}.size))\n"
        for order, guess in enumerate(self.initial_guess):
            res += f"initial_guess[{order}] = {guess}    # initial guess for {self.target_derivative_math(order)}\n"
        res += "\n"

        res += f"solution = scipy.integrate.solve_bvp(system, bc, {self.parameter}, initial_guess, {', '.join(f'{key}={val}' for key, val in kwargs.items())})\n"
        
        res += f"{self.parameter}_sol = solution.x\n"
        res += f"{", ".join(f"{self.target_derivative_python(order)}_sol" for order in range(self.highest_derivative))} = solution.y\n"

        res += "if not solution.success:\n"
        res += wh + "print(f\"{solution.message = }\")\n\n"

        if plot == True:
            res += "else:    # plot\n"
            res += wh + "plt.figure()\n\n"

            for order in range(self.highest_derivative):
                res += f"{wh}plt.subplot(1, {self.highest_derivative}, {order + 1})\n"
                res += f"{wh}plt.plot({self.parameter}_sol, {self.target_derivative_python(order)}_sol, label=\"{self.target_derivative_math(order)}\")\n"
                res += f"{wh}plt.xlabel(\"{self.parameter}\")\n"
                res += f"{wh}plt.ylabel(\"{self.target_derivative_math(order)}\")\n"
                res += f"{wh}plt.legend()\n\n"

            res += wh + "plt.tight_layout()\n"
            res += wh + "plt.show()\n\n"

        return res

    def run_scipy(self, plot = True, steps = 50, **kwargs):
        """
        Runs the scipy solver and returns the solution.

        Parameters
        ----------
        plot: If True, the solution is plotted after (and if) the calculation finished successfully.
        steps: The number of steps to use for the solver.
        kwargs: Additional keyword arguments to pass to scipy.integrate.solve_bvp().
        """

        def system(x, y):
            locals_dict = {self.target_derivative_python(order) : y[order] for order in range(self.highest_derivative)}
            locals_dict[self.parameter] = x
            return [
                *y[1:],
                eval(self.dequation, None, locals_dict)
                ]

        def bc(ya, yb):
            locals_dict = {f"{self.target_derivative_python(order)}_a" : ya[order] for order in range(self.highest_derivative)}
            locals_dict.update({f"{self.target_derivative_python(order)}_b" : yb[order] for order in range(self.highest_derivative)})
            return [eval(bc, None, locals_dict) for bc in self.bcs]

        x_values = np.linspace(*self.interval, steps)    # from, to, steps
        initial_guess = np.zeros((self.highest_derivative, x_values.size))  # Initial guess for y, y', ...
        for order, guess in enumerate(self.initial_guess):    # initial guess
            if isinstance(guess, str):
                initial_guess[order] = eval(guess, None, {self.parameter : x_values})
            else:
                initial_guess[order] = guess

        solution = scipy.integrate.solve_bvp(system, bc, x_values, initial_guess, **kwargs)
        if not solution.success:
            print(f"{solution.message = }")

        elif plot == True:
            plt.figure()

            for order in range(self.highest_derivative):
                plt.subplot(1, self.highest_derivative, order + 1)
                plt.plot(solution.x, solution.y[order], label=self.target_derivative_math(order))
                plt.xlabel(self.parameter)
                plt.ylabel(self.target_derivative_math(order))
                plt.legend()

            plt.tight_layout()
            plt.show()

        return solution.x, solution.y


if __name__ == "__main__":
    import math

    Reynolds = lambda v: v * d / nu
    def λ_impl(Re):
        Re = max(Re, 0.001)
        iterative = lambda lambda_guess: 1 / math.sqrt(lambda_guess[0]) + 2 * math.log10(2.51 / Re / math.sqrt(lambda_guess[0]) + k / d / 3.71)
        if Re <= 2300:
            return 64 / Re
        return scipy.optimize.fsolve(iterative, 0.02)[0]
    λ = np.vectorize(λ_impl)

    n = 2000
    nozzles = 100
    pumpendruck = 4*10**5
    g = 9.81
    L = 100.0
    d = 0.03
    d_D = 0.51 * 10**-3
    alpha = np.pi / 8
    V_0 = 1.5 * 10**-4
    k = 0.15 * 10**-3 # Rauheit, schätzwert von wikipedia
    nu = 10**-6
    rho = 1000
    dp = 4 * 10**5
    V_D = V_0 / nozzles
    A = d**2 * np.pi / 4
    A_D_s = d_D**2 * np.pi / 4
    U = V_0 / A
    u_D = V_D / (d_D**2 * np.pi / 4)
    ceta = dp / (rho * u_D**2 / 2)
    Re_0 = Reynolds(U)
    K_p = A**2 / A_D_s**2 * ceta / L**2
    K_g = g * L / U**2

    ode = NumericalODESolver(ode="v*v' = -K_p*dv/dx*d2v/dx2-λ(Reynolds(v))*v**2/2*L/d+d/Re_0/L*v''-K_g*math.sin(alpha)",
                                 interval=(0, 1),
                                 bcs=("v(1)=0", "K_p * v'(0)**2 = pumpendruck/(rho*U**2/2)"),
                                 initial_guess=("U - U * x", -U))


    #ode.run_scipy(steps = n, max_nodes=50000, verbose=2)
    print(ode.generate_scipy_string(steps = n, max_nodes=50000, verbose=2))
    exec(ode.generate_scipy_string(steps = n, max_nodes=50000, verbose=2))