import scipy, re, sympy, numpy as np, sys, math, matplotlib.pyplot as plt



class NumericalODESolver:
    """
    A wrapper around scipy.integrate.solve_bvp() to solve a boundary value problem (BVP) for an ordinary differential equation (ODE).
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_bvp.html for details.
    It simplifies the process of defining the ODE, boundary conditions and initial guess.
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
                 initial_guess : None | str | int | float | tuple[str | int | float,...] = None,
                 params : None | str | tuple[str,...] = None,
                 params_initial_guess : None | str | int | float | tuple[str | int | float,...] = None):
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
        Invalid examples: 'y(x)' (the variable 'x' must not be specified with the function), '1y' (starts with a digit).

        There are two ways to denote the derivative of the target function:
        - Leibnitz notation: The derivative of 'y' by variable 'x' is denoted as 'dy/dx'. The second derivative is denoted either as 'd2y/dx2' or 'd^2y/dx^2' etc.
          Invalid examples: 'dy/dx(x)' (the variable 'x' must not be specified with the derivative), 'd2y/d3x' (incorrect derivative order)
          The name of the variable has to be a valid python identifier like 'x', 'x1', 'z' etc.
        - Prime notation: The derivatives of 'y' are denoted as y', y'', y''' etc.
          Invalid example: "y'(x)" (the variable 'x' must not be specified with the derivative)
          If only prime notation is used 'x' is used as the default variable name.
          
        Parameters
        ----------
        ode: A string containing an ordinary differential equation.
            See above for syntax rules.
            E.g.: "y' = 2 * y + 1", "dy/dx = 2 * y + 1", "y + y' + d^2y/dx^2 = sin(x)"
            
        interval: The interval to solve the ode in.
            E.g.: (0, 1); ("a", "b")

        bcs: The boundary condition(s) for the target function and all but its highest order derivative.
            See above for general syntax rules.
            A bc must be an equation containing the target function and/or any of its derivatives (using prime notation) evaluated at one of the interval boundaries.
            Valid examples for interval (0, 1): "y(1)=5", "K_p * v'(0)**2 = 2 * p / (r * U**2) - y(0)"
            Invalid examples: "y=5" (no boundary specified), "dy/dx(1)=0" (can't use Leibnitz notation for bcs)

        initial_guess: The initial guess for the target function and all but its highest order derivative.
            See above for general syntax rules.
            Can make use of the target function's variable (e.g. 'x'). The variable will have type 'numpy.ndarray'.
            Therefor, functions from module 'numpy' should be prefered over those from module 'math'.
            If no initial guess is provided it is set to 0 for the target and all derivatives.
            Valid examples for variable "x": (0, 17); ("x,); ("x**2", "U - U * x")

        params: Names of the unknown parameters. If None (default), it is assumed that the problem doesn’t depend on any parameters.
            E.g.: ("a", "b", "c")

        params_initial_guess: The initial guess for the unknown parameters (if any).
            If no initial guess is provided it is set to 0 for all parameters.
            E.g.: (0, 1, 2)
        """
        
        self.target_name : str = None
        self.variable : str = None
        self.highest_derivative : int = None

        def set_fields(target, var, order, full_match):
            full_match = full_match.group()
            if self.target_name is None:
                self.target_name = target
            assert self.target_name == target, f"target name '{self.target_name}' does not match '{target}' in '{full_match}'"
            
            if var is not None:
                if self.variable is None:
                    self.variable = var
                assert self.variable == var, f"variable name '{self.variable}' does not match '{var}' in '{full_match}'"

            if self.highest_derivative is None:
                self.highest_derivative = order
            self.highest_derivative = max(self.highest_derivative, order)
            
            nonlocal ode
            ode = ode.replace(full_match, self._target_derivative_python(order), 1)

        # replace derivatives with python syntax
        for match in re.finditer(self.derivative_leibnitz_notation, ode):
            target, var = match.groups()
            set_fields(target, var, 1, match)
        for match in re.finditer(self.higher_derivative_leibnitz_notation_a, ode):
            order, target, var_order = match.groups()
            assert var_order.endswith(order), f"could not parse '{match.group()}'"
            set_fields(target, var_order.removesuffix(order), int(order), match)
        for match in re.finditer(self.higher_derivative_leibnitz_notation_b, ode):
            order1, target, var, order2 = match.groups()
            assert order1 == order2, f"could not parse '{match.group()}'"
            set_fields(target, var, int(order1), match)
        if self.variable is None:
            self.variable = "x"
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
        sympy_dequation = sympy.solve(lhs - rhs, self._target_derivative_python(self.highest_derivative))
        assert len(sympy_dequation) == 1, f"could not parse ode: '{ode}' ({len(sympy_dequation)})"
        ode = str(sympy_dequation[0])

        # reinsert object access syntax
        for temp, original in object_access_dict.items():
            ode = ode.replace(temp, original, 1)
        self.dequation = ode

        # interval
        assert len(interval) == 2, f"the interval '{interval}' must consist of 2 endpoints"
        self.interval = interval

        # initial guess
        if initial_guess is None:
            initial_guess = 0
        if not isinstance(initial_guess, tuple):
            initial_guess = [initial_guess]
        initial_guess = list(initial_guess)
        if len(initial_guess) < self.highest_derivative:
            initial_guess.extend([0] * (self.highest_derivative - len(initial_guess)))
        assert len(initial_guess) == self.highest_derivative, f"wrong number of initial guesses ({len(initial_guess)}), should be {self.highest_derivative}"
        self.initial_guess = tuple(initial_guess)

        # parameters
        if params is None:
            params = ()
        elif not isinstance(params, tuple):
            params = (params, )
        self.parameters = params
        self.k = len(self.parameters)

        # initial guess for parameters
        if params_initial_guess is None:
            params_initial_guess = 0
        if not isinstance(params_initial_guess, tuple):
            params_initial_guess = [params_initial_guess]
        params_initial_guess = list(params_initial_guess)
        if len(params_initial_guess) < self.k:
            params_initial_guess.extend([0] * (self.k - len(params_initial_guess)))
        assert len(params_initial_guess) == self.k, f"wrong number of initial guesses for parameters ({len(params_initial_guess)}), should be {self.k}"
        self.params_initial_guess = tuple(params_initial_guess)

        # boundary conditions
        bc_pattern = fr"\b{self.target_name}('*)\((\w+)\)"
        self.bcs : list[str] = []
        if isinstance(bcs, str):
            bcs = (bcs, )
        assert len(bcs) == self.highest_derivative + self.k, f"wrong number of boundary conditions ({len(bcs)}), should be {self.highest_derivative + self.k}"
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
                bc = bc.replace(f"{self.target_name}{'\''*derivative_order}({parameter})", f"{self._target_derivative_python(derivative_order)}_{postfix}", 1)
            lhs, rhs = bc.split("=")
            self.bcs.append(f"{lhs.strip()} - ({rhs.strip()})")

        pass

    def _target_derivative_math(self, order : int):
        if order == 0:
            return f"{self.target_name}({self.variable})"
        elif order == 1:
            return f"d{self.target_name}/d{self.variable}"
        return f"d^{order}{self.target_name}/d{self.variable}^{order}"
    def _target_derivative_python(self, order : int):
        if order == 0:
            return self.target_name
        elif order == 1:
            return f"d{self.target_name}d{self.variable}"
        return f"d{order}{self.target_name}d{self.variable}{order}"

    def generate_scipy_string(self, plot = True, steps = 50, **kwargs):
        """
        Generates a string containing python code which, if executed, solves the BVP.
        Either plug it into exec() or copy it to a new file and run it.
        If exec() is used the result of solve_bvp() can be accessed as 'solution'.

        Parameters
        ----------
        plot: If True, the solution is plotted afterwards (if the calculation finished successfully).
        steps: The number of steps to use for the solver.
        kwargs: Additional keyword arguments to pass to scipy.integrate.solve_bvp().
        """

        wh = "    "

        res = "import numpy as np, scipy, matplotlib.pyplot as plt\n\n"
        
        if self.k > 0:
            res += f"def system({self.variable}, y, p):\n"
            res += f"{wh}{' '.join(f'{p},' for p in self.parameters)} = p\n"
        else:
            res += f"def system({self.variable}, y):\n"
        res += f"{wh}{' '.join(f'{self._target_derivative_python(order)},' for order in range(self.highest_derivative))} = y\n"

        res += f"{wh}return [\n"
        for order in range(1, self.highest_derivative):
            res += f"{wh * 2}{self._target_derivative_python(order)},\n"
        res += f"{wh * 2}{self.dequation}\n"
        res += f"{wh}]\n\n"

        if self.k > 0:
            res += "def bc(ya, yb, p):\n"
            res += f"{wh}{' '.join(f'{p},' for p in self.parameters)} = p\n"
        else:
            res += "def bc(ya, yb):\n"
        res += f"{wh}{" ".join(f"{self._target_derivative_python(order)}_a," for order in range(self.highest_derivative))} = ya\n"
        res += f"{wh}{" ".join(f"{self._target_derivative_python(order)}_b," for order in range(self.highest_derivative))} = yb\n"
        res += f"{wh}return [\n{wh * 2}"
        res += f",\n{wh * 2}".join(self.bcs)
        res += f"\n{wh}]\n\n"

        res += f"{self.variable} = np.linspace({self.interval[0]}, {self.interval[1]}, {steps})    # from, to, steps\n"
        res += f"initial_guess = np.zeros(({self.highest_derivative}, {self.variable}.size))\n"
        for order, guess in enumerate(self.initial_guess):
            res += f"initial_guess[{order}] = {guess}    # initial guess for {self._target_derivative_math(order)}\n"
        if self.k > 0:
            res += f"parameters_initial_guess = [{', '.join(f'{p}' for p in self.params_initial_guess)}]\n"
        res += "\n"

        res += f"solution = scipy.integrate.solve_bvp(system, bc, {self.variable}, initial_guess, {"parameters_initial_guess, " if self.k > 0 else ""} {', '.join(f'{key}={val}' for key, val in kwargs.items())})\n"
        
        res += f"{self.variable}_sol = solution.x\n"
        res += f"{" ".join(f"{self._target_derivative_python(order)}_sol," for order in range(self.highest_derivative))} = solution.y\n"

        res += "if not solution.success:\n"
        res += wh + "print(f\"{solution.message = }\")\n\n"

        if plot == True:
            res += "else:    # plot\n"
            res += wh + "plt.figure()\n\n"

            for order in range(self.highest_derivative):
                res += f"{wh}plt.subplot(1, {self.highest_derivative}, {order + 1})\n"
                res += f"{wh}plt.plot({self.variable}_sol, {self._target_derivative_python(order)}_sol, label=\"{self._target_derivative_math(order)}\")\n"
                res += f"{wh}plt.xlabel(\"{self.variable}\")\n"
                res += f"{wh}plt.ylabel(\"{self._target_derivative_math(order)}\")\n"
                res += f"{wh}plt.legend()\n\n"

            res += wh + "plt.tight_layout()\n"
            res += wh + "plt.show()\n\n"

        return res



if __name__ == "__main__":
    n = 2000
    
    init = np.vectorize(lambda x: 1 if x < 0.5 else -1)
    ode = NumericalODESolver("y'' + k**2 * y = 0", (0, 1), ("y(0)=0", "y(1)=0", "y'(0) = k"), "init(x)", "k", 6)

    print(ode.generate_scipy_string(steps = n, max_nodes=50000, verbose=2))
    exec(ode.generate_scipy_string(steps = n, max_nodes=50000, verbose=2))
    

    sys.exit()
    Reynolds = lambda v: v * d / nu
    def λ_impl(Re):
        Re = max(Re, 0.001)
        iterative = lambda lambda_guess: 1 / math.sqrt(lambda_guess[0]) + 2 * math.log10(2.51 / Re / math.sqrt(lambda_guess[0]) + k / d / 3.71)
        if Re <= 2300:
            return 64 / Re
        return scipy.optimize.fsolve(iterative, 0.02)[0]
    λ = np.vectorize(λ_impl)

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
                                 bcs=("v(1)=0", "K_p * v'(0)**2 = pumpendruck/(rho*U**2/2)", "v(0)=0.7"),#
                                 initial_guess=("U - U * x", -U),
                                 params=("p",))


    #ode.run_scipy(steps = n, max_nodes=50000, verbose=2)
    print(ode.generate_scipy_string(steps = n, max_nodes=50000, verbose=2))
    exec(ode.generate_scipy_string(steps = n, max_nodes=50000, verbose=2))