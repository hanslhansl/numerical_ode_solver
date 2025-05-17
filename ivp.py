import scipy.integrate, re, sympy, numpy as np, sys, math, matplotlib.pyplot as plt
from _common import *


class IVP(ODESolverBase):
    """
    A wrapper around scipy.integrate.solve_ivp() to solve an initial value problem (IVP) for a system of ordinary differential equations (ODEs).
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html for details.
    It simplifies the process of defining the ODEs and initial state.

    Regarding syntax
    ----------------
    See class BVP for an explanation of the syntax rules.

    Parameters
    ----------
    odes: The ordinary differential equation(s).
        Used to define a callback passed as parameter 'fun' to solve_ivp().
        See above for syntax rules.
        If only prime notation is used 't' is used as the default variable name.
        E.g.: "y' = 2 * y + 1", "dy/dt = 2 * y + 1", "y + y' + d^2y/dt^2 = np.sin(t)"
            
    interval: The interval to solve the ode(s) in. 
        Passed as parameter 't_span' to solve_ivp().
        E.g.: (0, 1); ("a", "b")

    initial_state: The initial state (the value at interval[0]) for the target function(s) and all but the highest order derivative(s).
        Passed as parameter 'y0' to solve_ivp().
        See above for general syntax rules.
        If no initial state is provided it is set to 0 for the target function(s) and all derivatives.
        E.g.: (1, 0); "np.sin(1.5)"
    """

    def __init__(self, odes : str | tuple[str,...],
                 interval : tuple[str | int | float, str | int | float],
                 initial_state : None | str | int | float | tuple[str | int | float,...] = None):
        super().__init__(odes, interval, default_variable="t")

        # initial state
        if initial_state is None:
            initial_state = 0
        if not isinstance(initial_state, (tuple, list)):
            initial_state = [initial_state] * self.n
        assert len(initial_state) == self.n, f"wrong number of initial states ({len(initial_state)}), should be {self.n}"
        self.initial_state = tuple(initial_state)

        return

    def generate_scipy_string(self, plot = False, steps : None | int = None, **kwargs):
        """
        Generates a string containing python code which, if executed, solves the IVP.
        Either plug it into exec() or copy it to a new file and run it.
        If exec() is used the result of solve_ivp() can be accessed as 'solution'.

        Parameters
        ----------
        plot: If True, the solution is plotted afterwards (if the calculation finished successfully).
        steps: If provided, the number of steps to use for the solver.
        kwargs: Additional keyword arguments to pass to scipy.integrate.solve_ivp().
            E.g.: method="Radau"
        """

        res = "import numpy as np, scipy\n\n"
        
        res += self._system(False)

        res += self._variable_linspace(steps)
        res += f"initial_state = [{", ".join(str(s) for s in self.initial_state)}]    # initial state for {", ".join(self._all_targets_math)}\n\n"

        res += f"solution = scipy.integrate.solve_ivp(system, {self.interval}, initial_state{f", t_eval={self.variable}" if steps != None else ""}{''.join(f', {key}={val}' for key, val in kwargs.items())})\n"

        res += f"{self.variable}_sol = solution.t\n"
        res += " ".join(target + f"_sol," for target in self._all_targets_python) + " = solution.y\n"

        res += self._error_and_plot_string(plot, False)

        return res


if __name__ == "__main__":
    w = 2
    ode = IVP(odes="y''+w**2*y=0",
              interval=(0, 5),
              initial_state=(1, 0))
    s = ode.generate_scipy_string(plot=True, steps=50)
    print(s)
    exec(s)