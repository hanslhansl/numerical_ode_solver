import scipy.integrate, re, sympy, numpy as np, sys, math, matplotlib.pyplot as plt
from _common import *


class IVP(ODESolverBase):
    def __init__(self, odes : str | tuple[str,...],
                 interval : tuple[str | int | float, str | int | float],
                 initial_state : None | str | int | float | tuple[str | int | float,...]):
        super().__init__(odes, interval, default_variable="t")

        assert False, "initial_state soll equivalent zu BVP bcs sein"

        # initial state
        assert initial_state == None, "not implemented yet"
        self.initial_state = tuple(1 for _ in range(self.n))

        return

    def generate_scipy_string(self, plot = False, steps : None | int = None, **kwargs):

        "scipy.integrate.solve_ivp()"

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
              )#initial_state=[1]
    s = ode.generate_scipy_string(plot=True, steps=50)
    print(s)
    exec(s)