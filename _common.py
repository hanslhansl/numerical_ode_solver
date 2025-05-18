import re, sympy


wh = "    "

identifier_pattern = r"[a-zA-Z_]\w*"
derivative_leibnitz_notation_pattern = rf"\bd({identifier_pattern})/d({identifier_pattern})\b"
higher_derivative_leibnitz_notation_pattern_a = rf"\bd([1-9][0-9]*)({identifier_pattern})/d({identifier_pattern})\b"
higher_derivative_leibnitz_notation_pattern_b = rf"\bd\^([1-9][0-9]*)({identifier_pattern})/d({identifier_pattern})\^([1-9][0-9]*)\b"
derivative_prime_notation_pattern = rf"\b({identifier_pattern})('+)"
object_access_pattern_pattern = rf'\b(?:{identifier_pattern}\.)+{identifier_pattern}\b'


class ODESolverBase:
    def __init__(self, odes : str | tuple[str,...],
                 interval : tuple[str | int | float, str | int | float],
                 
                 default_variable):

        self.targets : dict[str, int] = {}
        self.variable : str = None

        def set_fields(ode_i, target, var, order, full_match):
            full_match = full_match.group()

            self.targets[target] = max(order, self.targets.get(target, order))
            
            if var is not None:
                if self.variable is None:
                    self.variable = var
                assert self.variable == var, f"variable name '{self.variable}' does not match '{var}' in '{full_match}'"
            assert self.variable is not None
            
            odes[ode_i] = odes[ode_i].replace(full_match, self._derivative_python(target, order), 1)
            
        # replace derivatives with python syntax
        if not isinstance(odes, (tuple, list)):
            odes = [odes]
        odes = list(odes)
        assert len(odes) != 0, "no odes provided"
        for i, ode in enumerate(odes):
            for match in re.finditer(derivative_leibnitz_notation_pattern, ode):
                target, var = match.groups()
                set_fields(i, target, var, 1, match)
            for match in re.finditer(higher_derivative_leibnitz_notation_pattern_a, ode):
                order, target, var_order = match.groups()
                assert var_order.endswith(order), f"could not parse '{match.group()}'"
                set_fields(i, target, var_order.removesuffix(order), int(order), match)
            for match in re.finditer(higher_derivative_leibnitz_notation_pattern_b, ode):
                order1, target, var, order2 = match.groups()
                assert order1 == order2, f"could not parse '{match.group()}'"
                set_fields(i, target, var, int(order1), match)
        if self.variable is None:
            self.variable = default_variable
        for i, ode in enumerate(odes):
            for match in re.finditer(derivative_prime_notation_pattern, ode):
                target, apostrophes = match.groups()
                set_fields(i, target, None, len(apostrophes), match)
        self.n = sum(order for order in self.targets.values())
        assert len(self.targets) == len(odes), f"found {len(self.targets)} target function(s) ({", ".join(self._derivative_math(target, 0) for target in self.targets.keys())}) but expected {len(odes)}"

        # find object access syntax
        object_access_list = []
        for i, _ in enumerate(odes):
            for match in re.finditer(object_access_pattern_pattern, odes[i]):
                original = match.group()
                temp = f"__{original.replace('.', '_')}"
                object_access_list.append((original, temp))
        object_access_list.sort(key=lambda x: len(x[0]), reverse=True)

        # replace object access syntax
        for i, _ in enumerate(odes):
            for original, temp in object_access_list:
                odes[i] = odes[i].replace(original, temp)

        # use sympy to solve odes for highest order derivatives
        parsed_equations = []
        for i, _ in enumerate(odes):
            lhs_str, rhs_str = odes[i].split('=')
            lhs = sympy.parse_expr(lhs_str.strip())
            rhs = sympy.parse_expr(rhs_str.strip())
            parsed_equations.append(lhs - rhs)
        sympy_dequations = sympy.solve(parsed_equations, [self._derivative_python(target, order) for target, order in self.targets.items()])
        assert len(sympy_dequations) == len(self.targets)

        # reinsert object access syntax
        odes.clear()
        for target, sympy_dequation in sympy_dequations.items():
            ode = str(sympy_dequation.simplify())
            for original, temp in object_access_list:
                ode = ode.replace(temp, original)
            odes.append(ode)
        self.odes = odes

        # interval
        assert len(interval) == 2, f"the interval '{interval}' must consist of 2 endpoints"
        self.interval = tuple(interval)

        return

    def _derivative_math(self, target : str, order : int):
        return f"{target}{"'"*order}({self.variable})"
        # leibnitz notation
        # if order == 0:
        #     return f"{target}({self.variable})"
        # elif order == 1:
        #     return f"d{target}/d{self.variable}"
        # return f"d^{order}{target}/d{self.variable}^{order}"
    def _derivative_python(self, target : str, order : int):
        if order == 0:
            return target
        elif order == 1:
            return f"d{target}d{self.variable}"
        return f"d{order}{target}d{self.variable}{order}"

    @property
    def _all_targets_python(self):
        return [self._derivative_python(target, order) for target, max_order in self.targets.items() for order in range(max_order)]
    @property
    def _all_targets_math(self):
        return [self._derivative_math(target, order) for target, max_order in self.targets.items() for order in range(max_order)]
    @property
    def _all_derivatives(self):
        return [self._derivative_python(target, order) for target, max_order in self.targets.items() for order in range(1, max_order + 1)]

    def _system(self, with_p : bool):
        res = ""

        if with_p:
            res += f"def system({self.variable}, y, p):\n"
            res += f"{wh}{' '.join(f'{p},' for p in self.parameters)} = p\n"
        else:
            res += f"def system({self.variable}, y):\n"
        res += f"{wh}{"".join(target + ', ' for target in self._all_targets_python)}= y\n"
        
        res += wh + "return [\n"
        for target, max_order in self.targets.items():
            res += "".join(f"{wh * 2}{self._derivative_python(target, order)},\n" for order in range(1, max_order))
        res += "".join(wh*2 + ode + ",\n" for ode in self.odes)
        res += wh + "]\n\n"

        return res
    def _solution(self, param_name : str, has_params : bool):
        res = ""
        
        #res += f"del {self.variable}\n"
        res += f"{self.variable} = solution.{param_name}\n"
        res += " ".join(target + f"," for target in self._all_targets_python) + " = solution.y\n"

        if has_params:
            res += " ".join(f"{p}," for p in self.parameters) + f" = solution.p\n"

        return res
    def _error_and_plot_string(self, plot : bool, plot_highest_derivative : bool):
        res = ""

        res += "\nif not solution.success:\n"
        res += wh + "# error\n"
        res += wh + "print(f\"{solution.message = }\")\n"

        if plot == True:
            res += "else:\n"
            res += wh + "# plot\n"
            res += wh + "import matplotlib.pyplot as plt\n"
            res += wh + "plt.figure()\n\n"

            i = 1
            for target, max_order in self.targets.items():
                for order in range(max_order+plot_highest_derivative):
                    res += f"{wh}plt.subplot(1, {self.n+len(self.targets)*plot_highest_derivative}, {i})\n"
                    res += f"{wh}plt.plot({self.variable}, {self._derivative_python(target, order)}, label=\"{self._derivative_math(target, order)}\")\n"
                    res += f"{wh}plt.xlabel(\"{self.variable}\")\n"
                    res += f"{wh}plt.ylabel(\"{self._derivative_math(target, order)}\")\n"
                    res += f"{wh}plt.legend()\n\n"
                    res += f"{wh}plt.grid(True)\n\n"
                    i += 1
                    
            res += wh + "plt.show()\n"

        return res
