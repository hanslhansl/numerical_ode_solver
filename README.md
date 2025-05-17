examples:

```Python
import numerical_ode_solver as nos, numpy as np

"""
Initial value problem on (0, 5):
    y''(x) + 2 * (x) = 0

Initial state:
    y(0) = 1
    y'(0) = 0
    
Solution:
    y(x) = cos(sqrt(2) * t)
"""
ode = nos.IVP(odes="y''+2*y=0",
          interval=(0, 5),
          initial_state=(1, 0))
s = ode.generate_scipy_string(plot=plot, steps=50)
print(s)
exec(s)


"""
Boundary value problem on (0, 1):
    y''(x) + y = 0
    
Boundary conditions:
    y(0) = 0
    y(1) = 1
    
Solution:
    y(x) = csc(1) * sin(x)
"""
ode = nos.BVP(odes="y'' + y = 0",
            interval=(0, 1),
            bcs=("y(0)=0", "y(1)=1"),
            initial_guess=(1, 1))
s = ode.generate_scipy_string(plot=plot)
print(s)
exec(s)


"""
Boundary value problem on (0, 1):
    y''(x) + z''(x) = -sin(x)
    z''(x) - y(x) = cos(x)

Boundary conditions:
    y(0) = 0
    y(1) = 1
    z(0) = 0
    z(1) = 0
    
Solution:
    y(x) = 
    z(x) = 
"""
ode = nos.BVP(odes=("y'' + z'' = -np.sin(x)", "z'' - y = np.cos(x)"),
            interval=(0, 1),
            bcs=("y(0) = 0", "y(1) = 1", "z(0) = 0", "z(1) = 0"))
s = ode.generate_scipy_string(plot=plot)
print(s)
exec(s)
```
