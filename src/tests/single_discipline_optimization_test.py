from openmdao.test_suite.components.paraboloid import Paraboloid

import openmdao.api as om

prob = om.Problem()
prob.model.add_subsystem("parab", Paraboloid(), promotes_inputs=["x", "y"])
prob.model.add_subsystem("const", om.ExecComp("g = x + y"), promotes_inputs=["x", "y"])

prob.model.add_design_var("x", lower=-50, upper=50)
prob.model.add_design_var("y", lower=-50, upper=50)
prob.model.add_objective("parab.f_xy")
prob.model.add_constraint("const.g", lower=-10, upper=0)

prob.model.set_input_defaults("x", 3.0)
prob.model.set_input_defaults("y", -4.0)

prob.driver = om.ScipyOptimizeDriver()
prob.driver.options["optimizer"] = "COBYLA"

prob.setup()
prob.run_driver()

print(prob["parab.f_xy"])
print(prob["parab.x"])
print(prob["parab.y"])
print(prob["const.g"])
