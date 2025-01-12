import openmdao.api as om
import numpy as np
import math


class SellarDis1(om.ExplicitComponent):

    def setup(self):
        self.add_input("x", val=0.0)
        self.add_input("z", val=np.zeros(2))
        self.add_input("y2", val=0.0)
        self.add_output("y1", val=0.0)

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        x = inputs["x"]
        z1 = inputs["z"][0]
        z2 = inputs["z"][1]
        y2 = inputs["y2"]

        outputs["y1"] = z1**2 + z2 + x - 0.2 * y2


class SellarDis2(om.ExplicitComponent):

    def setup(self):
        self.add_input("y1", val=0.0)
        self.add_input("z", val=np.zeros(2))
        self.add_output("y2", val=0.0)

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        y1 = inputs["y1"]
        z1 = inputs["z"][0]
        z2 = inputs["z"][1]
        outputs["y2"] = math.fabs(y1[0]) ** 0.5 + z1 + z2


class SellarMDA(om.Group):
    def setup(self):
        cycle: om.Group = self.add_subsystem("cycle", om.Group(), promotes=["*"])
        cycle.add_subsystem(
            "d1",
            SellarDis1(),
            promotes_inputs=["x", "z", "y2"],
            promotes_outputs=["y1"],
        )
        cycle.add_subsystem(
            "d2",
            SellarDis2(),
            promotes_inputs=["y1", "z"],
            promotes_outputs=["y2"],
        )

        cycle.set_input_defaults("x", 1.0)
        cycle.set_input_defaults("z", np.array([5, 2]))

        cycle.nonlinear_solver = om.NonlinearBlockGS()

        self.add_subsystem(
            "obj_cmp",
            om.ExecComp("obj = x**2 + z[1] + y1 + exp(-y2)", z=np.zeros(2), x=0.0),
            promotes=["x", "y1", "y2", "z", "obj"],
        )

        self.add_subsystem(
            "con_cmp1", om.ExecComp("g1 = 3.16 - y1", y1=0.0), promotes=["y1", "g1"]
        )
        self.add_subsystem(
            "con_cmp2", om.ExecComp("g2 = y2 - 24.0"), promotes=["g2", "y2"]
        )


prob = om.Problem()
prob.model = SellarMDA()

prob.driver = om.ScipyOptimizeDriver()
prob.driver.options["optimizer"] = "SLSQP"
prob.driver.options["tol"] = 1e-8

prob.model.add_design_var("x", lower=0, upper=10)
prob.model.add_design_var("z", lower=0, upper=10)
prob.model.add_objective("obj")
prob.model.add_constraint("g1", upper=0)
prob.model.add_constraint("g2", upper=0)
prob.model.approx_totals()

prob.setup()
prob.set_solver_print(level=0)

prob.run_driver()

print("minimum found at")
print(prob.get_val("x")[0])
print(prob.get_val("z"))

print("minumum objective")
print(prob.get_val("obj")[0])
