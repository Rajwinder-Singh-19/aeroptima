import openmdao.api as om

class Paraboloid(om.ExplicitComponent):
    
    """
    Dummy class that takes 2 inputs x and y in order to calculate the paraboloid surface f
    """

    def setup(self):
        """
        Every ExplicitComponent class needs a setup method to declare the inputs and outputs of the system

        PARAMETERS
            self -> Object of the class itself

        RETURNS
            None
        """
        self.add_input("x", val=0.0)
        self.add_input("y", val=0.0)
        self.add_output("f", val=0.0)

    def setup_partials(self):
        """
        Every ExplicitComponent class needs a setup_partials method to declare how the different derivatives of the system variables are calculated

        PARAMETERS
            self -> Object of the class itself

        RETURNS
            None
        """
        self.declare_partials("*", "*")

    def compute(self, inputs, outputs):
        """
        Every ExplicitComponent class needs a compute method to declare the way system computes the outputs from the inputs

        PARAMETERS
            self -> Object of the class itself
            inputs -> Input variables read via inputs[key].
            outputs -> Output variables read via outputs[key].

        RETURNS
            None
        """
        x = inputs["x"]
        y = inputs["y"]
        outputs["f"] = (x - 3.0) ** 2 + x * y + (y + 4.0) ** 2 - 3.0


if __name__ == "__main__":
    model = om.Group() # We need to make a group in order to place the subsystems in place
    model.add_subsystem("Parab", Paraboloid()) # Classes that define a component in the model are placed in model in this way

    prob = om.Problem(model) # In order to get the models to compute, they need to be placed in a problem construct
    prob.setup() # Initialize the problem variable
    prob.set_val("Parab.x", 3.0) # Assign numeric values to the model inputs. Notice the dot operator to signify the subsystem variable
    prob.set_val("Parab.y", -4.0)

    prob.run_model() # Run the calculation

    print(prob["Parab.f"]) # Check the output

    prob.set_val("Parab.x", 5.0)
    prob.set_val("Parab.y", -2.0)

    prob.run_model()

    print(prob["Parab.f"])
