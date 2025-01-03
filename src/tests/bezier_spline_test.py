if __name__ == "__main__":
    from database.UIUC_aerofoils import UIUC_DATABASE as UDB
    import parser.parsefoil as aeroparse
    import bezier.spline as bsp
    import matplotlib.pyplot as plt

    upper, lower = aeroparse.split_surfaces(UDB["bacxxx_dat"])
    upper_control_array = bsp.get_control_tensor(upper, 5, "L-BFGS-B")
    lower_control_array = bsp.get_control_tensor(lower, 5, "L-BFGS-B")

    upper_bezier = bsp.bezier_spline(upper_control_array, 30)
    lower_bezier = bsp.bezier_spline(lower_control_array, 30)

    plt.plot(upper[:, 0], upper[:, 1], c="red", label="Original Upper Curve")
    plt.plot(lower[:, 0], lower[:, 1], c="red", label="Original Lower Curve")
    plt.plot(
        upper_bezier[:, 0], upper_bezier[:, 1], c="blue", label="Upper Bezier Curve"
    )
    plt.plot(
        lower_bezier[:, 0], lower_bezier[:, 1], c="orange", label="Lower Bezier Curve"
    )
    plt.scatter(
        upper_control_array[:, 0],
        upper_control_array[:, 1],
        c="blue",
        label="Upper Control Points",
    )
    plt.scatter(
        lower_control_array[:, 0],
        lower_control_array[:, 1],
        c="orange",
        label="Lower Control Points",
    )
    plt.title("Aerofoil parametrized using cubic bezier splines")
    plt.legend()
    plt.xlim((-1, 2))
    plt.ylim((-1, 1))
    plt.show()
