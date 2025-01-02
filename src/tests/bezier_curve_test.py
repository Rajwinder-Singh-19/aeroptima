if __name__ == "__main__":
    import numpy as np
    import bezier.cubic as cb
    import matplotlib.pyplot as plt

    control_point_1 = np.array([0, 2])
    control_point_2 = np.array([1, 4])
    control_point_3 = np.array([3, 7])
    control_point_4 = np.array([7, 10])

    control_point_array = np.array(
        [
            control_point_1,
            control_point_2,
            control_point_3,
            control_point_4,
        ]
    )

    cubic_bezier_curve = cb.cubic_bezier_curve(
        *[i for i in control_point_array],
        n_points=100,
    )

    plt.plot(
        cubic_bezier_curve[:, 0],
        cubic_bezier_curve[:, 1],
        c="blue",
        label="Cubic Bezier Curve Interpolation",
    )
    plt.scatter(
        control_point_array[:, 0],
        control_point_array[:, 1],
        c="red",
        label="Control Points",
    )
    plt.plot(
        control_point_array[:, 0],
        control_point_array[:, 1],
        "--",
        c="red",
        label="Linear Interpolation",
    )
    plt.legend()
    plt.title("Cubic Bezier Curve")
    plt.show()
