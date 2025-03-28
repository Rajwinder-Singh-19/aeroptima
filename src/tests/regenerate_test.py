import os
import pandas as pd

df = pd.read_csv(
    os.getcwd()
    + "/src/database/naca_perturbed_dataset/NACA_PCA_perturbed_airfoil_data_seed_ah63k127_dat.csv"
)

UPPER_CONTROL = df.iloc[0, :80].to_numpy().reshape(4, 2, 10)
LOWER_CONTROL = df.iloc[0, 80:-2].to_numpy().reshape(4, 2, 10)

if __name__ == "__main__":
    from database.UIUC_aerofoils import UIUC_DATABASE as UDB
    from classes.bezierfoil import BezierFoil
    import matplotlib.pyplot as plt

    a18 = BezierFoil(UDB["ah63k127_dat"], param_method="arc_length", arc_length=0.1)
    #print(a18.upper_control - UPPER_CONTROL)
    #print(a18.lower_control - LOWER_CONTROL)

    print("ORIGINAL UPPER")
    print(a18.upper_control)
    print("ORIGINAL LOWER")
    print(a18.lower_control)

    print("DATA UPPER")
    print(UPPER_CONTROL)
    print("DATA LOWER")    
    print(LOWER_CONTROL)


    # a18.close_curve()
    a18_upper = a18.getUpperCurve(5)
    a18_lower = a18.getLowerCurve(5)

    plt.plot(
        a18.upper_coords[:, 0],
        a18.upper_coords[:, 1],
        c="red",
        label="Original Upper Surface",
    )
    plt.plot(
        a18.lower_coords[:, 0],
        a18.lower_coords[:, 1],
        c="red",
        label="Original Lower Surface",
    )
    plt.plot(a18_upper[:, 0], a18_upper[:, 1], c="blue", label="Upper Bezier")
    plt.plot(a18_lower[:, 0], a18_lower[:, 1], c="orange", label="Lower Bezier")
    plt.scatter(
        a18.upper_control[:, 0],
        a18.upper_control[:, 1],
        c="blue",
        label="Upper Control Points",
    )
    plt.scatter(
        a18.lower_control[:, 0],
        a18.lower_control[:, 1],
        c="orange",
        label="Lower Control Points",
    )
    plt.legend()
    plt.title("A18 Aerofoil Object Visualized")
    plt.xlim((-1, 2))
    plt.ylim((-1, 1))
    plt.show()
