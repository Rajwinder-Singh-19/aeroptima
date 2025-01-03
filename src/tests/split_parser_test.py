if __name__ == "__main__":
    from database.UIUC_aerofoils import UIUC_DATABASE as UDB
    import parser.parsefoil as aeroparse
    import matplotlib.pyplot as plt

    aerofoil_upper_coords, aerofoil_lower_coords = aeroparse.split_surfaces(
        UDB["a18_dat"]
    )

    plt.plot(
        aerofoil_upper_coords[:, 0],
        aerofoil_upper_coords[:, 1],
        c="blue",
        label="A18 Upper Surface",
    )
    plt.scatter(
        aerofoil_upper_coords[:, 0],
        aerofoil_upper_coords[:, 1],
        c="blue",
        label="Upper Data Points",
    )
    plt.plot(
        aerofoil_lower_coords[:, 0],
        aerofoil_lower_coords[:, 1],
        c="orange",
        label="A18 Lower Surface",
    )
    plt.scatter(
        aerofoil_lower_coords[:, 0],
        aerofoil_lower_coords[:, 1],
        c="orange",
        label="Lower Data Points",
    )
    plt.legend()
    plt.title("Split the aerofoil into upper and lower surfaces")
    plt.xlim((-1, 2))
    plt.ylim((-1, 1))
    plt.show()
