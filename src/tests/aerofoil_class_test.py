if __name__ == "__main__":
    from database.UIUC_aerofoils import UIUC_DATABASE as UDB
    from classes.aerofoil import Aerofoil
    import matplotlib.pyplot as plt
    bambino_6 = Aerofoil(UDB['bambino6_dat'], n_segments=5)
    bambino_6_upper = bambino_6.getUpperCurve(30)
    bambino_6_lower = bambino_6.getLowerCurve(30)    

    plt.plot(bambino_6.upper_coords[:, 0], bambino_6.upper_coords[:, 1], c="red", label="Original Upper Surface") 
    plt.plot(bambino_6.lower_coords[:, 0], bambino_6.lower_coords[:, 1], c="red", label="Original Lower Surface")     
    plt.plot(bambino_6_upper[:, 0], bambino_6_upper[:, 1], c="blue", label="Upper Bezier")
    plt.plot(bambino_6_lower[:, 0], bambino_6_lower[:, 1], c="orange", label="Lower Bezier")
    plt.scatter(bambino_6.upper_control[:, 0], bambino_6.upper_control[:, 1], c="blue", label="Upper Control Points")
    plt.scatter(bambino_6.lower_control[:, 0], bambino_6.lower_control[:, 1], c="orange", label="Lower Control Points")    
    plt.legend()
    plt.title("Bambino 6 Aerofoil Object Visualized")
    plt.xlim((-1, 2))
    plt.ylim((-1, 1))    
    plt.show()