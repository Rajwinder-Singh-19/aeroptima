if __name__ == "__main__":    
    from database.UIUC_aerofoils import UIUC_DATABASE as UDB
    import parser.aerofoil as aeroparse
    import matplotlib.pyplot as plt

    aerofoil_coords = aeroparse.dat2numpy(UDB['naca001035_dat'])

    plt.plot(aerofoil_coords[:,0], aerofoil_coords[:,1], c='b', label = 'NACA001035')
    plt.scatter(aerofoil_coords[:,0], aerofoil_coords[:,1], c='b', label = 'Co-ordinate Points')
    plt.legend()
    plt.xlim((-1, 2))
    plt.ylim((-1, 1))
    plt.show()