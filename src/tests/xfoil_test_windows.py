import subprocess
import os
from parser.parsefoil import *
import time
from database.UIUC_aerofoils import UIUC_DATABASE as UDB
from classes.bezierfoil import BezierFoil


def gen_xfoil_commands(
    folder: str,
    foil_dat_file: str,
    cadd_adj_no: int,
    reynolds: float,
    ncrit: int,
    niter: int,
    alfa: float,
):
    return f"""
    LOAD {folder}/{foil_dat_file}
    GDES CADD {cadd_adj_no} 10 0.0 1.0
    \n
    PCOP
    PPAR n 250
    \n
    OPER
    VISC
    {reynolds}
    VPAR
    N {ncrit} \n
    ITER {niter}
    PACC
    {folder}/Data.dat
    {folder}/Dump.dat
    ALFA {alfa}
    \n
    QUIT
    """


foil = BezierFoil(UDB['a63a108c_dat'], n_segments=10)
foil.save_foil(
    aerofoil_header_name="BezierFoil",
    save_folder="Foil",
    save_filename="Foil.dat",
    points_per_seg=10,
    write_precision=8,
)
if os.path.isfile(os.getcwd() + "/Foil/Data.dat"):
    os.remove(os.getcwd() + "/Foil/Data.dat")

if os.path.isfile(os.getcwd() + "/Foil/Dump.dat"):
    os.remove(os.getcwd() + "/Foil/Dump.dat")

start = time.time()
for i in range(10):
    xfoil_commands = gen_xfoil_commands("Foil", "Foil.dat", 1, 6.2e6, 10, 1000, 0)
    process = subprocess.Popen(
        ["xfoil.exe"],
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    process.communicate(xfoil_commands)
    process.wait()
    
end = time.time()

exec_time = end - start

print(f"10 XFOIL EXECUTIONS TAKE {exec_time} seconds")
