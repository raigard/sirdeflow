# All basic functions needed to run the debris flow model are presented here

import os.path
import numpy as np
from osgeo import gdal
import warnings
import multiprocessing as mp

# Prompted inputs:
print('Welcome to the alpha version of SIR-DEFLOW! \n\n')

while True:  # Assigns different viscosity values to calibration areas in a dictionary
    c_ar = int(input("Enter the number of calibration areas: \n"))
    zone_list = range(1, c_ar + 1)
    zone_vis = []
    area_list = []
    if c_ar > 1:
        for _ in zone_list:
            area_name = "area_" + format(_)
            vis = float(input("Enter fluid cinematic viscosity number for " + area_name + " " + "[N.s/m²]: \n"))
            zone_vis.append(int(vis))
            area_list.append(area_name)
        #        print(zone_vis)
        break
    elif c_ar == 1:
        vis = float(input("Enter fluid cinematic viscosity number for [N.s/m²]: \n"))
        break
    else:
        print("Invalid input. Integer value >= 1 expected \n")

# These variables are common to all calibration areas
dem = input("Enter path to DEM file - e.g.: C:/DEM.tif:\n")
slp = input("Enter path to SLOPE file in degrees - e.g.: C:/SLOPE.tif:\n")

while True:  # Allows to choose between fdr methods
    fdr_ar = input("Enter the desired fdr method (currently available: d8, dinf, mfd): \n")
    if fdr_ar == 'd8':
        fdr = input("Enter path to FLOW DIRECTION file - e.g.: C:/FDR.tif:\n")
        break
    elif fdr_ar == 'dinf':
        fdr = input("Enter path to FLOW DIRECTION file - e.g.: C:/FDR.tif:\n")
        break
    elif fdr_ar == 'mfd':
        break
    else:
        print('Invalid input, try "d8", "dinf", or "mfd" \n')

if fdr_ar == 'd8':
    fdr = gdal.Open(fdr)
    fdr = fdr.ReadAsArray()
    fdr = fdr.astype('float32')
elif fdr_ar == 'dinf':
    fdr = gdal.Open(fdr)
    fdr = fdr.ReadAsArray()
    fdr[:, 0:2] = -9999
    fdr[:, fdr.shape[1] - 2: fdr.shape[1]] = -9999
    fdr[0:2] = -9999
    fdr[fdr.shape[0] - 2:fdr.shape[0]] = -9999
    if not np.any(fdr > np.pi):
        fdr = fdr * (180 / np.pi)
    spread = float(input("Enter minimum spread coefficient (0 - 1.0): \n"))
elif fdr_ar == 'mfd':
    n = float(input("Enter multiple flow direction 'n' factor (1 - 99). Lower values will cause more spreading: \n"))
else:
    pass

while True:  # Allows to choose between different rheology
    rheo = input("Enter flow rheology (currently available: Newtonian (n), Bingham plastic (b), Herschel-Bulkley(hb) "
                 "and dilatant (d)): \n")
    if rheo == 'n':
        break
    elif rheo == 'b':
        z_plug = float(input("Enter plug height (z) in meters: \n "))
        m_hb = 1
        break
    elif rheo == 'hb':
        z_plug = float(input("Enter plug height (z) in meters: \n"))
        m_hb = float(input("Enter flow behavior exponent value (m > 0): \n"))
        break
    elif rheo == 'd':
        n_dil = float(input("Enter flow behavior exponent value (n > 1) \n"))
        break
    else:
        print('Invalid input, try "n", "b", "hb" or "d" \n')

hin = input("Enter path to file indicating initiation areas and failure depths - e.g.: C:/HIN.tif: \n")
pxl = float(input("Enter pixel size in meters: \n "))
# TODO: auto recognition of pixel size
ts = float(input("Enter time step in seconds: \n"))
i_s = int(input("Enter number of iterations to save a flow velocity grid. Enter 0 if this information is not desired."
                "\n"))
while True:
    stop_method = input("Enter stopping method: maximum deposition height (m) or number of iteration (i): \n")
    if stop_method == 'm':
        stop_value = float(input("Enter maximum height difference in meters for stopping criteria:\n"))
        break
    elif stop_method == 'i':
        stop_value = float(input("Enter number of iterations to stop the simulation:\n"))
        break
# TODO: create a error massage for invalid inputs with reprompt option | option to read inputs from a txt file

if c_ar > 1:  # zone identification grid to relate different entries to different calibration areas
    zone_grid = input("Enter path to ZONE GRID - e.g.: C:/ZGRID.tif: \n")
    zone_grid = gdal.Open(zone_grid).ReadAsArray()
else:
    pass


def check_dimension(_dem, _fdr, _hin, _slp):  # simple way to check if dimensions match, failure will interrupt the code
    dim1 = [_dem.shape[0], _dem.shape[1]]
    dim2 = [_slp.shape[0], _slp.shape[1]]
    dim3 = [_fdr.shape[0], _fdr.shape[1]]
    dim4 = [_hin.shape[0], _hin.shape[1]]

    #    coord1 = [_dem.bounds[0], _dem.bounds[1]]
    #    coord2 = [_slp.bounds[0], _slp.bounds[1]]
    #    coord3 = [_fdr.bounds[0], _fdr.bounds[1]]
    #    coord4 = [_hin.bounds[0], _hin.bounds[1]]

    dim = [dim1, dim2, dim3, dim4]
    #    coord = [coord1, coord2, coord3, coord4]  # TODO test coordinate checker
    if dim.count(dim[0]) == len(dim):  # and coord.count(coord[0]) == len(coord):
        print("Number of rows and columns from inputs matches. Starting calculations")
    else:
        raise Exception("Number of rows and/or columns from inputs does not match")


def check_dimension_mfd(_dem, _hin, _slp):  # mfd doesn't have a fdr grid
    dim1 = [_dem.shape[0], _dem.shape[1]]
    dim2 = [_slp.shape[0], _slp.shape[1]]
    dim4 = [_hin.shape[0], _hin.shape[1]]
    dim = [dim1, dim2, dim4]
    #    coord = [coord1, coord2, coord3, coord4]  # TODO test coordinate checker
    if dim.count(dim[0]) == len(dim):  # and coord.count(coord[0]) == len(coord):
        print("Number of rows and columns from inputs matches. Starting calculations")
    else:
        raise Exception("Number of rows and/or columns from inputs does not match")


# this first section is responsible for files opening and reading


height = gdal.Open(hin)

# Get information in order to save output file
gt = height.GetGeoTransform()
projection = height.GetProjection()

dem = gdal.Open(dem)
slp = gdal.Open(slp)

dem = dem.ReadAsArray()
dem = dem.astype('float32')
dem[dem == -9999] = 0.0001
dem[:, 0:2] = 0.0001
dem[:, dem.shape[1] - 2: dem.shape[1]] = 0.0001
dem[0:2] = 0.0001
dem[dem.shape[0] - 2:dem.shape[0]] = 0.0001

slp = slp.ReadAsArray()
slp = slp.astype('float32')
slp[slp == -9999] = 0
slp = np.sin(slp * np.pi / 180)
slp[:, 0:2] = 0
slp[:, slp.shape[1] - 2: slp.shape[1]] = 0
slp[0:2] = 0
slp[slp.shape[0] - 2:slp.shape[0]] = 0

hin = height.ReadAsArray()
hin = hin.astype('float32')
hin[hin == -9999] = 0.0001
hin[hin == 0] = 0.0001
hin[:, 0:2] = 0.0001
hin[:, hin.shape[1] - 2: hin.shape[1]] = 0.0001
hin[0:2] = 0.0001
hin[hin.shape[0] - 2:hin.shape[0]] = 0.0001
height = None
# with np.errstate(divide='ignore', invalid='ignore'):
#    hin = ((hin != -9999) / (
#            hin != -9999)) * hin  # flag -9999 and by algebra turn it to a NaN - works only for floats

if fdr_ar == 'mfd':
    check_dimension_mfd(dem, hin, slp)
else:
    check_dimension(dem, fdr, hin, slp)


def get_grid(grid, row, column):  # fetches a 3x3 window of data to perform analysis
    c_mid = grid.shape[1] - 1 > column > 0
    c_left = column - 1 < 0
    c_right = grid.shape[1] - 1 == column
    r_mid = grid.shape[0] - 1 > row > 0
    r_top = row - 1 < 0
    r_bot = grid.shape[0] - 1 == row
    if c_mid and r_mid:
        _grid = grid[row - 1:row + 2, column - 1:column + 2]
        return _grid
    elif r_top and c_mid:
        _grid = grid[row:row + 2, column - 1:column + 2]
        _grid = np.insert(_grid, 0, 0, axis=0)
        return _grid
    elif r_bot and c_mid:
        _grid = grid[row - 1:row + 1, column - 1:column + 2]
        _grid = np.insert(_grid, _grid.shape[0], 0, axis=0)
        return _grid
    elif c_left and r_mid:
        _grid = grid[row - 1:row + 2, column:column + 2]
        _grid = np.insert(_grid, 0, 0, axis=1)
        return _grid
    elif c_right and r_mid:
        _grid = grid[row - 1:row + 2, column - 1:column + 1]
        _grid = np.insert(_grid, _grid.shape[1], 0, axis=1)
        return _grid
    elif c_right and r_top:
        _grid = grid[row:row + 2, column - 1: column + 1]
        _grid = np.insert(_grid, _grid.shape[0], 0, axis=1)
        _grid = np.insert(_grid, 0, 0, axis=0)
        return _grid
    elif c_right and r_bot:
        _grid = grid[row - 1:row + 1, column - 1: column + 1]
        _grid = np.insert(_grid, _grid.shape[1], 0, axis=1)
        _grid = np.insert(_grid, _grid.shape[0], 0, axis=0)
        return _grid
    elif c_left and r_top:
        _grid = grid[row:row + 2, column:column + 2]
        _grid = np.insert(_grid, 0, 0, axis=1)
        _grid = np.insert(_grid, 0, 0, axis=0)
        return _grid
    elif c_left and r_bot:
        _grid = grid[row - 1:row + 1, column:column + 2]
        _grid = np.insert(_grid, 0, 0, axis=1)
        _grid = np.insert(_grid, _grid.shape[0], 0, axis=0)
        return _grid
    else:
        raise Exception("Oops! Something went wrong!\n Cause: get_grid function failed required conditions")


class Zones:
    #  Class to assign different parameters to different areas in a DEM
    def __init__(self, a, s, d, v, h):
        self.viscosity = float(v)
        self.flow_height = float(h)
        self.flow_direction = float(d)
        self.slope = float(s)
        self.altitude = float(a)


#    def calc_outflow(self):  # calculates profile mean velocity and outflow
#        q_out = float(pxl*(self.flow_height**2)/(3*self.viscosity))
#       return q_out

#    def calc_height(self, inflow, outflow):
#        q_in = inflow
#        q_out = outflow
#        self.flow_height += (ts/pxl)*(q_in - q_out)
#        return self.flow_height

'''
This section contains a simple approach to calculate multiple flow directions from Freeman (1991)
'''


def mfd_slope(dem_grid, pxl):
    mfd_slp = np.empty((dem_grid.shape[0], dem_grid.shape[1], 9))
    img_h = dem.shape[0]
    img_w = dem.shape[1]
    diag = 2 * (2 * ((pxl / 2) ** 2)) ** 1 / 2
    non_diag = pxl
    for row_i in range(0, img_h):
        for col_j in range(0, img_w):
            if dem[row_i, col_j] == 0.0001 or dem[row_i, col_j] == -9999:
                mfd_slp[row_i, col_j] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
            else:
                aux_grid = get_grid(dem, row_i, col_j)
                mfd_slp[row_i, col_j] = np.array([(- aux_grid[0, 0] + aux_grid[1, 1]) / diag,
                                                  (- aux_grid[0, 1] + aux_grid[1, 1]) / non_diag,
                                                  (- aux_grid[0, 2] + aux_grid[1, 1]) / diag,
                                                  (- aux_grid[1, 0] + aux_grid[1, 1]) / non_diag,
                                                  0,
                                                  (- aux_grid[1, 2] + aux_grid[1, 1]) / non_diag,
                                                  (- aux_grid[2, 0] + aux_grid[1, 1]) / diag,
                                                  (- aux_grid[2, 1] + aux_grid[1, 1]) / non_diag,
                                                  (- aux_grid[2, 2] + aux_grid[1, 1]) / diag])
                mfd_slp[row_i, col_j] = np.where(mfd_slp[row_i, col_j] < 0, 0, mfd_slp[row_i, col_j])
    return mfd_slp


def mfd_fractions(mfd_slp, n_factor):  # fractions are based on Freeman's formulae, then they're normalized with min/max
    n = n_factor
    img_h = dem.shape[0]
    img_w = dem.shape[1]
    for row_i in range(0, img_h):
        for col_j in range(0, img_w):
            frac = mfd_slp[row_i, col_j]
            p1, p2, p3, p4, p5, p6, p7, p8 = frac[0], frac[1], frac[2], frac[3], frac[5], frac[6], frac[7], frac[8]
            f1 = (p2 ** n + p3 ** n + p4 ** n + p5 ** n + p6 ** n + p7 ** n + p8 ** n)
            if f1 == 0:
                d1 = p1 ** n
            else:
                d1 = (p1 ** n) / f1
            f2 = (p1 ** n + p3 ** n + p4 ** n + p5 ** n + p6 ** n + p7 ** n + p8 ** n)
            if f2 == 0:
                d2 = p2 ** n
            else:
                d2 = (p2 ** n) / f2
            f3 = (p2 ** n + p1 ** n + p4 ** n + p5 ** n + p6 ** n + p7 ** n + p8 ** n)
            if f3 == 0:
                d3 = p3 ** n
            else:
                d3 = (p3 ** n) / f3
            f4 = (p2 ** n + p3 ** n + p1 ** n + p5 ** n + p6 ** n + p7 ** n + p8 ** n)
            if f4 == 0:
                d4 = p4 ** n
            else:
                d4 = (p4 ** n) / f4
            f5 = (p2 ** n + p3 ** n + p4 ** n + p1 ** n + p6 ** n + p7 ** n + p8 ** n)
            if f5 == 0:
                d5 = p5 ** n
            else:
                d5 = (p5 ** n) / f5
            f6 = (p2 ** n + p3 ** n + p4 ** n + p5 ** n + p1 ** n + p7 ** n + p8 ** n)
            if f6 == 0:
                d6 = p6 ** n
            else:
                d6 = (p6 ** n) / f6
            f7 = (p2 ** n + p3 ** n + p4 ** n + p5 ** n + p6 ** n + p1 ** n + p8 ** n)
            if f7 == 0:
                d7 = p7 ** n
            else:
                d7 = (p7 ** n) / f7
            f8 = (p2 ** n + p3 ** n + p4 ** n + p5 ** n + p6 ** n + p7 ** n + p1 ** n)
            if f8 == 0:
                d8 = p8 ** n
            else:
                d8 = (p8 ** n) / f8
            frac = np.array([d1, d2, d3, d4, 0, d5, d6, d7, d8])
            sum = np.sum(frac)
            if sum > 0:
                frac = frac / sum
            else:
                frac[:] = 0
                print('Cell ' + '(' + str(row_i) + ',' + str(col_j) +
                      ') is flat, has no downslope direction or is no_data')
            frac = np.where(frac < 0.0001, 0, frac)
            mfd_slp[row_i, col_j] = frac
    return mfd_slp


'''
This function calculates the fractions of flow towards each cell of a grid; each cell of the result contains a 3x3
grid with the fractions located on the direction of the flow. This format was chosen to also match the output of a 
Multiple Flow Direction (MFD) algorithm.
'''


def dinf_f1(a1, a2):  # cell contribution for deterministic infinity method
    f1 = a2 / (a1 + a2)
    return f1


def dinf_f2(a1, a2):
    f2 = a1 / (a1 + a2)
    return f2


def dinf_fractions(fdr_grid, spread):  # calculates contribution to each neighbor pixel
    fdr_inf = fdr_grid
    fdr_h = fdr_inf.shape[0]
    fdr_w = fdr_inf.shape[1]
    dinf = np.empty((fdr_grid.shape[0], fdr_grid.shape[1], 9))
    for row_i in range(0, fdr_h):
        for col_j in range(0, fdr_w):
            angle = fdr_inf[row_i, col_j]
            if 45 > angle >= 0:
                a1 = angle
                a2 = 45 - angle
                f1 = dinf_f1(a1, a2)
                f2 = dinf_f2(a1, a2)
                dinf[row_i, col_j] = np.array([0, 0, f2, 0, 0, f1, 0, 0, 0])
            elif 90 > angle >= 45:
                a1 = angle - 45
                a2 = 90 - angle
                f1 = dinf_f1(a1, a2)
                f2 = dinf_f2(a1, a2)
                dinf[row_i, col_j] = np.array([0, f2, f1, 0, 0, 0, 0, 0, 0])
            elif 135 > angle >= 90:
                a1 = angle - 90
                a2 = 135 - angle
                f1 = dinf_f1(a1, a2)
                f2 = dinf_f2(a1, a2)
                dinf[row_i, col_j] = np.array([f2, f1, 0, 0, 0, 0, 0, 0, 0])
            elif 180 > angle >= 135:
                a1 = angle - 135
                a2 = 180 - angle
                f1 = dinf_f1(a1, a2)
                f2 = dinf_f2(a1, a2)
                dinf[row_i, col_j] = np.array([f1, 0, 0, f2, 0, 0, 0, 0, 0])
            elif 225 > angle >= 180:
                a1 = angle - 180
                a2 = 225 - angle
                f1 = dinf_f1(a1, a2)
                f2 = dinf_f2(a1, a2)
                dinf[row_i, col_j] = np.array([[0, 0, 0, f1, 0, 0, f2, 0, 0]])
            elif 270 > angle >= 225:
                a1 = angle - 225
                a2 = 270 - angle
                f1 = dinf_f1(a1, a2)
                f2 = dinf_f2(a1, a2)
                dinf[row_i, col_j] = np.array([[0, 0, 0, 0, 0, 0, f1, f2, 0]])
            elif 315 > angle >= 270:
                a1 = angle - 270
                a2 = 315 - angle
                f1 = dinf_f1(a1, a2)
                f2 = dinf_f2(a1, a2)
                dinf[row_i, col_j] = np.array([[0, 0, 0, 0, 0, 0, 0, f1, f2]])
            elif 360 >= angle >= 315:
                a1 = angle - 315
                a2 = 360 - angle
                f1 = dinf_f1(a1, a2)
                f2 = dinf_f2(a1, a2)
                dinf[row_i, col_j] = np.array([[0, 0, 0, 0, 0, f2, 0, 0, f1]])
            else:
                pass
            if spread > 0:
                if np.any((dinf[row_i, col_j] <= spread)):
                    dinf[row_i, col_j] = np.where(dinf[row_i, col_j] <= spread, 0, dinf[row_i, col_j])
                    dinf[row_i, col_j] = np.where(dinf[row_i, col_j] >= spread, 1, dinf[row_i, col_j])
    return dinf


def d8_inflow(fdr_grid, q_grid):  # if fdr points towards the center cell, it will sum flow values from q_grid
    f = fdr_grid
    q = q_grid
    _dir = np.array([[2, 4, 8],
                     [1, -99999, 16],
                     [128, 64, 32]],
                    int)
    inflow = np.nansum(q[f == _dir])
    return inflow


'''
def param_dist(zone_viscosity):  # create objects to each zone and assigns to a list 
    obj_list = []
    zv = zone_viscosity
    _ = 0
    for _ in zv:
        obj_list.append(Zones(dem, slp, fdr, _, hin))
    return obj_list
'''


def calc_height(local_h, inflow, outflow):
    q_in = inflow
    q_out = outflow
    flow_height = local_h + (ts / pxl) * (q_in - q_out)
    return flow_height


def dinf_inflow_pointer(fdr_grid):  # fetch contributions from pixels
    fdr_inf = fdr_grid
    right = fdr_inf[1, 2, 3]
    ur = fdr_inf[0, 2, 6]
    up = fdr_inf[0, 1, 7]
    ul = fdr_inf[0, 0, 8]
    left = fdr_inf[1, 0, 5]
    dl = fdr_inf[2, 0, 2]
    down = fdr_inf[2, 1, 1]
    dr = fdr_inf[2, 2, 0]
    return right, ur, up, ul, left, dl, down, dr


def dinf_inflow(right, ur, up, ul, left, dl, down, dr, q_grid):  # calculates flow partition to both cells
    cont1 = right * q_grid[1, 2]
    cont2 = ur * q_grid[0, 2]
    cont3 = up * q_grid[0, 1]
    cont4 = ul * q_grid[0, 0]
    cont5 = left * q_grid[1, 0]
    cont6 = dl * q_grid[2, 0]
    cont7 = down * q_grid[2, 1]
    cont8 = dr * q_grid[2, 2]
    inflow = np.round((cont1 + cont2 + cont3 + cont4 + cont5 + cont6 + cont7 + cont8), decimals=4)
    return inflow


def d8_height(fdr_grid, q_grid, pxl_h0):
    f = fdr_grid
    _inflow = d8_inflow(f, q_grid)
    _outflow = q_grid[1, 1]
    _h = calc_height(pxl_h0, _inflow, _outflow)
    return _h


def dinf_height(fdr_grid, q_grid, pxl_h0):
    f = fdr_grid
    right, ur, up, lr, left, dl, down, dr = dinf_inflow_pointer(f)
    _inflow = dinf_inflow(right, ur, up, lr, left, dl, down, dr, q_grid)
    _outflow = q_grid[1, 1]
    _h = calc_height(pxl_h0, _inflow, _outflow)
    return _h


def qout_newtonian(flow_height, viscosity, slp_grid):  # calculates profile mean velocity and outflow
    flow_height = np.where(flow_height <= 0.0001, 0, flow_height)
    q_out = slp_grid * 9.806 * (flow_height ** 3) / (3 * viscosity)
    q_out = np.where(q_out * ts > flow_height * pxl, flow_height * pxl / ts, q_out)
    return q_out


def qout_dil(flow_height, viscosity, slp_grid, par_n):
    n = par_n
    flow_height = np.where(flow_height <= 0.0001, 0, flow_height)
    q_out = flow_height * ((n / (n + 1)) *
                           (((9.806 * flow_height ** (n + 1)) * slp_grid / viscosity) ** (1 / n)) * (
                                       1 - (n / (2 * n + 1))))
    q_out = np.where(q_out * ts > flow_height * pxl, flow_height * pxl / ts, q_out)
    return q_out


def qout_hb(flow_height, viscosity, slp_grid, par_m, z_plug):  # will need unit weight of the mixture
    z = z_plug
    flow_height = np.where(flow_height <= z, z, flow_height)
    m = par_m
    q_out = flow_height * ((m / (m + 1)) *
                           ((9.806 * (flow_height ** (m + 1)) * slp_grid / viscosity) ** (1 / m)) *
                           (1 - (m / (2 * m + 1)) * ((flow_height - z) / flow_height)))
    q_out = np.where(q_out * ts > flow_height * pxl, flow_height * pxl / ts, q_out)
    return q_out


''' this section will be useful when calibration areas are implemented
if c_ar > 1:
    zone_list = param_dist(zone_vis)
else:
    zone_list = Zones(dem, slp, fdr, vis, hin)
'''

'''
The first section of the code defines most functions needed for calculations. Further commands define the model behavior
'''


def d8_method(hin, fdr, max_dh, vis, slp, i_s):  # calculates flow propagation through D8
    h0 = hin
    h1 = np.full(hin.shape, 0.0001, dtype=float)
    dh = float(1.0)  # arbitrary value to ensure "while" condition is met
    iter = 0
    img_h = hin.shape[0]
    img_w = hin.shape[1]
    k = 0
    u = []
    while np.nanmax(dh) > max_dh:
        # outer loop defines stop criteria and calculates the height difference between time steps
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if rheo == 'n':
                q0 = qout_newtonian(h0, vis, slp)
            elif rheo == 'b' or rheo == 'hb':
                q0 = qout_hb(h0, vis, slp, m_hb, z_plug)
            elif rheo == 'd':
                q0 = qout_dil(h0, vis, slp, n_dil)
        h1[:] = 0.0001
        iter += 1
        print('Calculation step ' + str(iter))
        for row_i in range(0, img_h):
            for col_j in range(0, img_w):
                # inner loop calculates the new height for each cell
                if not np.any(get_grid(h0, row_i, col_j) != 0.0001):
                    pass
                else:
                    h1[row_i, col_j] = d8_height(
                        get_grid(fdr, row_i, col_j),
                        get_grid(q0, row_i, col_j),
                        h0[row_i, col_j]
                    )
        if iter - k == i_s:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                u_iter = np.round(q0 / h0, decimals=4)
                u_iter[u_iter == np.NaN] = -9999
            u.append(u_iter)
            k = iter
        h_1 = np.nan_to_num(h1)
        h_0 = np.nan_to_num(h0)
        dh = h_1 - h_0
        h0 = h1 * 1  # new initial height must be equal to h1 on subsequent calculations
    h0[h0 == 0.0001] = -9999
    h0[h0 == np.NaN] = -9999
    h0 = np.round(h0, decimals=4)
    time = str(iter * ts) + ' s'
    tt = str(iter * ts) + ' until stoppage.'
    print(tt)
    return h0, time, u


def dinf_method(hin, max_dh, fdr, vis, slp, spread, i_s):  # calculates flow propagation throgh deterministic infinity
    h0 = hin
    h1 = np.full(hin.shape, 0, dtype=float)
    dh = float(1.0)  # arbitrary value to ensure "while" condition is met
    iter = 0
    img_h = hin.shape[0]
    img_w = hin.shape[1]
    fdr_inf = fdr.astype(float)
    fdr_inf = dinf_fractions(fdr_inf, spread)
    u = []
    k = 0
    if stop_method == 'm':
        while np.nanmax(dh) > max_dh:
            # outer loop defines stop criteria and calculates the height difference between time steps
            if rheo == 'Newtoninan' or 'n' or 'newtonian':
                q0 = qout_newtonian(h0, vis, slp)
            elif rheo == 'Bingham' or 'Bingham plastic' or 'b' or 'bingham plastic' or 'hb' or 'Herschel-Bulkley':
                q0 = qout_hb(h0, vis, slp, m_hb, z_plug)
            elif rheo == 'd' or 'dilatant' or 'Dilatant':
                q0 = qout_dil(h0, vis, slp, n_dil)
            h1[:] = 0.0001
            iter += 1
            print('Calculation step ' + str(iter))
            for row_i in range(0, img_h):
                for col_j in range(0, img_w):
                    if not np.any(get_grid(h0, row_i, col_j) != 0.0001):
                        pass
                    else:
                        h1[row_i, col_j] = dinf_height(
                            get_grid(fdr_inf, row_i, col_j),
                            get_grid(q0, row_i, col_j),
                            h0[row_i, col_j]
                        )
            if iter - k == i_s:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    u_iter = np.round(q0 / h0, decimals=4)
                    u_iter[u_iter == np.NaN] = -9999
                u.append(u_iter)
                k = iter
            h_1 = np.nan_to_num(h1)
            h_0 = np.nan_to_num(h0)
            dh = h_1 - h_0
            h0 = h1 * 1
        h0[h0 == 0.0001] = -9999
        h0[h0 == np.NaN] = -9999
        h0 = np.round(h0, decimals=4)
        time = str(iter * ts) + ' s'
        tt = str(iter * ts) + ' until stoppage.'
        print(tt)
        return h0, time, u
    elif stop_method == 'i':
        while iter_n >= iter:
            # outer loop defines stop criteria and calculates the height difference between time steps
            if rheo == 'Newtoninan' or 'n' or 'newtonian':
                q0 = qout_newtonian(h0, vis, slp)
            elif rheo == 'Bingham' or 'Bingham plastic' or 'b' or 'bingham plastic' or 'hb' or 'Herschel-Bulkley':
                q0 = qout_hb(h0, vis, slp, m_hb, z_plug)
            elif rheo == 'd' or 'dilatant' or 'Dilatant':
                q0 = qout_dil(h0, vis, slp, n_dil)
            h1[:] = 0.0001
            iter += 1
            print('Calculation step ' + str(iter))
            for row_i in range(0, img_h):
                for col_j in range(0, img_w):
                    if not np.any(get_grid(h0, row_i, col_j) != 0.0001):
                        pass
                    else:
                        h1[row_i, col_j] = dinf_height(
                            get_grid(fdr_inf, row_i, col_j),
                            get_grid(q0, row_i, col_j),
                            h0[row_i, col_j]
                        )
            if iter - k == i_s:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    u_iter = np.round(q0 / h0, decimals=4)
                    u_iter[u_iter == np.NaN] = -9999
                u.append(u_iter)
                k = iter
            h0 = h1 * 1
        h0[h0 == 0.0001] = -9999
        h0[h0 == np.NaN] = -9999
        h0 = np.round(h0, decimals=4)
        time = str(iter * ts) + ' s'
        tt = str(iter * ts) + ' until stoppage.'
        print(tt)
        return h0, time, u


def mfd_method(hin, stop_value, dem, vis, slp, n, i_s):  # calculates flow propagation through deterministic infinity
    h0 = hin
    h1 = np.empty(hin.shape, dtype=float)
    dh = float(1.0)  # arbitrary value to ensure "while" condition is met
    iter = 0
    img_h = hin.shape[0]
    img_w = hin.shape[1]
    fdr_mfd = mfd_slope(dem, pxl)
    fdr_mfd = mfd_fractions(fdr_mfd, n)
    u = []
    k = 0
    print('Multiple flow direction calculated. Starting simulation.')
    if stop_method == 'm':
        while np.nanmax(dh) > stop_value:
            # outer loop defines stop criteria and calculates the height difference between time steps
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if rheo == 'Newtoninan' or rheo == 'n' or rheo == 'newtonian':
                    q0 = qout_newtonian(h0, vis, slp)
                elif rheo == 'Bingham' or rheo == 'Bingham plastic' or rheo == 'b' or rheo == 'bingham plastic' \
                        or rheo == 'hb' or rheo == 'Herschel-Bulkley':
                    q0 = qout_hb(h0, vis, slp, m_hb, z_plug)
                elif rheo == 'd' or rheo == 'dilatant' or rheo == 'Dilatant':
                    q0 = qout_dil(h0, vis, slp, n_dil)
            h1[:] = 0.0001
            iter += 1
            print('Calculation step ' + str(iter))
            for row_i in range(0, img_h):
                for col_j in range(0, img_w):
                    if not np.any(get_grid(h0, row_i, col_j) != 0.0001):
                        pass
                    else:
                        h1[row_i, col_j] = dinf_height(
                            get_grid(fdr_mfd, row_i, col_j),
                            get_grid(q0, row_i, col_j),
                            h0[row_i, col_j]
                        )
            h_1 = np.nan_to_num(h1)
            h_0 = np.nan_to_num(h0)
            dh = h_1 - h_0
            h0 = h1 * 1
        if iter - k == i_s:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                u_iter = np.round(q0 / h0, decimals=4)
                u_iter[u_iter == np.NaN] = -9999
            u.append(u_iter)
            k = iter
        h0[h0 == 0.0001] = -9999
        h0[h0 == np.NaN] = -9999
        h0 = np.round(h0, decimals=4)
        time = str(iter * ts) + ' s'
        tt = str(iter * ts) + ' until stoppage.'
        print(tt)
        return h0, time, u
    elif stop_method == 'i':
        while stop_value > iter:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if rheo == 'Newtoninan' or rheo == 'n' or rheo == 'newtonian':
                    q0 = qout_newtonian(h0, vis, slp)
                elif rheo == 'Bingham' or rheo == 'Bingham plastic' or rheo == 'b' or rheo == 'bingham plastic' \
                        or rheo == 'hb' or rheo == 'Herschel-Bulkley':
                    q0 = qout_hb(h0, vis, slp, m_hb, z_plug)
                elif rheo == 'd' or rheo == 'dilatant' or rheo == 'Dilatant':
                    q0 = qout_dil(h0, vis, slp, n_dil)
            h1[:] = 0.0001
            iter += 1
            print('Calculation step ' + str(iter))
            for row_i in range(0, img_h):
                for col_j in range(0, img_w):
                    if not np.any(get_grid(h0, row_i, col_j) != 0.0001):
                        pass
                    else:
                        h1[row_i, col_j] = dinf_height(
                            get_grid(fdr_mfd, row_i, col_j),
                            get_grid(q0, row_i, col_j),
                            h0[row_i, col_j]
                        )
            h_1 = np.nan_to_num(h1)
            h_0 = np.nan_to_num(h0)
            dh = h_1 - h_0
            h0 = h1 * 1
        if iter - k == i_s:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                u_iter = np.round(q0 / h0, decimals=4)
                u_iter[u_iter == np.NaN] = -9999
            u.append(u_iter)
            k = iter
        h0[h0 == 0.0001] = -9999
        h0[h0 == np.NaN] = -9999
        h0 = np.round(h0, decimals=4)
        time = str(iter * ts) + ' s'
        tt = str(iter * ts) + ' until stoppage.'
        print(tt)
        return h0, time, u


if fdr_ar == 'd8':
    output, total_time, vel = d8_method(hin, fdr, stop_value, vis, slp, i_s)
elif fdr_ar == 'dinf':
    output, total_time, vel = dinf_method(hin, stop_value, fdr, vis, slp, spread, i_s)
elif fdr_ar == 'mfd':
    output, total_time, vel = mfd_method(hin, stop_value, dem, vis, slp, n, i_s)


directory = os.getcwd()
os.makedirs(directory + '/results', exist_ok=True)
nrows = output.shape[0]
ncols = output.shape[1]
driver = gdal.GetDriverByName('GTiff')
output_name = str(fdr_ar) + 'res' + str(pxl) + 'v' + str(vis) + 't' \
              + str(ts) + 'h' + str(stop_value) + 'tt' + str(total_time) + str(rheo) + ".tiff"
dataset = driver.Create(output_name, ncols, nrows, 1, gdal.GDT_Float32)
dataset.GetRasterBand(1).WriteArray(output)
dataset.SetGeoTransform(gt)
dataset.SetProjection(projection)
dataset.GetRasterBand(1).SetNoDataValue(-9999)
dataset.FlushCache()
dataset = None

if i_s > 0:
    for item in range(0, len(vel)):
        vel_name = str(fdr_ar) + str(vis) + 't' + str(ts) + 'h' + str(stop_value) + 'u' + str(
            ts * i_s * (item + 1)) + ".tiff"
        velocity = driver.Create(vel_name, ncols, nrows, 1, gdal.GDT_Float32)
        velocity.GetRasterBand(1).WriteArray(output)
        velocity.SetGeoTransform(gt)
        velocity.SetProjection(projection)
        GetRasterBand(1).SetNoDataValue(-9999)
        velocity.FlushCache()

'''
transform = Affine.from_gdal(*gt)
xres = gt[1]
c0x, c0y = transform.c, transform.f  # upper left
c1x, c1y = transform * (0, nrows)     # lower left
c2x, c2y = transform * (ncols, nrows)  # lower right
c3x, c3y = transform * (ncols, 0)     # upper right
header = np.array([
    ['ncols \t\t\t' + str(ncols)],
    ['nrows \t\t\t' + str(nrows)],
    ['xllcorner \t\t' + str(c1x)],
    ['yllcorner \t\t' + str(c1y)],
    ['cellsize \t\t' + str(xres)],
    ['nodata_value \t' + str(-9999)]
], dtype=object)

# name of file will have utilized viscosity, time step, maximum height difference, total time of debris flow
np.savetxt(directory +
           '/results/v' + str(vis) +
           't' + str(ts) +
           'h' + str(max_dh) +
           'tt' + str(total_time) +
           '.txt',
           output,
           delimiter=' ',
           fmt='%10.4f')
np.savetxt(directory + '/results/header_' + str(vis) + '.txt',
           header,
           delimiter=' ',
           fmt='%s')

if i_s > 0:
    for item in range(0, len(vel)):
        np.savetxt(directory +
                   '/results/v' + str(vis) +
                   'u' + str((item + 1) * i_s * ts) +
                   '.txt',
                   output,
                   delimiter=' ',
                   fmt='%10.4f')
'''

print('Simulation successfully finished')
