import numpy as np 
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os

# ====================== scale ======================= #

# This function takes the 2d list of pixel positions and 
# scales it accordingly, using hand fed- values for points
# on the x and y axis

# this will later be done by reading the reference values
# from a file

def scale(xp, yp, xref, xcoord, xref2, xcoord2, yref, ycoord):
    
    # xref = input("Introduce x value of reference: ")
    # xcoord = input("Position of reference pixel: ")
    # 158,  187,  241  117
    # 3003, 3494, 4254 2358 
    
    slope = (xcoord2 - xcoord) / (xref2 - xref)
    
    b = xcoord - slope * xref

    
    for i in range(len(xp)):
        xp[i] = xp[i] * slope + b
    
    # for i in range(len(yp)):
    #     yp[i] = yp[i] * ycoord / yref
    # yref = input("y value of reference: ")
    # ycoord = input('position of pixel: ')
    
    for i in range(len(yp)):
        yp[i] = yp[i] * yref / ycoord

# ==================================================== #

# ===================== gaussfit ===================== #

def gaussfit(xp,yp):
    def gaussian(x, A, mu, sigma):
        return A * np.exp(-(x - mu)**2 / (2 * sigma**2))
    xp = np.array(xp)
    yp = np.array(yp)
    # defining a boolean mask for selecting a certain interval 
    # of data that we will fit using the gaussian function
    
    
    peak_xval = int(input('input the peak energy \n')) #3003
    parameter = int(input('input the parameter \n'))
    interval_start = peak_xval - parameter
    interval_end = peak_xval + parameter
       
    interval_mask = (xp >= interval_start) & (xp <= interval_end)
    x_interval = xp[interval_mask]
    y_interval = yp[interval_mask]

    # fitting the data in the interval with the gaussian function
    popt, pcov = curve_fit(gaussian, x_interval, y_interval, p0=[1, np.mean(x_interval), 1])
    
    print("Optimized parameters (A, mu, sigma):", popt)
    
    # plotting the result
    plt.figure()
    plt.scatter(xp, yp, color='black', marker='.', s=2)
    plt.plot(x_interval, gaussian(x_interval, *popt), color='red', label='Fitted Gaussian (Interval)',linewidth = 2)
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    A_opt, mu_opt, sigma_opt = popt
    plt.text(0.02, 0.98, f'A = {A_opt:.2f}\nμ = {mu_opt:.2f}\nσ = {sigma_opt:.2f}', fontsize=12, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='left')
    plt.title('Fitting Data within Interval with Gaussian Distribution')
    plt.savefig('/home/alex/Desktop/Cercetare/BetaDecay_Project/program/imgrec/gaussianfit.eps')
    plt.show()
    
# ==================================================== #



# ==================== lorentzfit ==================== #

# DESCRIPTION : fits a data interval with a cauchy-laurentz distribution
#               and returns the resulting plot and distribution parameters

def lorentzfit(xp: list, yp: list, peak_xval:float, interval_inflim: int, 
               interval_suplim: int, savepath: str):
    
    def lorentzian(x, x0, gamma, A):
        return A * gamma**2 / ((x - x0)**2 + gamma**2)

    xp = np.array(xp)
    yp = np.array(yp)
    # defining a boolean mask for selecting a certain interval 
    # of data that we will fit using the gaussian function
       
    interval_mask = (xp >= interval_inflim) & (xp <= interval_suplim)
    x_interval = xp[interval_mask]
    y_interval = yp[interval_mask]
    
    initial_guess = [peak_xval, 1, 1]  # Initial guess for parameters: x0, gamma, A
    params, covariance = curve_fit(lorentzian, x_interval, y_interval, p0=initial_guess)

    x0_opt,gamma, A = params
    
    # plt.scatter(xp, yp, color='black', marker='.', s=2)
    # plt.plot(x_interval, lorentzian(x_interval, *params), color='red', label='Fit',linewidth = 2)
    plt.plot(xp, yp, color = 'black')
    plt.plot(xp,lorentzian(xp,*params),color = 'green',linewidth = 2, linestyle = 'dashed',label = 'lorentz fit')
    plt.legend()
    plt.xlabel('E(keV)')
    plt.ylabel('N')
    plt.text(0.02, 0.98, f'x0 = {x0_opt:.2f}\n gamma = {gamma:.2f}', fontsize=12, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='left')
    plt.title('Fitting data to Lorentzian distribution')
    plt.savefig( savepath, format = 'png')
    plt.show()


# ==================================================== #



# ==================== input file ==================== #

def input(path):
    data = []
    with open(path, 'r') as file:
        
        for line in file:
            
            line = line.strip()
            
            if line and not line.startswith('#'):
                data.append(line)
    
    return data          

# ==================================================== #   

# DESCRIPTION : the subroutine reduces the number of points of 
#               the resulting digitized graph and prepares it 
#               for analysis

def smooth(data: list[float, float]):
    
    # declaring the vector that stores the resulting graph
    smooth_data = []
    mean = 0
    n = 0
    
    for i in range(len(data) - 1):
        n += 1
        mean += data[i,1]
        
        if data[i + 1, 0] != data[i, 0]:
            mean = mean / n
            smooth_data.append([data[i,0],mean])
            mean = 0
            n = 0
    return smooth_data

# DESCRIPTION : takes the amplitudes spectrum and filters it with
#               band filter

def filter(x: float, val_low: float, val_high: float, k: float ):
    def f_down(val_high):
        f_down = 1 / (1 + np.exp(2 * k * (x - val_high)))
        return f_down
    
    def f_up(val_low):
        f_up = 1 / (1 + np.exp(-2 * k * (x - val_low)))
        return f_up
    
    result = (f_down(val_high) + f_up(val_low) - f_up(-val_low) - f_down(-val_high))
    
    return result

# DESCRIPTION : creates a filter function that acts as a band
#               pass filter on an amplitude spectrum
# INPUT       : freq     = vector of frequencies
#             : amp      = vector of amplitudes
#             : val_low  = inferior cutoff point
#             : va-_high = superior cutoff point 

def band_filter(freq: list[float], amp: list[float], val_low: float, val_high: float, k: float):
    
    new_amp =  amp * filter(freq, val_low, val_high, k)
    y = filter(freq, val_low, val_high, k)
    
    plt.figure()
    plt.title('Filtered Function vs frequency')
    plt.scatter(freq, y, marker='.')
    plt.savefig(f'graphs/filter_function_{val_low}',format = 'png')
    
    plt.figure()
    plt.title('Filtered Amplitude Spectre')
    plt.plot(freq, new_amp, label = f'val_low = {val_low}')  
    plt.legend()  
    plt.savefig(f'graphs/filtered_amp_spec_{val_low}',format = 'png')
    return new_amp

def delete_contents_of_folder(folder_path):
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)  # Delete the file
            elif os.path.isdir(file_path):
                os.rmdir(file_path)   # Delete the directory
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

def sim_up_filter_function(x: float, val: float, k: float):
    
    def f_down(val):
        f_down = 1 / (1 + np.exp(2 * k * (x - val)))
        return f_down
    
    def f_up(val):
        f_up = 1 / (1 + np.exp(-2 * k * (x + val)))
        return f_up
    
    result = f_down(val) + f_up(val) - 1
    return result

def up_filter(freq: list[float], amp: list[float], val: float, k:float,out):
    
    y = sim_up_filter_function(freq, val, k)
    
    new_amp = amp * sim_up_filter_function(freq, val, k) 
    
    plt.figure()
    plt.scatter(freq, y)
    plt.title('Filter Function')
    plt.savefig(os.path.join(out,'filter_function.png'), format = 'png')
    
    plt.figure()
    plt.plot(freq, new_amp)
    plt.title('Filtered Amplitude Spectre')
    plt.savefig(os.path.join(out,'filtered_amp_spec.png'), format = 'png')
    plt.close()
    return new_amp

# this function finds the local maxima of a given given plot

def find_maxima(data_path: str = None, data: list[tuple[float, float]] = None):
    if data is None and data_path is None:
        print('Error: Either data or data_path should be provided.')
        return None
    
    if data is None:
        data = np.loadtxt(data_path)
    else:
        data = np.array(data)
    
    sign_change = np.diff(np.sign(np.diff(data[:,1]))) # Sign change of the difference between y coordinates
    maxima_indices = np.where(sign_change == -2)[0] + 1 # Gives the x positions of the maxima 
    local_maxima = data[maxima_indices]
    
    return local_maxima

# this function finds the local minima of a given plot 

def find_minima(data_path):
    
    data = np.loadtxt(data_path)
    
    sign_change = np.diff(np.sign(np.diff(data[:,1]))) # sign change of the difference between y coordinates
    minima_indices = np.where(sign_change == 2)[0] + 1
    local_minima = data[minima_indices]
    
    return local_minima

# new lorentzian fit function that accomodates the need of more lorentzians
# on the same plot

def new_lorentzfit(xp: list, yp: list, peak_xval:float, interval_inflim: int, 
                   interval_suplim: int):
    
    def lorentzian(x, x0, gamma, A):
        return A * gamma**2 / ((x - x0)**2 + gamma**2)

    xp = np.array(xp)
    yp = np.array(yp)
    
    # defining a boolean mask for selecting a certain interval 
    # of data that we will fit using the gaussian function
       
    interval_mask = (xp >= interval_inflim) & (xp <= interval_suplim)
    x_interval = xp[interval_mask]
    y_interval = yp[interval_mask]
    
    initial_guess = [peak_xval, 1, 1]  # Initial guess for parameters: x0, gamma, A
    params, covariance = curve_fit(lorentzian, x_interval, y_interval, p0=initial_guess)

    x0_opt,gamma, A = params
    
    if gamma < 0:
        print("something went wrong")
        return 1

    plt.plot(xp,lorentzian(xp,*params),linewidth = 2, linestyle = 'dashed',label = f'gamma = {gamma:.2f}')
    
    return 0

# defines a new scale function that uses a dynamic number of points to 
# obtain the fit function

def new_scalefunction(data_path_ox: str, data_path_oy: str, xpoints: list[float], ypoints: list[float]):
    
    data_ox = np.loadtxt(data_path_ox)
    data_oy = np.loadtxt(data_path_oy)
    
    # finding the parameters of the 1d polynomial for the ox axis
    parameters_ox = np.polyfit(data_ox[:,0], data_ox[:,1], 1)
    
    # scaling the other values ox values
    for i in range(len(xpoints)):
        xpoints[i] = xpoints[i] * parameters_ox[0] + parameters_ox[1]
    
    # finding the scaling factor for oy axis
    parameter_oy = data_oy[1] / data_oy[0]
    
    # scaling the other oy values
    for i in range(len(ypoints)):
        ypoints[i] = ypoints[i] * parameter_oy
    
    return (xpoints, ypoints)