import tkinter as tk
from tkinter import ttk
from PIL import Image
from PIL import ImageTk
import libraries as lib
import numpy as np
import matplotlib.pyplot as plt

def status_update(label):
    label.config(text = 'altceva')

def image_update(label):
    global photo
    new_image = Image.open('/home/alex/Desktop/cercetare/bdecay/program/imgrec/datagrahps/fe47.png')
    photo = ImageTk.PhotoImage(new_image)
    label.config(image = photo)
    print('merge')
    
def print_val(entry):
    val = entry.get()
    print(val)
    return 1
    
def make_fit(entry1, entry2, entry3, path_entry):
    
    default_path = 'imgrec/smooth_data.txt' # default path to data set
    
    data_path = path_entry.get() # path to the data file
    
    if data_path =='':
        data_path = default_path
    
    savepath ='imgrec/ceva.png' 
    data = np.loadtxt(data_path)
    
    x = [item[0] for item in data]
    y = [item[1] for item in data]
    
    peak_val = float(entry2.get())
    interval_bot = float(entry1.get())
    interval_top = float(entry3.get())
    
    lib.new_lorentzfit(x,y, peak_val, interval_bot, interval_top)
    
    return 0

# defines status change message

def change_status():
    pass

def close_plot():
    plt.close()

# function that initializes a plot

def plot_init(path_entry):
    
    default_path = 'imgrec/smooth_data.txt' # default path to data set
    
    data_path = path_entry.get() # reading the path from the entry field
    
    # if no path is given, take the default path to data set
    
    if data_path == '':
        data_path = default_path
        
    
    data = np.loadtxt(data_path)
    
    x = [item[0] for item in data]
    y = [item[1] for item in data]
    
    plt.figure()
    plt.plot(x,y,color = 'black', linewidth = 3)
    return 0

def plot_show():
    plt.title('Lorentzian Data Fitting', fontsize = 20)
    plt.xlabel('E(keV)', fontsize = 20)
    plt.ylabel('Events', fontsize = 20)
    plt.savefig('imgrec/fit.png',format = 'png') 
    plt.tick_params(axis= 'both', labelsize = 20)
    plt.legend()
    plt.show()
    return 0
