import tkinter as tk
from tkinter import messagebox
import re
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
import os
from PIL import Image, ImageTk
from libraries import libraries as lib


def file_select(button : tk.Button):
    global input_file
    input_file =  filedialog.askopenfilename(initialdir = "/home/alex/Desktop/",title = "Select file",filetypes = (("csv files","*.csv"),("all files","*.*")))
    button.config(bg = 'green')
    return  0

def run(out, root: tk.Tk, progress: ttk.Progressbar,entry:tk.Entry):
    try:
        global input_file
            
        # Create the directory if it doesn't exist
        if not os.path.exists(out):
            print(f"Creating directory: {out}")
            os.makedirs(out)
        else:
            print(f"Directory already exists: {out}")

        # Ensure the directory is created
        if os.path.exists(out):
            print(f"Directory successfully created: {out}")
        else:
            print(f"Failed to create directory: {out}")

        # Deleting the contents of the graphs folder
        lib.delete_contents_of_folder(out)
        
        root.update_idletasks()
        progress['value'] = 10
            
        file_path = input_file
        
        delimiters = [',', ';', ':', '|',' ']

        # Read the file content
        with open(file_path, 'r') as file:
            file_content = file.read()

        # Create a regex pattern to match any of the delimiters
        pattern = '|'.join(map(re.escape, delimiters))

        # Replace all delimiters with a single space (or any other delimiter of your choice)
        uniform_content = re.sub(pattern, ' ', file_content)

        # Save the uniform content to a temporary file or use StringIO to avoid file I/O
        from io import StringIO
        temp_file = StringIO(uniform_content)
        data = np.loadtxt(temp_file, delimiter=' ')
        
        dt = 0.001
        
        data = lib.smooth(data) # transforming it into a proper function

        root.update_idletasks()
        progress['value'] = 20
        
        x = [item[0] for item in data]
        y = [item[1] for item in data]

        plt.figure()
        plt.plot(x ,y)
        plt.xlabel('E(keV)', fontsize = 12)
        plt.ylabel('Events', fontsize = 12)
        plt.savefig(os.path.join(out,'raw_data.png'),format = 'png')

        n = len(data)
        fourier_coeff = np.fft.fft(y, n)
        psd = abs(fourier_coeff) / n
        freq = (1 / (dt * n)) * np.arange(n)

        root.update_idletasks()
        progress['value'] = 30
        
        plt.figure()
        plt.title('power series density vs frequency')
        plt.plot(freq,psd)

        amp_spec = np.abs(fourier_coeff)
        freq_bins = np.fft.fftfreq(len(x), len(y)/2)

        # filtering the noise
        # filter = 0.25
        # indices = psd > filter
        # psd_filtered = psd * indices
        # fourier_coeff = indices * fourier_coeff

        root.update_idletasks()
        progress['value'] = 40
        
        plt.figure(3)
        plt.title('amplitudes spectre')
        plt.plot(freq_bins, amp_spec)
        plt.savefig(os.path.join(out,'unfiltered_amp_spec.png'),format = 'png')

        # trying to filter the amplitude spectrum in order to
        # see smaller amplitudes
        
        try:
            val = float(entry.get())
        except Exception as e:
            val = 0.0001
        print(val)
        
        k = 10000
        new_amp = lib.up_filter(freq_bins, amp_spec, val, k, out)
        
        root.update_idletasks()
        progress['value'] = 50
        
        filtered_fft = new_amp * np.exp(1j * np.angle(fourier_coeff))

        # performing the inverse transform

        invfour = np.fft.ifft(filtered_fft)
        
        progress['value'] = 100
        root.update_idletasks()
        
        plt.figure()
        plt.title('Filtered Inverse Fourier Transform')
        plt.plot(x, invfour)
        plt.savefig(os.path.join(out,'filtered_inverse.png'),format = 'png')

        smooth_data = np.column_stack((x,np.real(invfour)))

        np.savetxt(os.path.join(out,'smooth_data.csv'),smooth_data, delimiter= '\t')
        
        progress['value'] = 0
        root.update_idletasks()
    except ValueError as er:
        progress['value'] = 0
        root.update_idletasks()
        messagebox.showerror(f"ValueError","Please make sure that the file contains only float values.")
        
    except Exception as e:
        progress['value'] = 0
        root.update_idletasks()
        messagebox.showerror("Invalid Input", "Please select a file first!")

        
def show(label : tk.Label, out):
    image_path = os.path.join(out,'filtered_inverse.png')
    filtered_graph = tk.PhotoImage(file= image_path)
    label.config(image= filtered_graph)
    label.image = filtered_graph
    