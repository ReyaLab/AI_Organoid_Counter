#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 15:26:47 2025

@author: mchva


This is the Tkinter GUI for Mari's image analysis program

"""

import tkinter as tk
import os
import sys
import pandas as pd

def build_summary_column_descriptions(outside=False):
    temp= pd.DataFrame({
        "Column": [
            "image",
            "Number of organoids",
            "organoid_volume",
            "organoid_area",
            "mean_pixel_value",
            "circularity",
            "necrotic_area",
            "percent_necrotic",
            "outside_necrotic_area"
        ],
        "Meaning": [
            "path to image file analyzed",
            "number of organoids in this image file",
            "total volume (estimated with sphere formula) of organoids in this file",
            "total pixel area of organoids in this file",
            "mean pixel value (brightness, 0 is dark, 255 is light) of organoids in this file",
            "mean organoid circularity in this file, (4*pi*area)/(perimeter^2)",
            "total area in pixels of necrosis within organoids in this file",
            "necrotic area / organoid_area",
            "total area in pixels of necrotic regions outside organoids, in this file"
        ]
    })
    if outside==True:
        return(temp)
    else:
        return(temp.head(8))

def find_image_files(path):
    """
    Recursively returns full paths to all supported image files under 'path',
    excluding generated result files and temporary tile files.
    """
    final = []
    try:
        entries = sorted(os.listdir(path))
    except Exception as e:
        print(f"Could not access {path}: {e}")
        return final

    for entry in entries:
        full = os.path.join(path, entry)
        if os.path.isdir(full):
            final += find_image_files(full)
        elif os.path.isfile(full):
            name = entry.lower()
            if (
                (name.endswith('.tif') or name.endswith('.tiff') or
                 name.endswith('.png') or name.endswith('.jpg') or name.endswith('.jpeg'))
                and ('result' not in name)
                and ('_part' not in name)
            ):
                final.append(full)
    return final

pending_path_update = None

def delayed_on_entry_change(*args):
    global pending_path_update
    if pending_path_update is not None:
        root.after_cancel(pending_path_update)
    pending_path_update = root.after(2000, run_path_update)

def run_path_update():
    global pending_path_update
    pending_path_update = None
    on_entry_change()
    
def find_models(path):
    """
    Recursively returns a list of all huggingface models (at any depth) under 'path' 
    containing at least one .bin and at least one .safetensors file.
    """
    final = []
    try:
        entries = os.listdir(path)
    except Exception as e:
        print(f"Could not access {path}: {e}")
        return final

    bin_found = any(
        os.path.isfile(os.path.join(path, f)) and f.lower().endswith('.bin')
        for f in entries
    )
    safetensors_found = any(
        os.path.isfile(os.path.join(path, f)) and f.lower().endswith('.safetensors')
        for f in entries
    )
    if bin_found and safetensors_found:
        final.append(path)
    
    # Recurse into subdirectories
    for entry in entries:
        subdir = os.path.join(path, entry)
        if os.path.isdir(subdir):
            final += find_models(subdir)
    return final
    
def find_directories(path):
    """
    Recursively returns a list of all directories (at any depth) under 'path' containing at least one .tif or .tiff file.
    """
    final = []
    # List everything in the current directory
    try:
        entries = os.listdir(path)
    except Exception as e:
        print(f"Could not access {path}: {e}")
        return final  # Skip if directory can't be accessed

    # Check if current directory contains any tiff files
    tiff_files = [f for f in entries if os.path.isfile(os.path.join(path, f)) and (f.lower().endswith('.tif') or f.lower().endswith('.tiff'))]
    
    png_files = [f for f in entries if os.path.isfile(os.path.join(path, f)) and (f.lower().endswith('.png')) and not ('result' in f)]
    jpg_files = [f for f in entries if os.path.isfile(os.path.join(path, f)) and (f.lower().endswith('.jpg') or f.lower().endswith('.jpeg'))]
    if len(tiff_files+png_files+jpg_files) > 0:
        final.append(path)
    
    # Recurse into subdirectories
    for entry in entries:
        subdir = os.path.join(path, entry)
        if os.path.isdir(subdir):
            final += find_directories(subdir)
    return final

def read_config(path):
    # Default values as strings (to return everything as string)
    #written by chat
    modelpath = find_models(path)
    if len(modelpath)>0:
        modelpath = modelpath[0]
    else:
        modelpath = path
    defaults = {
        "colony area cutoff in pixels": "1000",
        "circularity cutoff": ".25",
        "path": path,
        "include necrotic areas": "False",
        "default model to use":modelpath,
        "z-stack centroid distance cutoff in pixels": "30",
        "z-stack colony overlap cutoff": ".4"
    }
    
    # Read file and parse values
    try:
        with open(path+'config.txt', 'r') as f:
            for line in f:
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip().lower()
                    value = value.strip()
                    # Find default key that matches (ignore case, match by start of string)
                    for default_key in defaults:
                        if key.startswith(default_key):
                            defaults[default_key] = value
                            break
    except FileNotFoundError:
        # File doesn't exist, write default config
        with open(path+'config.txt', 'w') as f:
            f.write(f"colony area cutoff in pixels: {defaults['colony area cutoff in pixels']}\n")
            f.write(f"circularity cutoff: {defaults['circularity cutoff']}\n")
            f.write(f"path: {defaults['path']}\n")
            f.write(f"default model to use: {defaults['default model to use']}\n")
            f.write(f"include necrotic areas: {defaults['include necrotic areas']}\n")
            f.write(f"z-stack centroid distance cutoff in pixels: {defaults['z-stack centroid distance cutoff in pixels']}\n")
            f.write(f"z-stack colony overlap cutoff: {defaults['z-stack colony overlap cutoff']}\n")
    
    # Return list in the desired order
    return [
        defaults["colony area cutoff in pixels"],
        defaults["circularity cutoff"],
        defaults["path"],
        defaults["default model to use"],
        defaults["include necrotic areas"],
        defaults["z-stack centroid distance cutoff in pixels"],
        defaults["z-stack colony overlap cutoff"]
    ]

class HoverText:
    """Chatgpt wrote all the code for hovertext lmao"""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        if self.tooltip:
            return
        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)  # Remove window decorations
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + 20
        self.tooltip.wm_geometry(f"+{x}+{y}")
        label = tk.Label(self.tooltip, text=self.text, background="white", relief="solid", borderwidth=1, justify="left", anchor="w")
        label.pack()

    def hide_tooltip(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

filename = ""
global file_list 
file_list = [""]
root = tk.Tk()
root.title("Reya Lab Organoid Detector AI")
root.geometry("1400x900")  # Set the window size

file_chosen = tk.StringVar(root) 
# Set the default value of the variable 
file_chosen.set("Select an Option") 

global multiple_groups
multiple_groups = tk.IntVar()
multiple_groups.set(False)

def checkbox_change():
    if (checkbox_var.get()):
        dropdown.config(state="disabled")
        global group_analysis
        group_analysis = tk.IntVar()
        global checkbox_group
        #modifying a global object, root, from a function is bad practice. Oh well!
        checkbox_group= tk.Checkbutton(root, text='Z-stack',variable=group_analysis, onvalue=True, offvalue=False)
        checkbox_group.grid(row=4, column=2, padx=10, pady=5)
        
        global multiple_groups
        multiple_groups = tk.IntVar()
        global checkbox_multiple
        checkbox_multiple= tk.Checkbutton(root, text='Do multiple folders',variable=multiple_groups, onvalue=True, offvalue=False)
        checkbox_multiple.grid(row=4, column=3, padx=10, pady=5)
        
        HoverText(checkbox_group, "Z-stack all tif files in a single folder.")
        HoverText(checkbox_multiple, "Apply selected analysis to every folder in the path.")
    else:
        dropdown.config(state="normal")
        try:
            checkbox_group.grid_forget()
            group_analysis.set(False)
            
            checkbox_multiple.grid_forget()
            multiple_groups.set(False)
        except:
            pass
        
def on_entry_change(*args):
    global file_list
    if (entry_path.get() == "") or (entry_path.get() == "Enter image path"):  # Check if the entry box is filled
        label_hidden.config(text="Enter valid path.")
        file_list = [""]
        try:
            file_chosen.set("")
        except:
            pass
    else:
        try:
            global checkbox_var
            global checkbox_all
            global dropdown

            file_list = os.listdir(entry_path.get())
            file_list = [x for x in file_list if (((x[-4::]==".tif") or (x[-5::]==".tiff") or (x[-4::]==".png") or (x[-4::]==".jpg") or (x[-5::]==".jpeg")) and ('result' not in x))]
            if len(file_list) == 0 and len(find_directories(entry_path.get()))==0:
                label_hidden.config(text="There are no .tif/.png/.jpg files in that folder.")
                try:
                    file_chosen.set("")
                except:
                    pass
                return(0)
            else:
                label_hidden.config(text="")
            if len(file_list) == 0:
                file_list = [""]
            label_drop = tk.Label(root, text="Choose file:")
            label_drop.grid(row=4, column=0, padx=10, pady=5, sticky="w")
            frame = tk.Frame(root)
            dropdown = tk.OptionMenu(frame, file_chosen, *file_list) 
            dropdown.pack(side="left", padx=5)
            checkbox_var = tk.IntVar()
            checkbox_all= tk.Checkbutton(frame, text='Analyze group',variable=checkbox_var, onvalue=True, offvalue=False, command=checkbox_change)
            checkbox_all.pack(side="right", padx=5)
            frame.grid(row=4, column=1, padx=5, pady=5)
            HoverText(checkbox_all, "Analyze multiple files.")
            try:
                checkbox_group.grid_forget()
                group_analysis.set(False)
                
                checkbox_multiple.grid_forget()
                multiple_groups.set(False)
            except:
                pass
        except:
            label_hidden.config(text="Enter valid path.")
            file_list = [""]
            try:
                file_chosen.set("")
            except:
                pass
script_path = os.path.abspath( __file__ )    
script_path = script_path.split('/')    
script_path = script_path[:-1]
script_path = '/'.join(script_path)
script_path = script_path+'/'

config = read_config(script_path)
entry_path_txt = tk.StringVar(value=config[2])
entry_path = tk.Entry(root, width=30, textvariable=entry_path_txt)
entry_path.grid(row=3, column=1, padx=10, pady=5)
entry_path_txt.trace_add("write", delayed_on_entry_change)

label_size_min = tk.Label(root, text="Organoid size minimum cutoff:")
label_size_min.grid(row=0, column=0, padx=10, pady=5, sticky="w")
entry_size_min = tk.Entry(root, width=30)
entry_size_min.grid(row=0, column=1, padx=10, pady=5)
entry_size_min.insert(0, config[0])

label_circularity = tk.Label(root, text="Organoid circularity cutoff:")
label_circularity.grid(row=2, column=0, padx=10, pady=5, sticky="w")
entry_circularity = tk.Entry(root, width=30)
entry_circularity.grid(row=2, column=1, padx=10, pady=5)
entry_circularity.insert(0, config[1])

def checkbool(s):
    s = s.strip().lower()
    if ('true' in s) or ('yes' in s) or ('1' in s):
        return True
    else:
        return False
modelpath = config[3]
do_necrosis = tk.BooleanVar(value=checkbool(config[4]))
checkbox_necrosis= tk.Checkbutton(root, text='Analyze necrotic areas',variable=do_necrosis, onvalue=True, offvalue=False)
checkbox_necrosis.grid(row=0, column=2, padx=10, pady=5, sticky="w")
HoverText(checkbox_necrosis, "Detect necrotic regions inside organoids and report necrotic_area and percent_necrotic.")
find_necrosis_outside = tk.BooleanVar(value=False)
checkbox_necrosis_outside = None

def update_necrosis_checkbox(*args):
    global checkbox_necrosis_outside

    if do_necrosis.get():
        if checkbox_necrosis_outside is None:
            checkbox_necrosis_outside = tk.Checkbutton(
                root,
                text='Find necrosis outside organoids',
                variable=find_necrosis_outside,
                onvalue=True,
                offvalue=False
            )
            checkbox_necrosis_outside.grid(row=1, column=2, padx=10, pady=5, sticky="w")
            HoverText(checkbox_necrosis_outside,"Also detect necrotic regions outside organoids and report them separately as outside_necrotic_area. Currently does not work with z-stack analyses.")
        else:
            checkbox_necrosis_outside.grid(row=1, column=2, padx=10, pady=5, sticky="w")
            HoverText(checkbox_necrosis_outside,"Also detect necrotic regions outside organoids and report them separately as outside_necrotic_area. Currently does not work with z-stack analyses.")
    else:
        find_necrosis_outside.set(False)
        if checkbox_necrosis_outside is not None:
            checkbox_necrosis_outside.grid_forget()

do_necrosis.trace_add("write", update_necrosis_checkbox)
update_necrosis_checkbox()
model_options = find_models(script_path)
label_modelpath = tk.Label(root, text="Select model:")
label_modelpath.grid(row=2, column=2, padx=10, pady=5, sticky="w")
modelpathvar = tk.StringVar()
modelpathvar.set(modelpath)
dropdown_models = tk.OptionMenu(root, modelpathvar, *model_options) 
dropdown_models.grid(row=2, column=3, padx=10, pady=5, sticky="w")
label_hidden = tk.Label(root, text="")
label_hidden.grid(row=5, column=0, padx=10, pady=5, sticky="w")
def run_analysis(filn, path, filelist = [], params = [0,0, False, ""], do_all = False, multiple_folders = False):
    sys.path.append(script_path)
    label_hidden.config(text="Initiated analysis.")
    if (multiple_folders == True) and (group_analysis.get() == True):
        directories = find_directories(path)
    
        if len(directories) < 1:
            label_hidden.config(text="No folders with tif files detected.")
            return 0
    
        label_hidden.config(text='Detected the following folders:\n' + '\n'.join(directories[:20]))
        results = None
    
        for count, folder in enumerate(sorted(directories), start=1):
            if folder[-1] != '/':
                folder = folder + '/'
    
            folder_files = sorted([
                f for f in os.listdir(folder)
                if os.path.isfile(os.path.join(folder, f))
                and (f.lower().endswith('.tif') or f.lower().endswith('.tiff'))
                and ('result' not in f.lower())
                and ('_part' not in f.lower())
            ])
    
            if len(folder_files) < 1:
                continue
    
            try:
                if group_analysis.get():
                    result = run_analysis("", folder, folder_files, params, True, False)
                else:
                    for fname in folder_files:
                        temp = run_analysis(fname, folder, [fname], params, False, False)
                        if results is None:
                            results = temp
                        else:
                            results = pd.concat([results, temp])
                    label_hidden.config(text=f"Finished {count} out of {len(directories)} folders.")
                    root.update()
                    continue
    
                if results is None:
                    results = result
                else:
                    results = pd.concat([results, result])
    
                label_hidden.config(text=f"Finished {count} out of {len(directories)} folders.")
                root.update()
    
            except Exception as e:
                print(f"Error analyzing {folder}: {e}")
    
        Parameters = pd.DataFrame({
            "Minimum organoid size in pixels": [params[0]],
            "Minimum organoid circularity": [params[1]],
            "Model": [params[3]]
        })
    
        with pd.ExcelWriter(path + 'summary.xlsx') as writer:
            results.to_excel(writer, sheet_name="Organoid data", index=False)
            Parameters.to_excel(writer, sheet_name="Parameters", index=False)
            build_summary_column_descriptions(params[4]).to_excel(
                writer,
                sheet_name="Parameters",
                index=False,
                startrow=len(Parameters) + 3
            )
    
        return 0
    elif (multiple_folders == True) and (group_analysis.get() != True):
            all_files = find_image_files(path)
        
            if len(all_files) < 1:
                label_hidden.config(text="No tif files detected.")
                return 0
        
            label_hidden.config(text='Detected the following tif files:\n' + '\n'.join(all_files[:20]))
            results = None
        
            for count, fullpath in enumerate(all_files, start=1):
                folder = os.path.dirname(fullpath)
                if folder[-1] != '/':
                    folder = folder + '/'
                fname = os.path.basename(fullpath)
        
                try:
                    result = run_analysis(fname, folder, [fname], params, True, False)
                    label_hidden.config(text=f"Finished {count} out of {len(all_files)} files.")
                    root.update()
        
                    if results is None:
                        results = result
                    else:
                        results = pd.concat([results, result])
        
                except Exception as e:
                    print(f"Error analyzing {fullpath}: {e}")
        
            Parameters = pd.DataFrame({
                "Minimum organoid size in pixels": [params[0]],
                "Minimum organoid circularity": [params[1]],
                "Model": [params[3]]
            })
        
            with pd.ExcelWriter(path + 'summary.xlsx') as writer:
                results.to_excel(writer, sheet_name="Organoid data", index=False)
                Parameters.to_excel(writer, sheet_name="Parameters", index=False)
                build_summary_column_descriptions(params[4]).to_excel(
                    writer,
                    sheet_name="Parameters",
                    index=False,
                    startrow=len(Parameters) + 3
                )
            return 0
    if do_all == False:
        import Organoid_analyzer_AI as MA
        from PIL import Image, ImageTk
        import cv2
        img,result = MA.main([path, filn, params[0], params[1], script_path, params[2], params[3], params[4]])
        cv_image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        del img
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(cv_image_rgb)
        del cv_image_rgb
        # Resize image (optional)
        pil_image = pil_image.resize((600, 600), Image.LANCZOS)
        
        # Convert PIL image to PhotoImage
        img = ImageTk.PhotoImage(pil_image)
        del pil_image
        # Display in Tkinter label
        labelimg = tk.Label(root, image=img)
        labelimg.photo = img
        labelimg.grid(row=6, column=1)
    elif (do_all == True) and (group_analysis.get() == True):
        import Organoid_analyzer_Zstack as MA
        from PIL import Image, ImageTk
        import cv2
        img,result = MA.main([path, filelist, params[0], params[1], script_path, params[2], int(config[5]), float(config[6]), params[3], params[4]])
        result.insert(loc=0, column="image", value=path)
        cv_image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        del img
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(cv_image_rgb)
        del cv_image_rgb
        # Resize image (optional)
        pil_image = pil_image.resize((600, 600), Image.LANCZOS)
        
        # Convert PIL image to PhotoImage
        img = ImageTk.PhotoImage(pil_image)
        del pil_image
        # Display in Tkinter label
        labelimg = tk.Label(root, image=img)
        labelimg.photo = img
        labelimg.grid(row=6, column=1)
        return(result)
    else:
        import Organoid_analyzer_AI as MA
        from PIL import Image, ImageTk
        import cv2
        results = None
        for x in filelist:
            img,result = MA.main([path, x, params[0], params[1], script_path, params[2], params[3], params[4]])
            result.insert(loc=0, column="image", value=path+x)
            cv_image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            del img
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(cv_image_rgb)
            del cv_image_rgb
            # Resize image (optional)
            pil_image = pil_image.resize((600, 600), Image.LANCZOS)
            
            # Convert PIL image to PhotoImage
            img = ImageTk.PhotoImage(pil_image)
            del pil_image
            # Display in Tkinter label
            labelimg = tk.Label(root, image=img)
            labelimg.photo = img
            labelimg.grid(row=6, column=1)
            root.update()
            if results is None:
            	results = result
            else:
            	results = pd.concat([results, result])
        if multiple_groups.get() == False:
            Parameters = pd.DataFrame({"Minimum organoid size in pixels":[params[0]], "Minimum organoid circularity":[params[1]], 'Model':[params[3]]})
            with pd.ExcelWriter(path+'summary.xlsx') as writer:
                	results.to_excel(writer, sheet_name="Organoid data", index=False)
                	Parameters.to_excel(writer, sheet_name="Parameters", index=False)
        else:
            return(results)           
    label_hidden.config(text="Finished analysis.")
    
def click_conf():
    global file_list
    if (not entry_size_min.get().isnumeric()) or (entry_circularity.get()==''):
        label_hidden.config(text="Please fix input.")
    elif ((file_chosen.get() in file_list) or (checkbox_var.get())) and (len(file_list)>0) and ((file_list != ['']) or multiple_groups.get()) and (int(entry_size_min.get())>=0 or float(entry_circularity.get())>=0):
        file_list = os.listdir(entry_path.get())
        file_list = [x for x in file_list if (((x[-4::]==".tif") or (x[-5::]==".tiff") or (x[-4::]==".png") or (x[-4::]==".jpg") or (x[-5::]==".jpeg")) and ('result' not in x))]
        label_hidden.config(text="")
        filename = file_chosen.get()
        #print(modelpathvar.get())
        run_analysis(filename,entry_path.get(),file_list,[int(entry_size_min.get()),float(entry_circularity.get()),do_necrosis.get(),modelpathvar.get(),find_necrosis_outside.get()],checkbox_var.get(),multiple_groups.get())
    else:
        label_hidden.config(text="Please fix input.")

btn_confirm = tk.Button(root, text = "Confirm parameters and analyze" , fg = "black", command=click_conf)
# Set Button Grid
btn_confirm.grid(row=5, column=1, padx=10, pady=5)
on_entry_change()

HoverText(entry_size_min, "Minimum size in pixels to be considered a valid organoid.")
HoverText(entry_circularity, "Minimum circularity cutoff of organoid, defined as 4*pi*area/perimeter^2. \nCloser to 1 is more circular.")


root.mainloop()
