#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 14:23:27 2025

@author: mattc
"""

from PIL import Image
import numpy as np
import os
import cv2

def pad(img_np, tw=2048, th=1536):
    """
    Pads a numpy image (grayscale or RGB) to 2048x1536 (width x height) with white pixels.
    Pads at the bottom and right as needed.
    """
    height, width = img_np.shape[:2]
    pad_bottom = max(0, th - height)
    pad_right = max(0, tw - width)
    # Padding: (top, bottom, left, right)
    if img_np.ndim == 3:
        # Color image (H, W, 3)
        border_value = [255, 255, 255]
    else:
        # Grayscale image (H, W)
        border_value = 255

    padded = cv2.copyMakeBorder(
        img_np, 
        top=0, bottom=pad_bottom,
        left=0, right=pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=border_value
    )
    return padded
    
def cut_img(path, x, patch_size=512):
    img_map = {}
    img = Image.fromarray(pad(cv2.imread(path + x)))
    name = x.split(".")[0]
    width, height = img.size
    i_num = height // patch_size
    j_num = width // patch_size
    count = 1
    for i in range(int(i_num)):
        for j in range(int(j_num)):
            img2 = img.crop((
                patch_size * j,
                patch_size * i,
                patch_size * (j + 1),
                patch_size * (i + 1)
            ))
            cv2.imwrite(path+name+'_part'+str(count)+'.tif', np.array(img2))
            img_map[count] = path+name+'_part'+str(count)+'.tif'
            count +=1
    return img_map, i_num, j_num
    


def stitch(img_map, i_num, j_num, min_width=2048, min_height=1536):
    tiles = []
    count = 1
    for x in img_map:
        temp = img_map[x]
        img_map[x] = cv2.imread(temp)
        if (img_map[x] is None):
            img_map[x] = cv2.imread(temp, cv2.IMREAD_UNCHANGED)
        os.remove(temp)
    for i in range(i_num):
        row_tiles = []
        for j in range(j_num):
            tile = np.array(img_map[count])
            row_tiles.append(tile)
            count += 1
        row_img = np.hstack(row_tiles)
        tiles.append(row_img)
    stitched = np.vstack(tiles)

    # Pad the stitched image if it's less than min_width/min_height
    h, w = stitched.shape[:2]
    pad_h = max(0, min_height - h)
    pad_w = max(0, min_width - w)
    if pad_h > 0 or pad_w > 0:
        # Pad as (top, bottom), (left, right), (channels)
        if stitched.ndim == 3:
            stitched = np.pad(stitched, ((0, pad_h), (0, pad_w), (0, 0)), 'constant')
        else:
            stitched = np.pad(stitched, ((0, pad_h), (0, pad_w)), 'constant')
    return stitched



import matplotlib.pyplot as plt

def visualize_segmentation(mask, image=0):
    plt.figure(figsize=(10, 5))

    if(not np.isscalar(image)):
        # Show original image if it is entered
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis("off")

    # Show segmentation mask
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="gray")  # Show as grayscale
    plt.title("Segmentation Mask")
    plt.axis("off")

    plt.show()



# Load image processor
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
image_processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b3")

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")  # Open and convert to RGB
    inputs = image_processor(image, return_tensors="pt")  # Preprocess for model
    return image, inputs["pixel_values"]

def postprocess_mask(logits):
    mask = torch.argmax(logits, dim=1)  # Take argmax across the class dimension
    return mask.squeeze().cpu().numpy()  # Convert to NumPy array


def eval_img(image_path, model):
    # Load and preprocess image
    image, pixel_values = preprocess_image(image_path)
    pixel_values = pixel_values.to(device)
    with torch.no_grad():  # No gradient calculation for inference
        outputs = model(pixel_values=pixel_values)  # Run model
        logits = outputs.logits
    # Convert logits to segmentation mask
    segmentation_mask = postprocess_mask(logits)
    #visualize_segmentation(segmentation_mask,image)
    segmentation_mask = cv2.resize(segmentation_mask, (512, 512), interpolation=cv2.INTER_LINEAR_EXACT)
    return(segmentation_mask)


# for x in img_map:
#     mask = eval_img(img_map[x])
#     cv2.imwrite(img_map[x], mask)
# del mask,x
# p = stitch(img_map)
# visualize_segmentation(p)

# num_colony = np.count_nonzero(p == 1)  # Counts number of 1s
# num_necrosis = np.count_nonzero(p == 2)

# num_necrosis/num_colony

def find_colonies(mask, size_cutoff, circ_cutoff):
    binary_mask = np.where(mask == 1, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contoursf = []
    for x in contours:
        area = cv2.contourArea(x)
        if (area < size_cutoff):
            continue
        perimeter = cv2.arcLength(x, True)

        # Avoid division by zero
        if perimeter == 0:
            continue
        
        # Calculate circularity
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        if circularity >= circ_cutoff:
            contoursf.append(x)
    return(contoursf)

def find_necrosis(mask):
    binary_mask = np.where(mask == 2, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return(contours)

# contour_image = np.zeros_like(p)
# contours =  find_necrosis(p)
# cv2.drawContours(contour_image, contours, -1, (255), 2)
# visualize_segmentation(contour_image)
import pandas as pd
def compute_centroid(contour):
    M = cv2.moments(contour)
    if M["m00"] == 0:  # Avoid division by zero
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


def contours_overlap_using_mask(contour1, contour2, image_shape=(1536, 2048)):
    """Check if two contours overlap using a bitwise AND mask."""
    import numpy as np
    import cv2
    mask1 = np.zeros(image_shape, dtype=np.uint8)
    mask2 = np.zeros(image_shape, dtype=np.uint8)


    # Draw each contour as a white shape on its respective mask
    cv2.drawContours(mask1, [contour1], -1, 255, thickness=cv2.FILLED)
    cv2.drawContours(mask2, [contour2], -1, 255, thickness=cv2.FILLED)


    # Compute bitwise AND to find overlapping regions
    overlap = cv2.bitwise_and(mask1, mask2)
    
    return np.any(overlap)

def analyze_colonies(mask, size_cutoff, circ_cutoff, img):
    colonies = find_colonies(mask, size_cutoff, circ_cutoff)
    necrosis = find_necrosis(mask)
    
    data = []
    
    for colony in colonies:
        colony_area = cv2.contourArea(colony)
        centroid = compute_centroid(colony)
        if colony_area <= 50:
            continue
        mask = np.zeros(img.shape, np.uint8)
        cv2.drawContours(mask, [colony], -1, 255, cv2.FILLED)
        pix = img[mask == 255]
        # Check if any necrosis contour is inside the colony
        necrosis_area = 0
        nec_list =[]
        for nec in necrosis:
            # Check if the first point of the necrosis contour is inside the colony
            if contours_overlap_using_mask(colony, nec):
                nec_area = cv2.contourArea(nec)
                necrosis_area += nec_area
                nec_list.append(nec)

        data.append({
            "organoid_area": colony_area,
            "necrotic_area": necrosis_area,
            "centroid": centroid,
            "percent_necrotic": necrosis_area/colony_area,
            "contour": colony,
            "nec_contours": nec_list,
            'mean_pixel_value':np.mean(pix)
        })

    # Convert results to a DataFrame
    df = pd.DataFrame(data)
    df.index = range(1,len(df.index)+1)
    return(df)

import torch    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
def main(args):
    path = args[0]
    file = args[1]
    min_size = args[2]
    min_circ = args[3]
    do_necrosis = args[5]
    print('passed do necrosis: ' + str(do_necrosis))
    colonies = {}
    img_map, i_num, j_num = cut_img(path, file)
    from transformers import SegformerForSemanticSegmentation
    # Load fine-tuned model
    model = SegformerForSemanticSegmentation.from_pretrained(args[6])  # Adjust path
    model.to(device)
    model.eval()  # Set to evaluation mode
    for z in img_map:
        mask = eval_img(img_map[z], model)
        cv2.imwrite(img_map[z], mask)
    del mask,z
    p = stitch(img_map, i_num, j_num)
    colonies = analyze_colonies(p, min_size, min_circ, cv2.imread(path + file))
    if len(colonies) <=0:
    	caption = np.ones((150, 2048, 3), dtype=np.uint8) * 255  # Multiply by 255 to make it white
    	cv2.putText(caption, 'No organoids detected.', (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    	img = pad(cv2.imread(path + file))
    	file = file.split('.')[0]
    	cv2.imwrite(path+file+'_result.png', np.vstack((img, caption)))
    	if do_necrosis:
    	   colonies = {'Number of organoids':[0],'organoid_volume':[0],"organoid_area":[0],'mean_pixel_value':[None], "necrotic_area":[0],"percent_necrotic":[0]}
    	   colonies = pd.DataFrame(colonies)
    	   return(np.vstack((img, caption)), colonies)
    	else:
    	   colonies = {'Number of organoids':[0],'organoid_volume':[0],"organoid_area":[0],'mean_pixel_value':[None]}
    	   colonies = pd.DataFrame(colonies)
    	   return(np.vstack((img, caption)), colonies)
    
    img = pad(cv2.imread(path + file))
    img = cv2.copyMakeBorder(img,top=0, bottom=10,left=0,right=10, borderType=cv2.BORDER_CONSTANT,  value=[255, 255, 255]) 
    
    
    colonies = colonies.sort_values(by=["organoid_area"], ascending=False)
    colonies = colonies[colonies["organoid_area"]>= min_size]
    colonies.index = range(1,len(colonies.index)+1) 
    
    for i in range(len(colonies)): 
        cv2.drawContours(img, [list(colonies["contour"])[i]], -1, (0, 255, 0), 2)
        if do_necrosis == True:
            cv2.drawContours(img, list(colonies['nec_contours'])[i], -1, (0, 0, 255), 2)
        coords = list(list(colonies["centroid"])[i])
        if coords[0] > 1950:
            #if a colony is too close to the right edge, makes the label move to left
            coords[0] = 1950
        cv2.putText(img, str(colonies.index[i]), tuple(coords), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
    img = cv2.copyMakeBorder(img,top=10, bottom=0,left=10,right=0, borderType=cv2.BORDER_CONSTANT,  value=[255, 255, 255]) 
    colonies["circularity"]=[4*np.pi/((cv2.arcLength(cnt, True))**2) for cnt in list(colonies['contour'])]
    colonies["circularity"]=colonies["circularity"]*colonies["organoid_area"]
    colonies = colonies.drop('contour', axis=1)
    colonies = colonies.drop('nec_contours', axis=1)
    
    colonies.insert(loc=0, column="organoid_number", value=[str(x) for x in range(1, len(colonies)+1)])
    total_area_dark = sum(colonies['necrotic_area'])
    total_area_light = sum(colonies['organoid_area'])
    ratio = total_area_dark/(abs(total_area_light)+1)
    radii = [np.sqrt(x/3.1415) for x in list(colonies['organoid_area'])]
    volumes = [4.189*(x**3) for x in radii]
    colonies['organoid_volume'] = volumes
    del radii, volumes
    meanpix = sum(colonies['mean_pixel_value'] * colonies['organoid_area'])/total_area_light
    #colonies.loc[len(colonies)+1] = ["Total", total_area_light, total_area_dark, None, ratio,meanpix, sum(colonies['organoid_volume'])]
    colonies.loc[len(colonies)+1] = {
    "organoid_number": "Total",
    "organoid_volume": sum(colonies["organoid_volume"]),
    "organoid_area": total_area_light,
    "mean_pixel_value": meanpix,
    "centroid": None,
    "necrotic_area": total_area_dark,
    "percent_necrotic": ratio,
    "circularity": colonies["circularity"].mean()
    }
    del meanpix
    colonies = colonies[["organoid_number", 'organoid_volume', "organoid_area",'mean_pixel_value', "centroid", "circularity", "necrotic_area","percent_necrotic"]]
    if do_necrosis == False:
        colonies = colonies.drop('necrotic_area', axis=1)
        colonies = colonies.drop('percent_necrotic', axis=1)
    Parameters = pd.DataFrame({"Minimum organoid size in pixels":[min_size], "Minimum organoid circularity":[min_circ]})
    file = file.split('.')[0]
    with pd.ExcelWriter(path+file+'.xlsx') as writer:
        colonies.to_excel(writer, sheet_name="Organoid data", index=False)
        Parameters.to_excel(writer, sheet_name="Parameters", index=False)
    caption = np.ones((150, 2068, 3), dtype=np.uint8) * 255  # Multiply by 255 to make it white
    if do_necrosis == True:
        cv2.putText(caption, "Total area necrotic: "+str(total_area_dark)+ ", Total area living: "+str(total_area_light)+", Ratio: "+str(ratio), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    else:
        cv2.putText(caption, "Total area: "+str(total_area_light), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    cv2.putText(caption, "Total number of organoids: "+str(len(colonies)-1), (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)


    cv2.imwrite(path+file+'_result.png', np.vstack((img, caption)))
    colonies = colonies.drop('organoid_number', axis=1)
    colonies = colonies.drop('centroid', axis=1)
    colonies.insert(0, 'Number of organoids', [len(colonies)-1]*len(colonies))
    return(np.vstack((img, caption)),colonies.iloc[-1:])

