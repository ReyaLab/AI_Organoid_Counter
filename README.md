# AI_Organoid_Counter
Uses a segmentation NN to count organoids in brightfield images.

Test the web app on our [Huggingface Space!](https://huggingface.co/spaces/ReyaLabColumbia/OrganoidCounter)

Note: The program works the same way as a web app, but the installable program here should run faster, is capable of analyzing entire groups of folders at the click of a button, and you can select any model. Therefore, we recommend installing this github program for heavy use.

Fine-tuned on custom data from the following segmentation NN:
https://huggingface.co/nvidia/mit-b3

https://github.com/NVlabs/SegFormer?tab=readme-ov-file

Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., & Luo, P. (2021). SegFormer: Simple and efficient design for semantic segmentation with transformers. arXiv preprint arXiv:2105.15203. https://arxiv.org/abs/2105.15203

The file ColonyAssaySegformerTern.py was used to download the model and fine-tune it using custom data. FIJI was used to create masks for the training data, and the file Image_Rotator.py was used to rotate, flip, and subdivide the large 1536x2048 images into 512x512 sections. The file Additional_training.py was used to fine-tune the model further. 

Works for any image that fits within 1536x2048 pixels. The images are cut into sections and fed into the segmentation model as tiles. Compatible with .tif, .png, and .jpg images.

For the actual usage of the program, only the model itself and the following three Python files are needed:

Organoid_analyzer_gui.py
Organoid_analyzer_AI.py
Organoid_analyzer_Zstack.py

<img width="572" alt="example" src="https://github.com/user-attachments/assets/739217d5-60bf-459b-a548-64d1ed42c316" />
<img width="572" alt="example2" src="https://github.com/user-attachments/assets/4d8a9ec4-f1f7-4bad-9ebe-16ba7b83b074" />

Updated tool to include organoid circularity on 5/26/26! Also note the model on huggingface was improved. This model is based on the base nvidia-b3 segformer, so it will only work with
the newest SMORE.

# Installation:
Have Python3, and the following libraries: Numpy, Pandas, Tkinter, Opencv, Transformers, PIL, matplotlib. NVIDIA's CUDA is recommended for performance but not strictly necessary for usage. We recommend you use a conda environment to avoid conflicts with other projects.

You can install the python packages with the following code:
pip install numpy pandas opencv-python transformers pillow matplotlib

Download the three python files above and the model itself (place the model.safetensors, config.json, and training_args.bin files in one folder with the model name).

https://huggingface.co/ReyaLabColumbia/Segformer_Organoid_Counter_GP

Place the model subfolder and all the files in the same folder and it should be ready to run. To run in Windows, you can just left click Organoid_analyzer_gui.py. 

To set up a clickable icon in Linux, add the path to the Organoid_analyzer_gui.py file into the launch_gui.sh file (instructions inside) and then put the path to the launch_gui.sh file into the launcher.desktop file (instructions inside). Then put the launcher.desktop file on your /desktop and you can run it by clicking.

In any OS, you can run the program by left clicking inside the folder, open in terminal, and then "python Organoid_analyzer_gui.py".

The following is an example of the output by the program:
![Group_analysis_results3](https://github.com/user-attachments/assets/cd5feeca-e5a1-40db-afbc-6f53d5f71f71)

[Group_analysis_results3.xlsx](https://github.com/user-attachments/files/20004750/Group_analysis_results3.xlsx)

![Group_analysis_results](https://github.com/user-attachments/assets/3efed9ee-5fd2-4b1d-a49f-d6c5fc57bb23)
[Group_analysis_results.xlsx](https://github.com/user-attachments/files/21046062/Group_analysis_results.xlsx)

![Group_analysis_results](https://github.com/user-attachments/assets/6e866944-b373-4edb-89d8-89b76127bff8)
[Group_analysis_results.xlsx](https://github.com/user-attachments/files/21046066/Group_analysis_results.xlsx)

This is what the working UI looks like.
<img width="1447" height="948" alt="Screenshot from 2025-07-17 23-25-35" src="https://github.com/user-attachments/assets/e865662b-b9df-4c27-a349-fe0a876fc4e6" />

# Changelog:
6/24/2026-fixed bug so that do multiple files should really find all tif files
