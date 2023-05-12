import os

# folder_path = "E:\SBME 6th Term\Computer Vision\Projects & Tasks\CV Final Project\Repo\CV_It-is-me\cropped_faces" 
# folder_path ="E:\SBME 6th Term\Computer Vision\Projects & Tasks\CV Final Project\CV_It-is-me\Images\Train"
folder_path ="E:\SBME 6th Term\Computer Vision\Projects & Tasks\CV Final Project\CV_It-is-me\\faces_train"

# list then ame of all images in the folder to the txt file
with open("faces.txt", "w") as file:
    for filename in os.listdir(folder_path):
        file.write(filename + "\n")