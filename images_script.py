import os

# folder_path = "E:\SBME 6th Term\Computer Vision\Projects & Tasks\CV Final Project\Repo\CV_It-is-me\cropped_faces" 
Train_path ="E:\SBME 6th Term\Computer Vision\Projects & Tasks\CV Final Project\CV_It-is-me\Images\Train"
Test_path ="E:\SBME 6th Term\Computer Vision\Projects & Tasks\CV Final Project\CV_It-is-me\Images\Test"
# folder_path ="E:\SBME 6th Term\Computer Vision\Projects & Tasks\CV Final Project\CV_It-is-me\\faces_train"


# list then ame of all images in the folder to the txt file
with open("Train_Images.txt", "w") as file:
    for filename in os.listdir(Train_path):
        file.write(filename + "\n")

with open("Test_Images.txt", "w") as file:
    for filename in os.listdir(Test_path):
        file.write(filename + "\n")