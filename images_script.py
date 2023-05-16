import os

def write_text_file(path,  output_name) :
    # list then ame of all images in the folder to the txt file
    txt_file =  "image_lists//" + output_name + ".txt"
    with open(txt_file, "w") as file:
        for filename in os.listdir(path):
            file.write(filename + "\n")

# folder_path = "E:\SBME 6th Term\Computer Vision\Projects & Tasks\CV Final Project\Repo\CV_It-is-me\cropped_faces" 
Train_path ="E:\SBME 6th Term\Computer Vision\Projects & Tasks\CV Final Project\CV_It-is-me\Images\Train"
Test_path ="E:\SBME 6th Term\Computer Vision\Projects & Tasks\CV Final Project\CV_It-is-me\Images\Test"
# folder_path ="E:\SBME 6th Term\Computer Vision\Projects & Tasks\CV Final Project\CV_It-is-me\\faces_train"

detect_train_path = "E:\SBME 6th Term\Computer Vision\Projects & Tasks\CV Final Project\CV_It-is-me\Detect\Train"
detect_test_path = "E:\SBME 6th Term\Computer Vision\Projects & Tasks\CV Final Project\CV_It-is-me\Detect\Test"

team_train_path = "E:\SBME 6th Term\Computer Vision\Projects & Tasks\CV Final Project\CV_It-is-me\Images\Team_Data\Train"
team_test_path = "E:\SBME 6th Term\Computer Vision\Projects & Tasks\CV Final Project\CV_It-is-me\Images\Team_Data\Test"


# list then ame of all images in the folder to the txt file
# write_text_file(Train_path,  "Train_Images.txt")

# write_text_file(Test_path,  "Test_Images.txt")

write_text_file(detect_train_path,  "detect_train")

# write_text_file(detect_test_path,  "detect_test")

# write_text_file(team_train_path,  "team_train")

# write_text_file(team_test_path,  "team_test")

