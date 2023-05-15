// #include "face_detection.cpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <numeric>


#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

vector<Mat> faces_detection(Mat image)
{
    vector<Mat> detectedFaces;
    CascadeClassifier face_cascade;
    // face_cascade.load("C:\\Users\\Anwar\\Desktop\\CV Task 5\\haarcascade_frontalface_alt.xml");
    face_cascade.load("E:\\SBME 6th Term\\Computer Vision\\Projects & Tasks\\CV Final Project\\CV_It-is-me\\haarcascade_frontalface_alt.xml");

    vector<Rect> faces;
    face_cascade.detectMultiScale(image, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

    detectedFaces.push_back(image);
    for (size_t i = 0; i < faces.size(); i++)
    {
        Mat faceROI = image(faces[i]).clone();
        // resize(faceROI, faceROI, Size(100, 100), 0, 0, INTER_LINEAR);
        rectangle(image, faces[i], Scalar(0, 255, 0), 2);
        detectedFaces.push_back(faceROI);
    }
    resize(image, image, Size(300, 300), 0, 0, INTER_LINEAR);

    return detectedFaces;
}



// read training list
vector<string> readList(string listFilePath)
{
    vector<string> facesPath;
    ifstream file(listFilePath.c_str(), ifstream::in);

    if (!file)
    {
        cout << "Fail to open file: " << listFilePath << endl;
        exit(0);
    }

    string line, path, id;
    while (getline(file, line))
    {
        stringstream lines(line);
        getline(lines, path);

        path.erase(remove(path.begin(), path.end(), '\r'), path.end());
        path.erase(remove(path.begin(), path.end(), '\n'), path.end());
        // path.erase(remove(path.begin(), path.end(), ' '), path.end());

        facesPath.push_back(path);
    }
    // vector<Mat> images = readImages(facesPath);
    return facesPath;
}

int main(int, char **)
{
    // pathes to folders
    string train_detection_folder_path = "E:\\SBME 6th Term\\Computer Vision\\Projects & Tasks\\CV Final Project\\CV_It-is-me\\Detect\\Train";
    string test_detection_folder_path = "E:\\SBME 6th Term\\Computer Vision\\Projects & Tasks\\CV Final Project\\CV_It-is-me\\Detect\\Test";
    // pathes to script files
    string detect_training_script_file = "E:\\SBME 6th Term\\Computer Vision\\Projects & Tasks\\CV Final Project\\CV_It-is-me\\image_lists\\detect_train.txt";
    string detect_test_script_file = "E:\\SBME 6th Term\\Computer Vision\\Projects & Tasks\\CV Final Project\\CV_It-is-me\\image_lists\\detect_test.txt";
    // pathes to save folder
    string detected_train_folder = "E:\\SBME 6th Term\\Computer Vision\\Projects & Tasks\\CV Final Project\\CV_It-is-me\\Team_Detected_Faces\\Train" ;
    string detected_test_folder = "E:\\SBME 6th Term\\Computer Vision\\Projects & Tasks\\CV Final Project\\CV_It-is-me\\Team_Detected_Faces\\Test" ;

    // read script files
    vector<string> train_files_detect = readList(detect_training_script_file);
    // vector<string> test_files_detect = readList(detect_test_script_file);

    // cout << train_files_detect[1] << endl;


    // loop over files vector and read images and detect faces for training
    for (const auto &filename : train_files_detect)
    {
        string file_path = train_detection_folder_path + "\\" + filename;
        Mat image = imread(file_path);
        // Check if the image was loaded successfully
        if (image.empty())
        {
            printf("Could not read the image\n");
            cout << file_path <<endl;
            return 1;
        }
        vector<Mat> detectedFaces;
        detectedFaces = faces_detection(image);

        // namedWindow("Face_detection ", WINDOW_NORMAL);
        // imshow("Face_detection", detectedFaces[0]);
        // view all images in detectedFaces
        for (size_t i = 1; i < detectedFaces.size(); i++)
        {
            // namedWindow(filename+ to_string(i), WINDOW_NORMAL);
            // imshow( filename + to_string(i), detectedFaces[i]);

            // print image size
            // cout << "Image " << i << " size: " << detectedFaces[i].size() << endl;

            // Save the image
            string save_folder_path = "E:\\SBME 6th Term\\Computer Vision\\Projects & Tasks\\CV Final Project\\CV_It-is-me\\Team_Detected_Faces\\Train\\";
            bool success = imwrite(detected_train_folder+  filename + "-" + to_string(i) + ".jpg", detectedFaces[i]);

            // Check if the image was saved successfully
            if (!success)
            {
                printf("Could not save the image\n");
                return 1;
            }
        }
    }

    
    // // loop over files vector and read images and detect faces for testing
    // for (const auto &filename : test_files_detect)
    // {
    //     string file_path = test_detection_folder_path + "\\" + filename;
    //     Mat image = imread(file_path);
    //     // Check if the image was loaded successfully
    //     if (image.empty())
    //     {
    //         printf("Could not read the image\n");
    //         cout << file_path <<endl;
    //         return 1;
    //     }
    //     vector<Mat> detectedFaces;
    //     detectedFaces = faces_detection(image);

    //     // namedWindow("Face_detection ", WINDOW_NORMAL);
    //     // imshow("Face_detection", detectedFaces[0]);
    //     // view all images in detectedFaces
    //     for (size_t i = 1; i < detectedFaces.size(); i++)
    //     {
    //         // namedWindow(filename+ to_string(i), WINDOW_NORMAL);
    //         // imshow( filename + to_string(i), detectedFaces[i]);

    //         // print image size
    //         // cout << "Image " << i << " size: " << detectedFaces[i].size() << endl;

    //         // Save the image
    //         string save_folder_path = "E:\\SBME 6th Term\\Computer Vision\\Projects & Tasks\\CV Final Project\\CV_It-is-me\\Team_Detected_Faces\\Test\\";
    //         bool success = imwrite(detected_test_folder+  filename + "-" + to_string(i) + ".jpg", detectedFaces[i]);

    //         // Check if the image was saved successfully
    //         if (!success)
    //         {
    //             printf("Could not save the image\n");
    //             return 1;
    //         }
    //     }
    // }



    // waitKey(0);
}

