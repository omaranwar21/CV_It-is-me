#include "face_detection.hpp"

vector<Mat> faces_detection(Mat image)
{
    vector<Mat> detectedFaces;
    CascadeClassifier face_cascade;
    // face_cascade.load("C:\\Users\\Anwar\\Desktop\\CV Task 5\\haarcascade_frontalface_alt.xml");
    face_cascade.load("E:\\SBME 6th Term\\Computer Vision\\Projects & Tasks\\CV Final Project\\CV_It-is-me\\Detection\\haarcascade_frontalface_alt.xml");

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



// // testing

// int main(int, char **)
// {
//     // Mat image = cv::imread("C:\\Users\\Anwar\\Desktop\\test8.jpg");
//     // string image_path =  "E:\\SBME 6th Term\\Computer Vision\\Projects & Tasks\\CV Final Project\\Our Images\\Train\\Group\\Team_Group (2).JPG";
//     string image_path =  "E:\\SBME 6th Term\\Computer Vision\\Projects & Tasks\\CV Final Project\\CV_It-is-me\\Detect\\Test\\OmarSaad_ (4).JPG";

//     Mat image = cv::imread(image_path);

//     // Check if the image was loaded successfully
//     if (image.empty())
//     {
//         printf("Could not read the image\n");
//         return 1;
//     }

//     vector<Mat> detectedFaces;
//     detectedFaces = faces_detection(image);

//     namedWindow("Face_detection ", WINDOW_NORMAL);
//     imshow("Face_detection", detectedFaces[0]);
//     // view all images in detectedFaces
//     for (size_t i = 0; i < detectedFaces.size(); i++)
//     {
//         namedWindow("Face " + to_string(i), WINDOW_NORMAL);
//         imshow("Face " + to_string(i), detectedFaces[i]);

//         // print image size
//         cout << "Image " << i << " size: " << detectedFaces[i].size() << endl;

//         // Save the image
//         string save_folder_path = "E:\\SBME 6th Term\\Computer Vision\\Projects & Tasks\\CV Final Project\\CV_It-is-me\\Team_Detected_Faces\\";
//         bool success = imwrite(save_folder_path + to_string(i) + ".jpg" , detectedFaces[i]);

//         // Check if the image was saved successfully
//         if (!success)
//         {
//             printf("Could not save the image\n");
//             return 1;
//         }
//     }
//     waitKey(0);
// }
