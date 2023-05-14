#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

vector<Mat> faces_detection(Mat image){
    vector<Mat> detectedFaces;
    CascadeClassifier face_cascade;
    face_cascade.load("C:\\Users\\Anwar\\Desktop\\CV Task 5\\haarcascade_frontalface_alt.xml");

    vector<Rect> faces;
    face_cascade.detectMultiScale(image, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

    detectedFaces.push_back(image);
    for (size_t i = 0; i < faces.size(); i++) {
        Mat faceROI = image(faces[i]);
        resize(faceROI, faceROI, Size(100, 100), 0, 0, INTER_LINEAR);
        rectangle(image, faces[i], Scalar(0, 255, 0), 2);
        detectedFaces.push_back(faceROI);
    }
    resize(image, image, Size(300, 300), 0, 0, INTER_LINEAR);

    return detectedFaces;
}

int main(int, char**) {
    Mat image = cv::imread("C:\\Users\\Anwar\\Desktop\\test8.jpg");

    vector<Mat> detectedFaces;
    detectedFaces = faces_detection(image);

    imshow("Face detection", detectedFaces[0]);
    // view all images in detectedFaces
    for (size_t i = 1; i < detectedFaces.size(); i++)
    {   
        namedWindow("Face " + to_string(i), WINDOW_NORMAL);
        imshow("Face " + to_string(i), detectedFaces[i]);
    }
    waitKey(0);
}
