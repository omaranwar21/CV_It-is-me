#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int, char**) {
    Mat image = cv::imread("C:\\Users\\Anwar\\Desktop\\test8.jpg");
    // convert to gray
    // cvtColor(image, image, COLOR_BGR2GRAY);
    CascadeClassifier face_cascade;
    face_cascade.load("C:\\Users\\Anwar\\Desktop\\CV Task 5\\haarcascade_frontalface_alt.xml");

    vector<Rect> faces;
    face_cascade.detectMultiScale(image, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

    for (const auto& face : faces) {
        rectangle(image, face, Scalar(0, 255, 0), 2);
    }

    imshow("Face detection", image);
    waitKey(0);
}
