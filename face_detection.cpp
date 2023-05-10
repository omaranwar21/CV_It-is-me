#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat face_detection(Mat image){
    CascadeClassifier face_cascade;
    face_cascade.load("C:\\Users\\Anwar\\Desktop\\CV Task 5\\haarcascade_frontalface_alt.xml");

    vector<Rect> faces;
    face_cascade.detectMultiScale(image, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

    for (const auto& face : faces) {
        rectangle(image, face, Scalar(0, 255, 0), 2);
    }

    return image;
} 
int main(int, char**) {
    Mat image = cv::imread("C:\\Users\\Anwar\\Desktop\\test8.jpg");

    image = face_detection(image);

    imshow("Face detection", image);
    waitKey(0);
}
