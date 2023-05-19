#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char const *argv[])
{
    cv::VideoCapture cap(0); // 0 refers to the default camera
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open the camera" << std::endl;
        return -1;
    }

    cv::namedWindow("Face Detection", cv::WINDOW_NORMAL);
    cv::resizeWindow("Face Detection", 640, 480);

    cv::Mat frame;
cv::CascadeClassifier face_cascade;
face_cascade.load("C:\\Users\\Anwar\\Desktop\\CV Task 5_2\\haarcascade_frontalface_alt.xml");

while (true) {
    cap >> frame;
    if (frame.empty()) {
        std::cerr << "Error: Cannot capture the frame" << std::endl;
        break;
    }

    std::vector<cv::Rect> faces;
    face_cascade.detectMultiScale(frame, faces, 1.1, 2, 0|cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));

    for (const auto& face : faces) {
        cv::rectangle(frame, face, cv::Scalar(0, 255, 0), 2);
        // add text on the frame
        cv::putText(frame, "Face", cv::Point(face.x, face.y - 5), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(0, 255, 0), 2);
    }
    cv::imshow("Face Detection", frame);
    cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(frame, frame);
    
    if (cv::waitKey(1) == 'q') { // Press 'q' to exit
        break;
    }
}
    // return 0;
}
