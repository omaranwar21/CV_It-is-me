#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

vector<Mat> readImages(void)
{

    string folder_path = "C:\\Users\\Anwar\\Desktop\\lfw_funneled\\Abdoulaye_Wade";
    vector<string> filenames = {"Abdoulaye_Wade_0001.jpg", "Abdoulaye_Wade_0002.jpg", "Abdoulaye_Wade_0003.jpg", "Abdoulaye_Wade_0004.jpg"};
    vector<Mat> images;
    // int counter = 0;
    for (const auto& filename : filenames) {
        string file_path = folder_path + "\\" + filename;
        Mat img = imread(file_path);
        // if (img.empty()) {
        //     cerr << "Failed to read image: " << file_path << endl;
        //     return;
        // }
        images.push_back(img);
        // imshow("Image"+to_string(counter++), img);
    }

    cout << "Number of images: " << images.size() << endl;

    return images;
}

int main(int argc, char const *argv[])
{
    vector<Mat> images = readImages();

    // display images
    int counter = 0;
    for (const auto& img : images) {
        imshow("Image"+to_string(counter++), img);
    }
    waitKey(0);
    return 0;
}
