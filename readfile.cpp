/**************************************************************************************************************************/
#include <iostream>
#include "opencv2/opencv.hpp"
#include <fstream>
/**************************************************************************************************************************/

/**************************************************************************************************************************/
using namespace std;
using namespace cv;
/**************************************************************************************************************************/

/**************************************************************************************************************************/
vector<Mat> readImages(vector<string> trainFacesPath)
{
    string folder_path = "C:\\Users\\Anwar\\Desktop\\CV Task 5\\cropped_faces";
    vector<Mat> images;

    for (const auto& filename : trainFacesPath) {
        string file_path = folder_path + "\\" + filename;
        Mat img = imread(file_path);
        images.push_back(img);
    }

    cout << "Number of images: " << images.size() << endl;

    return images;
}

//read training list
vector<Mat> readList(string listFilePath)
{
    vector<string> facesPath;
    ifstream file(listFilePath.c_str(), ifstream::in);
    
    if (!file) {
        cout << "Fail to open file: " << listFilePath << endl;
        exit(0);
    }
    
    string line, path, id;
    while (getline(file, line)) {
        stringstream lines(line);
        getline(lines, path);
        
        path.erase(remove(path.begin(), path.end(), '\r'), path.end());
        path.erase(remove(path.begin(), path.end(), '\n'), path.end());
        path.erase(remove(path.begin(), path.end(), ' '), path.end());
        
        facesPath.push_back(path);
    }
    vector<Mat> images = readImages(facesPath);
    return images;
}
/**************************************************************************************************************************/


/**************************************************************************************************************************/
int main(int argc, char** argv)
{
    string trainListFilePath = "C:\\Users\\Anwar\\Desktop\\CV Task 5\\images_list.txt";
    vector<Mat> images = readList(trainListFilePath); 
    int counter = 0;
    for (const auto& img : images){
        imshow("Image"+to_string(counter++), img);
    }
    waitKey(0);

    return 0;
}
/**************************************************************************************************************************/