#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

vector<Mat> readImages(vector<string> trainFacesPath)
{
    string folder_path = "C:\\Users\\Anwar\\Desktop\\CV Task 5\\cropped_faces";
    vector<Mat> images;

    for (const auto& filename : trainFacesPath) {
        string file_path = folder_path + "\\" + filename;
        Mat img = imread(file_path);
        resize(img, img, Size(100, 100));
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

Mat PCA_Matrix(vector<Mat> images){
    Mat data, mean, eigenvalues, eigenvectors;
    int n = (int)images.size();
    int d = images[0].rows * images[0].cols;
    data = Mat(d, n, images[0].type());
    int counter = 0;

    for (int i = 0; i < n; i++){
        Mat image = images[i].clone().reshape(0, d);
        image.convertTo(image, images[0].type());
        image.copyTo(data.col(i));
    }
    return data;
}

int main(int argc, char const *argv[])
{       
    string trainListFilePath = "C:\\Users\\Anwar\\Desktop\\CV Task 5\\images_list.txt";
    vector<Mat> images = readList(trainListFilePath);
    Mat data = PCA_Matrix(images);

    // get the mean of each column

    cout<< data.rows << endl;
    cout<< data.cols << endl;
    
    //resize image
    imshow("Image", data);
    waitKey(0);
    return 0;
}

