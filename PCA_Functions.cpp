#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

vector<Mat> readImages(vector<string> trainFacesPath)
{
    string folder_path = "E:\\SBME 6th Term\\Computer Vision\\Projects & Tasks\\CV Final Project\\CV_It-is-me\\Images\\Train";
    vector<Mat> images;

    for (const auto &filename : trainFacesPath)
    {
        string file_path = folder_path + "\\" + filename;
        Mat img = imread(file_path);
        resize(img, img, Size(100, 100));
        // convert to gray
        cvtColor(img, img, COLOR_BGR2GRAY);
        images.push_back(img);
    }

    cout << "Number of images: " << images.size() << endl;

    return images;
}

// read training list
vector<Mat> readList(string listFilePath)
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
        path.erase(remove(path.begin(), path.end(), ' '), path.end());

        facesPath.push_back(path);
    }
    vector<Mat> images = readImages(facesPath);
    return images;
}


// function to calculate the Mean face from the faces matrix
vector<float> Mean_Face(Mat image)
{
    vector<float> Mean_Image;
    float mean;
    for (int r = 0; r < image.rows; r++)
    {
        mean = 0;
        for (int c = 0; c < image.cols; c++)
        {
            mean += image.at<uchar>(r, c);
        }
        // print image cols
        // cout << image.cols << endl;
        mean = mean / (image.cols);
        Mean_Image.push_back(mean);
    }
    return Mean_Image;
}

// vector<Mat> PCA_Matrix(vector<Mat> images)
// {
//     Mat data;
//     vector<Mat> dataReturned;

//     int n = (int)images.size();
//     int d = images[0].rows * images[0].cols;
//     data = Mat(d, n, images[0].type());

//     for (int i = 0; i < n; i++)
//     {
//         Mat image = images[i].clone().reshape(0, d);
//         image.convertTo(image, images[0].type());
//         image.copyTo(data.col(i));
//     }

//     Mat covarianceMatrix(data.rows, data.rows, CV_32FC1);
//     Mat data_transpose = data.t();
//     dataReturned.push_back(covarianceMatrix);
//     dataReturned.push_back(data);
//     return dataReturned;
// }

// function to transform the image column vector to image shape(100*100)
Mat vectorToImage(vector<float> Mean)
{
    Mat meanImage = Mat(100, 100, CV_8UC1);
    int counter = 0;
    for (int r = 0; r < meanImage.rows; r++)
    {
        for (int c = 0; c < meanImage.cols; c++)
        {
            meanImage.at<uchar>(r, c) = Mean[counter];
            counter++;
        }
    }
    return meanImage;
}

// function to substract from each image column the mean vectorto be  normalized
Mat NormalizeFaces(Mat faces , vector<float> Mean){
    Mat normalized_faces = faces.clone();
    // substract from each row in data_covarianceMatrix[1] o
    for (int c = 0; c < normalized_faces.cols; c++)
    {
        for (int r = 0; r < normalized_faces.rows; r++)
        {
            normalized_faces.at<uchar>(r, c) = normalized_faces.at<uchar>(r, c) - Mean[r];
        }
    }

    return normalized_faces;
}

