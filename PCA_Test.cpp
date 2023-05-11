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

vector<float> Mean_Face(Mat image)
{
    // Mat mean;
    vector<float> Mean_Image;
    float mean;
    // loop on image
    for (int r = 0; r < image.rows; r++)
    {
        mean = 0;
        for (int c = 0; c < image.cols; c++)
        {
            mean += image.at<uchar>(r, c);
        }
        // print image rows
        // cout << image.cols << endl;
        mean = mean / (image.cols);
        Mean_Image.push_back(mean);
    }

    return Mean_Image;
}

vector<Mat> PCA_Matrix(vector<Mat> images)
{
    Mat data;
    vector<Mat> dataReturned;

    int n = (int)images.size();
    int d = images[0].rows * images[0].cols;
    data = Mat(d, n, images[0].type());

    for (int i = 0; i < n; i++)
    {
        Mat image = images[i].clone().reshape(0, d);
        image.convertTo(image, images[0].type());
        image.copyTo(data.col(i));
    }

    Mat covarianceMatrix(data.rows, data.rows, CV_32FC1);

    Mat data_transpose = data.t();

    dataReturned.push_back(covarianceMatrix);
    dataReturned.push_back(data);
    return dataReturned;
}

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

int main(int argc, char const *argv[])
{
    string trainListFilePath = "E:\\SBME 6th Term\\Computer Vision\\Projects & Tasks\\CV Final Project\\CV_It-is-me\\Train_images_list.txt";
    vector<Mat> images = readList(trainListFilePath);
    vector<Mat> data_covarianceMatrix = PCA_Matrix(images);

    vector<float> Mean = Mean_Face(data_covarianceMatrix[1]);

    // print shape of Mean
    cout << "-----------Mean-------------" << endl;
    cout << Mean.size() << endl;
    // print     firstelement  in mean
    cout << Mean[0] << endl;
    cout << Mean[1] << endl;

    // loop over average face and put every 100 elements in row in cv mat image of size 100*100

    Mat meanImage = vectorToImage(Mean);

    cout << meanImage.size() << endl;

    int x = meanImage.at<uchar>(0, 0);
    cout << x;
    // cout << "meanImage = " << endl << " "  << meanImage << endl << endl;

    cout << "-----------ROWS-------------" << endl;
    cout << data_covarianceMatrix[1].rows << endl;
    cout << (int)data_covarianceMatrix[1].at<uchar>(0, 0) << endl;
    cout << (int)data_covarianceMatrix[1].at<uchar>(0, 1) << endl;

    // Normalize face
    Mat normalized_faces =  NormalizeFaces(data_covarianceMatrix[1] , Mean);

    cout << "-----------Normalized-------------" << endl;
    cout << (int)normalized_faces.at<uchar>(0, 0) << endl;
    cout << (int)normalized_faces.at<uchar>(0, 1) << endl;
    cout << (int)normalized_faces.at<uchar>(0, 2) << endl;

cout << (int)normalized_faces.at<uchar>(1, 0) << endl;

    return 0;
}
