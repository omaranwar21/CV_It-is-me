#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

#include <vector>
#include <string>
#include <numeric>

using namespace cv;
using namespace std;

class ReadWrite{
    public:
        ReadWrite();
        vector<string> readList(string listFilePath);
        vector<Mat> readImages(string folder_path, vector<string> trainFacesPath);
        void writeData(Mat avgVector, Mat eigenVector, Mat weights);
        vector<Mat> readData();

    private:
        void writeMean(Mat avg);
        void writeEigenVectors(Mat eigen);
        void writeWeights(Mat weights);
        Mat readMean();
        Mat readEigen();
        Mat readWeights();
        int noOfRows;
};

ReadWrite::ReadWrite(){
    this->noOfRows = 0;
}

vector<string> ReadWrite::readList(string listFilePath){
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
    return facesPath;
}

vector<Mat> ReadWrite::readImages(string folder_path, vector<string> trainFacesPath){
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
    this->noOfRows = (int)images.size();
    cout << "Number of images: " << images.size() << endl;
    return images;
}

void ReadWrite::writeData(Mat avgVector, Mat eigenVector, Mat weights){
    writeMean(avgVector);
    writeEigenVectors(eigenVector);
    writeWeights(weights);
}

void ReadWrite::writeMean(Mat avg)
{
    string meanPath = "C:\\Users\\Anwar\\Desktop\\CV Task 5\\mean.txt";
    ofstream writeMeanFile(meanPath.c_str(), ofstream::out | ofstream::trunc);
    if (!writeMeanFile) {
        cout << "Fail to open file: " << meanPath << endl;
    }
    
    for (int i = 0; i < avg.rows; i++) {
        writeMeanFile << avg.at<float>(i);
        writeMeanFile << " ";
    }
    
    writeMeanFile.close();
}

void ReadWrite::writeEigenVectors(Mat eigen)
{
    string eigenPath = "C:\\Users\\Anwar\\Desktop\\CV Task 5\\eigen.txt";
    ofstream writeEigenFile(eigenPath.c_str(), ofstream::out | ofstream::trunc);
    if (!writeEigenFile) {
        cout << "Fail to open file: " << eigenPath << endl;
    }
    
    for (int i = 0; i < eigen.rows; i++) {
        for (int j = 0; j < eigen.cols; j++) {
            writeEigenFile << eigen.row(i).at<float>(j);
            writeEigenFile << " ";
        }
        writeEigenFile << "\n";
    }
    
    writeEigenFile.close();
}

void ReadWrite::writeWeights(Mat weights)
{
    string weightsPath = "C:\\Users\\Anwar\\Desktop\\CV Task 5\\weights.txt";
    ofstream writeWeightsFile(weightsPath.c_str(), ofstream::out | ofstream::trunc);
    if (!writeWeightsFile) {
        cout << "Fail to open file: " << weightsPath << endl;
    }
    
    for (int i = 0; i < weights.rows; i++) {
        for (int j = 0; j < weights.cols; j++) {
            writeWeightsFile << weights.row(i).at<float>(j);
            writeWeightsFile << " ";
        }
        writeWeightsFile << "\n";
    }
    
    writeWeightsFile.close();
}

vector<Mat> ReadWrite::readData(){
    vector<Mat> data(3);
    data[0] = readMean();
    data[1] = readEigen();
    data[2]= readWeights();
    return data;
}

Mat ReadWrite::readMean()
{
    Mat mean = Mat::zeros(10000, 1, CV_32FC1);
    string meanPath = "C:\\Users\\Anwar\\Desktop\\CV Task 5\\mean.txt";
    ifstream readMean(meanPath.c_str(), ifstream::in);
    
    if (!readMean) {
        cout << "Fail to open file: " << meanPath << endl;
    }
    
    string line;
    for (int i = 0; i < 1; i++) {
        getline(readMean, line);
        stringstream lines(line);
        for (int j = 0; j < mean.rows; j++) {
            string data;
            getline(lines, data, ' ');
            mean.col(i).at<float>(j) = atof(data.c_str());
        }
    }
    
    readMean.close();
    //cout << mean.col(0).at<float>(1) << endl;
    return mean;
}

Mat ReadWrite::readEigen()
{
    Mat eigen = Mat::zeros(this->noOfRows, 10000, CV_32FC1);
    string eigenPath = "C:\\Users\\Anwar\\Desktop\\CV Task 5\\eigen.txt";
    ifstream readEigen(eigenPath.c_str(), ifstream::in);
    
    if (!readEigen) {
        cout << "Fail to open file: " << eigenPath << endl;
    }
    
    string line;
    for (int i = 0; i < this->noOfRows; i++) {
        getline(readEigen, line);
        stringstream lines(line);
        for (int j = 0; j < eigen.cols; j++) {
            string data;
            getline(lines, data, ' ');
            eigen.at<float>(i,j) = atof(data.c_str());
        }
    }
    
    readEigen.close();
    //cout << eigen.row(14).at<float>(9998) << endl;
    return eigen;
}

Mat ReadWrite::readWeights()
{
    int noOfCols = this->noOfRows;
    Mat weights = Mat::zeros(this->noOfRows, noOfCols, CV_32FC1);
    string weightsPath = "C:\\Users\\Anwar\\Desktop\\CV Task 5\\weights.txt";
    ifstream readweights(weightsPath.c_str(), ifstream::in);
    
    if (!readweights) {
        cout << "Fail to open file: " << weightsPath << endl;
    }
    
    string line;
    for (int i = 0; i < this->noOfRows; i++) {
        getline(readweights, line);
        stringstream lines(line);
        for (int j = 0; j < noOfCols; j++) {
            string data;
            getline(lines, data, ' ');
            weights.at<float>(i,j) = (float)atof(data.c_str());
        }
    }
    
    readweights.close();
    //cout << eigen.row(14).at<float>(9998) << endl;
    return weights;
}

// class 