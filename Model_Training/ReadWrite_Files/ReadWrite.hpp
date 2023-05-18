#ifndef READWRITE_CLASS

#define READWRITE_CLASS

#include "../common.hpp"

class ReadWrite{
    public:
        ReadWrite(string write_folder_path);
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
        string write_folder_path;
};

#endif