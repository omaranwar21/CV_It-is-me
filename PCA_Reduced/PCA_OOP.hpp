#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

#include <vector>
#include <string>
#include <numeric>

using namespace std;
using namespace cv;

#define EIGEN_VECTORS_NUMBER 489

class train_PCA
{
    public:
        train_PCA(vector<Mat> images);
        void init(vector<Mat> images);
        Mat getAverageVector(); 
        Mat getEigenVectors(); 
        Mat getWeights(); 
        ~train_PCA();

    private:
        void mergeMatrix(vector<Mat> images);
        void computeAverageVector();
        void normalizedFacesMatrix();
        void computeBestEigenVectors();
        Mat allFacesMatrix;     // done
        Mat avgVector;          // done
        Mat normalizedMatrix;     // done
        Mat covarianceMatrix;   // done
        Mat allEigenValues;     // done
        Mat allEigenVectors;    // done
        Mat K_eigen_vectors;
        Mat eigenVector;        // done
        Mat weights;            // done
        int imgSize = -1;       // Dimension of features
        int imgRows = -1;       // row# of image
};