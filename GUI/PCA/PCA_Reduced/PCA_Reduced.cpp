#include "PCA/PCA_Reduced/PCA_Reduced.hpp"

train_PCA::train_PCA(vector<Mat> images, ReadWrite p1)
{
    this->imgSize = images[0].rows*images[0].cols;
    this->imgRows = images[0].rows;
    init(images,p1);

}

void train_PCA::init(vector<Mat> images,ReadWrite p1)
{
    mergeMatrix(images);
    computeAverageVector();
    normalizedFacesMatrix();
    this->covarianceMatrix = (this->normalizedMatrix.t()) * this->normalizedMatrix;
    computeBestEigenVectors();
    p1.writeData(this->avgVector, this->eigenVector,  this->weights);
}

void train_PCA::mergeMatrix(vector<Mat> images)
{
    int col = int(images.size());
    this->allFacesMatrix.create(imgSize, col, CV_32FC1);

    for (int i = 0; i < col; i++)
    {
        Mat tmpMatrix = this->allFacesMatrix.col(i);
        // Load grayscale image 0
        Mat tmpImg;
        images[i].convertTo(tmpImg, CV_32FC1);
        // convert to 1D matrix
        tmpImg.reshape(1, this->imgSize).copyTo(tmpMatrix);
    }
}

void train_PCA::computeAverageVector()
{
    // To calculate average face, 1 means that the matrix is reduced to a single column.
    // vector is 1D column vector, face is 2D Mat
    reduce(this->allFacesMatrix, this->avgVector, 1, REDUCE_AVG);
}

void train_PCA::normalizedFacesMatrix()
{
    this->allFacesMatrix.copyTo(this->normalizedMatrix);
    for (int i = 0; i < this->normalizedMatrix.cols; i++)
    {
        subtract(this->normalizedMatrix.col(i), this->avgVector, this->normalizedMatrix.col(i));
    }
}

void train_PCA::computeBestEigenVectors()
{
    // Get all eigenvalues and eigenvectors from covariance matrix
    eigen(this->covarianceMatrix, this->allEigenValues, this->allEigenVectors);

    // select best k eigen vectors
    this->K_eigen_vectors = Mat::zeros(EIGEN_VECTORS_NUMBER, this->allEigenVectors.cols, CV_32FC1);
    this->allEigenVectors.rowRange(Range(0, EIGEN_VECTORS_NUMBER)).copyTo(this->K_eigen_vectors);

    // convert lower dimension to original dimension
    this->eigenVector = this->K_eigen_vectors * (this->normalizedMatrix.t());

    this->weights = this->normalizedMatrix.t() * this->eigenVector.t();
}

Mat train_PCA::getAverageVector()
{
    return this->avgVector;
}

Mat train_PCA::getEigenVectors()
{
    return this->eigenVector;
}

Mat train_PCA::getWeights()
{
    return this->weights;
}

train_PCA::~train_PCA(){}
