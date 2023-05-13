#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

#include <vector>
#include <string>
#include <numeric>

using namespace cv;
using namespace std;

Mat allFacesMatrix;   // done
Mat avgVector;        // done
Mat subFacesMatrix;   // done
Mat covarianceMatrix; // done
Mat allEigenValues;   // done
Mat allEigenVectors;  // done
Mat K_eigen_vectors;
Mat eigenVector;  // done
Mat weights;      // done
int imgSize = -1; // Dimension of features
int imgRows = -1; // row# of image

#define EIGEN_VECTORS_NUMBER 489

// read training list
vector<string> readList(string listFilePath)
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
    // vector<Mat> images = readImages(facesPath);
    return facesPath;
}

vector<Mat> readImages(string folder_path, vector<string> trainFacesPath)
{
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

Mat read_test_image(string test_image_name)
{
    string folder_path = "E:\\SBME 6th Term\\Computer Vision\\Projects & Tasks\\CV Final Project\\CV_It-is-me\\Images\\Test";
    string file_path = folder_path + "\\" + test_image_name;
    Mat test_image = imread(file_path);
    resize(test_image, test_image, Size(100, 100));
    // convert to gray
    cvtColor(test_image, test_image, COLOR_BGR2GRAY);
    return test_image;
}

Mat test_image_to_vector(Mat testimage)
{
    Mat test_image_vector;
    testimage.convertTo(testimage, CV_32FC1);
    testimage.reshape(1, imgSize).copyTo(test_image_vector);
    return test_image_vector;
}

Mat normalize_test_img(Mat test_image_vector, Mat avgVector)
{
    // subtract test_image_vector from avgVector
    Mat normalized_test_img = avgVector.clone();
    for (int i = 0; i < avgVector.rows; i++)
    {
        normalized_test_img.at<float>(i, 0) = test_image_vector.at<float>(i, 0) - avgVector.at<float>(i, 0);
    }
    return normalized_test_img;
}
Mat calulate_test_weight(Mat normalized_test_img, Mat eigenVector)
{
    Mat test_weight = normalized_test_img.t() * eigenVector.t();
    return test_weight;
}

vector<double> calulate_eucledien_distance(Mat weights, Mat test_weight)
{
    vector<double> eucledien_distance;
    for (int i = 0; i < 50; i++)
    {
        double dist = norm(weights.row(i), test_weight, NORM_L2);
        eucledien_distance.push_back(dist);
    }
    return eucledien_distance;
}

void PCA_getImgSize(Mat sampleImg)
{
    // Dimession of Features
    imgSize = sampleImg.rows * sampleImg.cols;
    // cout << "Per Image Size is: " << size << endl;
}

// put all face images to one matrix, order in column
void PCA_mergeMatrix(vector<Mat> images)
{
    int col = int(images.size());
    allFacesMatrix.create(imgSize, col, CV_32FC1);

    for (int i = 0; i < col; i++)
    {
        Mat tmpMatrix = allFacesMatrix.col(i);
        // Load grayscale image 0
        Mat tmpImg;
        images[i].convertTo(tmpImg, CV_32FC1);
        // convert to 1D matrix
        tmpImg.reshape(1, imgSize).copyTo(tmpMatrix);
    }
}

// compute average face
void PCA_getAverageVector()
{
    // To calculate average face, 1 means that the matrix is reduced to a single column.
    // vector is 1D column vector, face is 2D Mat
    Mat face;
    reduce(allFacesMatrix, avgVector, 1, REDUCE_AVG);
}

void PCA_subtractMatrix()
{
    allFacesMatrix.copyTo(subFacesMatrix);
    for (int i = 0; i < subFacesMatrix.cols; i++)
    {
        subtract(subFacesMatrix.col(i), avgVector, subFacesMatrix.col(i));
    }
}

void PCA_getBestEigenVectors(Mat covarianceMatrix)
{
    // Get all eigenvalues and eigenvectors from covariance matrix
    eigen(covarianceMatrix, allEigenValues, allEigenVectors);

    // select best k eigen vectors
    K_eigen_vectors = Mat::zeros(EIGEN_VECTORS_NUMBER, allEigenVectors.cols, CV_32FC1);
    allEigenVectors.rowRange(Range(0, EIGEN_VECTORS_NUMBER)).copyTo(K_eigen_vectors);

    // convert lower dimension to original dimension
    eigenVector = K_eigen_vectors * (subFacesMatrix.t());
    // cout << "EigenVector size: " << eigenVector.size() << endl;
    // cout << "K_eigen_vectors size: " << K_eigen_vectors.size() << endl;
    // cout << "subFacesMatrix size: " << subFacesMatrix.size() << endl;
    // calculate weights
    weights = subFacesMatrix.t() * eigenVector.t();
}
// PCA Algorithm
void PCA_init(vector<Mat> images)
{
    PCA_getImgSize(images[0]);
    imgRows = images[0].rows;
    PCA_mergeMatrix(images);
    PCA_getAverageVector();
    PCA_subtractMatrix();
    covarianceMatrix = (subFacesMatrix.t()) * subFacesMatrix;
    PCA_getBestEigenVectors(covarianceMatrix);
}

// Projects an image onto the subspace spanned by the eigenfaces
Mat project_image(Mat test_image, Mat avgVector, Mat eigenVector)
{
    Mat test_image_vector = test_image_to_vector(test_image);

    Mat normalized_test_img = normalize_test_img(test_image_vector, avgVector);

    Mat test_weight = normalized_test_img.t() * eigenVector.t();

    return test_weight;
}

vector<double> calculate_eucledien_distance(Mat weights, Mat test_weight)
{
    vector<double> eucledien_distance;
    for (int i = 0; i < weights.rows; i++)
    {
        double dist = norm(weights.row(i), test_weight, NORM_L2);
        eucledien_distance.push_back(dist);
    }
    return eucledien_distance;
}

// Recognizes a face using PCA and a set of training images
int recognize_face(Mat weights, Mat test_weight)
{
    // calculate eucledien distance
    vector<double> eucledien_distance = calculate_eucledien_distance(weights , test_weight );

    // git index of minimum eucledien distance
    auto it = min_element(eucledien_distance.begin(), eucledien_distance.end());
    int min_index = distance(eucledien_distance.begin(), it);
    // cout << "MIN DISTANCE = " << eucledien_distance[min_index] ;

    return min_index;
}

// function to return label of each training persom
vector<string> specify_labels (vector<string> images_files)
{
    vector<string> labels;
    int counter =0 ;
    int person_number = 1;
    string person_str;
    for (int i = 0; i < images_files.size(); i++)
    {
        counter++;
        person_str = "Person : " + std::to_string(person_number);
        labels.push_back(person_str);
        if(counter == 10)
        {
            counter = 0;
            person_number++;
        }
        
    }
    return labels;
}

string string_split(string input,string delimiter = "_")
{
    string token = input.substr(0, input.find(delimiter));
    // string token = input.substr(input.find(delimiter)+1,input.length());
    return token;
}