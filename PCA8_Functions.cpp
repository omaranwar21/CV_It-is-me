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


Mat readMean()
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

Mat readEigen(int noOfRows)
{
    Mat eigen = Mat::zeros(noOfRows, 10000, CV_32FC1);
    string eigenPath = "C:\\Users\\Anwar\\Desktop\\CV Task 5\\eigen.txt";
    ifstream readEigen(eigenPath.c_str(), ifstream::in);
    
    if (!readEigen) {
        cout << "Fail to open file: " << eigenPath << endl;
    }
    
    string line;
    for (int i = 0; i < noOfRows; i++) {
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

Mat readWeights(int noOfRows)
{
    int noOfCols = noOfRows;
    Mat weights = Mat::zeros(noOfRows, noOfCols, CV_32FC1);
    string weightsPath = "C:\\Users\\Anwar\\Desktop\\CV Task 5\\weights.txt";
    ifstream readweights(weightsPath.c_str(), ifstream::in);
    
    if (!readweights) {
        cout << "Fail to open file: " << weightsPath << endl;
    }
    
    string line;
    for (int i = 0; i < noOfRows; i++) {
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







static void WriteTrainData_writeMean(Mat avg)
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

static void writeEigenVectors(Mat eigen)
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


static void writeWeights(Mat weights)
{
    string weightsPath = "C:\\Users\\Anwar\\Desktop\\CV Task 5\\weights.txt";
    ofstream writeWeightsFile(weightsPath.c_str(), ofstream::out | ofstream::trunc);
    if (!writeWeightsFile) {
        cout << "Fail to open file: " << weightsPath << endl;
    }
    
    for (int i = 0; i < weights.rows; i++) {
        for (int j = 0; j < weights.cols; j++) {
            writeWeightsFile << weights.at<float>(i,j);
            writeWeightsFile << " ";
        }
        writeWeightsFile << "\n";
    }
    
    writeWeightsFile.close();
}

static void writeData(){
    WriteTrainData_writeMean(avgVector);
    writeEigenVectors(eigenVector);
    writeWeights(weights);
}


vector<Mat> readData(int noOfFaces){
    vector<Mat> data(3);
    data[0] = readMean();
    data[1] = readEigen(noOfFaces);
    data[2]= readWeights(noOfFaces);
    return data;
}




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
    string folder_path = "C:\\Users\\Anwar\\Desktop\\CV Task 5\\Images\\Test";
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
            cout << "------------------------------------------------------------------" << endl;
    testimage.reshape(1, testimage.rows*testimage.cols).copyTo(test_image_vector);
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
    writeData();
}

// Projects an image onto the subspace spanned by the eigenfaces
Mat project_image(Mat test_image, Mat avgVector, Mat eigenVector)
{
    Mat test_image_vector = test_image_to_vector(test_image);
    // cout << "------------------------------------------------------------------" << endl;


    Mat normalized_test_img = normalize_test_img(test_image_vector, avgVector);

    Mat test_weight = normalized_test_img.t() * eigenVector.t();

    return test_weight;
}

vector<double> calculate_eucledien_distance(Mat weights, Mat test_weight)
{
    cout << " ====================================================" << endl;
    cout << test_weight.rows << ", " << test_weight.cols << endl;
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