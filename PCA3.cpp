#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

Mat allFacesMatrix;         //done
Mat avgVector;              //done
Mat subFacesMatrix;         //done
Mat eigenVector;
int imgSize = -1;//Dimension of features
int imgRows = -1;//row# of image

vector<Mat> readImages(vector<string> trainFacesPath)
{
    string folder_path = "C:\\Users\\Anwar\\Desktop\\CV Task 5\\cropped_faces";
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

// MyPCA::MyPCA(vector<string>& _facesPath)
// {
// 	init(_facesPath);
// }

void PCA_getImgSize(Mat sampleImg)
{
    //Dimession of Features
    imgSize = sampleImg.rows * sampleImg.cols;
    //cout << "Per Image Size is: " << size << endl;
}

//put all face images to one matrix, order in column
void PCA_mergeMatrix(vector<Mat>images)
{
    int col = int(images.size());
    allFacesMatrix.create(imgSize, col, CV_32FC1);
    
    for (int i = 0; i < col; i++) {
        Mat tmpMatrix = allFacesMatrix.col(i);
        //Load grayscale image 0
        Mat tmpImg;
        images[i].convertTo(tmpImg, CV_32FC1);
        //convert to 1D matrix
        tmpImg.reshape(1, imgSize).copyTo(tmpMatrix);
    }
    //cout << "Merged Matix(Width, Height): " << mergedMatrix.size() << endl;
}

//compute average face
void PCA_getAverageVector()
{
    //To calculate average face, 1 means that the matrix is reduced to a single column.
    //vector is 1D column vector, face is 2D Mat
    Mat face;
    reduce(allFacesMatrix, avgVector, 1, REDUCE_AVG);
    
}

void PCA_subtractMatrix()
{
    allFacesMatrix.copyTo(subFacesMatrix);
    for (int i = 0; i < subFacesMatrix.cols; i++) {
        subtract(subFacesMatrix.col(i), avgVector, subFacesMatrix.col(i));
    }
}

void PCA_getBestEigenVectors(Mat covarianceMatrix)
{
    //Get all eigenvalues and eigenvectors from covariance matrix
    Mat allEigenValues, allEigenVectors;
    eigen(covarianceMatrix, allEigenValues, allEigenVectors);
    
    eigenVector = allEigenVectors * (subFacesMatrix.t());
    //Normalize eigenvectors
    for(int i = 0; i < eigenVector.rows; i++ )
    {
        Mat tempVec = eigenVector.row(i);
        normalize(tempVec, tempVec);
    }
    
    Mat eigenFaces, allEigenFaces;
        for (int i = 0; i < eigenVector.rows; i++) {
            eigenVector.row(i).reshape(0, imgRows).copyTo(eigenFaces);
            normalize(eigenFaces, eigenFaces, 0, 1, cv::NORM_MINMAX);
            if(i == 0){
                allEigenFaces = eigenFaces;
            }else{
                hconcat(allEigenFaces, eigenFaces, allEigenFaces);
            }
        }
        cout<< allEigenFaces.rows << endl;
        cout<< allEigenFaces.cols << endl;
        namedWindow("EigenFaces", WINDOW_NORMAL);
        imshow("EigenFaces", allEigenFaces);
        waitKey(0);
}
//PCA Algorithm
void PCA_init(vector<Mat> images)
{
    PCA_getImgSize(images[0]);
    imgRows = images[0].rows;
    PCA_mergeMatrix(images);
    PCA_getAverageVector();
    PCA_subtractMatrix();
    Mat covarianceMatrix = (subFacesMatrix.t()) * subFacesMatrix;
    PCA_getBestEigenVectors(covarianceMatrix);
}



int main(int argc, char const *argv[])
{
    /* code */
    vector<Mat> images = readList("C:\\Users\\Anwar\\Desktop\\CV Task 5\\images_list.txt");
    PCA_init(images);
    return 0;
}

// Mat MyPCA::getFacesMatrix()
// {
//     return allFacesMatrix;
// }

// Mat MyPCA::getAverage()
// {
//     return avgVector;
// }

// Mat MyPCA::getEigenvectors()
// {
//     return eigenVector;
// }

// MyPCA::~MyPCA() {}