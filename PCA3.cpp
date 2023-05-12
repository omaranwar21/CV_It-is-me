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
int imgRows = -1;//row# of image.
Mat trainFacesInEigen(30, 30, CV_32FC1);

static vector<Mat> readImages(vector<string> trainFacesPath)
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

    // cout << "Number of images: " << images.size() << endl;

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

//read faces in eigenspace that has been trained
Mat readFaces(int noOfFaces, vector<string>& loadedFaceID)
{
    Mat faces = Mat::zeros(noOfFaces, noOfFaces, CV_32FC1);
    string facesDataPath = "C:\\Users\\Anwar\\Desktop\\CV Task 5\\facesdata.txt";
    ifstream readFaces(facesDataPath.c_str(), ifstream::in);
    
    if (!readFaces) {
        cout << "Fail to open file: " << facesDataPath << endl;
    }
    
    string line, id;
    loadedFaceID.clear();
    for (int i = 0; i < noOfFaces; i++) {
        getline(readFaces, line);
        stringstream lines(line);
        getline(lines, id, ':');
        loadedFaceID.push_back(id);
        for (int j = 0; j < noOfFaces; j++) {
            string data;
            getline(lines, data, ' ');
            faces.col(i).at<float>(j) = atof(data.c_str());
        }
    }
    
    readFaces.close();
    //cout << faces.row(14).at<float>(14) << endl;
    return faces;
}
//read average face of all faces
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
//read eigenvector
Mat readEigen(int noOfFaces)
{
    Mat eigen = Mat::zeros(noOfFaces, 10000, CV_32FC1);
    string eigenPath = "C:\\Users\\Anwar\\Desktop\\CV Task 5\\eigen.txt";
    ifstream readEigen(eigenPath.c_str(), ifstream::in);
    
    if (!readEigen) {
        cout << "Fail to open file: " << eigenPath << endl;
    }
    
    string line;
    for (int i = 0; i < noOfFaces; i++) {
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
// MyPCA::MyPCA(vector<string>& _facesPath)
// {
// 	init(_facesPath);
// }

static void PCA_getImgSize(Mat sampleImg)
{
    //Dimession of Features
    imgSize = sampleImg.rows * sampleImg.cols;
    //cout << "Per Image Size is: " << size << endl;
}

//put all face images to one matrix, order in column
static void PCA_mergeMatrix(vector<Mat>images)
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
    // imshow("allFacesMatrix",allFacesMatrix);
    // cout<< allFacesMatrix.size()<<endl;
    // waitKey(0);
    //cout << "Merged Matix(Width, Height): " << mergedMatrix.size() << endl;
}

//compute average face
static void PCA_getAverageVector()
{
    //To calculate average face, 1 means that the matrix is reduced to a single column.
    //vector is 1D column vector, face is 2D Mat
    // Mat face;
    reduce(allFacesMatrix, avgVector, 1, REDUCE_AVG);
    
}

static void PCA_subtractMatrix()
{
    allFacesMatrix.copyTo(subFacesMatrix);
    for (int i = 0; i < subFacesMatrix.cols; i++) {
        subtract(subFacesMatrix.col(i), avgVector, subFacesMatrix.col(i));
    }
    cout<< subFacesMatrix.size()<< endl;
}

static void PCA_getBestEigenVectors(Mat covarianceMatrix)
{
    //Get all eigenvalues and eigenvectors from covariance matrix
    Mat allEigenValues, allEigenVectors;
    eigen(covarianceMatrix, allEigenValues, allEigenVectors);
    cout<<allEigenVectors.size()<< endl;

    //Get best eigenvalues
    eigenVector = allEigenVectors * (subFacesMatrix.t());

    cout << eigenVector.size()<<endl;
    //Normalize eigenvectors
    for(int i = 0; i < eigenVector.rows; i++ )
    {
        Mat tempVec = eigenVector.row(i);
        normalize(tempVec, tempVec);
    }
    
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

Mat PCA_getFacesMatrix()
{
    return allFacesMatrix;
}

Mat PCA_getAverage()
{
    return avgVector;
}

Mat PCA_getEigenvectors()
{
    return eigenVector;
}


static void WriteTrainData_project(int numberOfFaces)
{
    //cout << "Write Class"<<_trainPCA.getFacesMatrix().size() << endl;
    Mat facesMatrix = PCA_getFacesMatrix();
    Mat avg = PCA_getAverage();
    Mat eigenVec = PCA_getEigenvectors();
    
    for (int i = 0; i < numberOfFaces; i++) {
        Mat temp;
        Mat projectFace = trainFacesInEigen.col(i);
        subtract(facesMatrix.col(i), avg, temp);
        projectFace = eigenVec * temp;
    }
    //cout << trainFacesInEigen.col(0).size() <<endl;
}

static void writeTrainFacesData(int numberOfFaces)
{
    string facesDataPath = "C:\\Users\\Anwar\\Desktop\\CV Task 5\\facesdata.txt";
    ofstream writeFaceFile(facesDataPath.c_str(), ofstream::out | ofstream::trunc);
    if (!writeFaceFile) {
        cout << "Fail to open file: " << facesDataPath << endl;
    }
    
    for (int i = 0; i < numberOfFaces; i++) {
        //writeFaceFile << i + 1 << "#";
        writeFaceFile << i << ":";
        for (int j = 0; j < trainFacesInEigen.rows; j++) {
            writeFaceFile << trainFacesInEigen.col(i).at<float>(j);
            writeFaceFile << " ";
        }
        writeFaceFile << "\n";
    }
    
    writeFaceFile.close();
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

static void writeEigen(Mat eigen)
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

Mat getFacesInEigen()
{
    return trainFacesInEigen;
}


void WriteTrainData()
{
    int numberOfFaces = PCA_getFacesMatrix().cols;
    trainFacesInEigen.create(numberOfFaces, numberOfFaces, CV_32FC1);
    WriteTrainData_project(numberOfFaces);
    writeTrainFacesData(numberOfFaces);
    WriteTrainData_writeMean(PCA_getAverage());
    writeEigen(PCA_getEigenvectors());
}

Mat testVec;
Mat testPrjFace;
string closetFaceID = "None";
double closetFaceDist = 100000000000000000;


void FaceRecognizer_prepareFace(Mat testImg)
{
    testImg.convertTo(testImg, CV_32FC1);
    cout << testImg.type() << endl;
    testImg.reshape(0, testImg.rows*testImg.cols).copyTo(testVec);
    cout << testVec.type() << endl;
}

void FaceRecognizer_projectFace(Mat testVec, Mat _avgVec, Mat _eigenVec){
    // Mat tmpData = testVec.clone();
    
    // cout << _avgVec.size() << endl;
    // cout << testVec.size() << endl;
    Mat test_sub_avgVector = _avgVec.clone();
    for (int i = 0; i< avgVector.cols; i++)
    {
        test_sub_avgVector.at<float>(i,0) = testVec.at<float>(i,0) - _avgVec.at<float>(i,0);
    }
    cout<< testVec.at<float>(55,0) << endl;
    // cout << _eigenVec.type() << endl;
    // cout << tmpData.type() << endl;
    // for (int i = 0; i < 30; i++)
    // {
        /* code */
        // _eigenVec.convertTo(_eigenVec, CV_32FC1);
        // tmpData.convertTo(tmpData, CV_32FC1);
        testPrjFace = _eigenVec * test_sub_avgVector;
        // cout<< i << endl;
    // }
    
    cout << "hear 2 ===================" << endl;
}
//Find the closet Euclidean Distance between input and database
void FaceRecognizer_recognize(Mat testPrjFace, Mat _facesInEigen, vector<string> _loadedFacesID)

// cout << 
{
    for (int i =0; i < _loadedFacesID.size(); i++) {
        Mat src1 = _facesInEigen.row(i);
        Mat src2 = testPrjFace.t();
        
        cout << _loadedFacesID[i];
        double dist = norm(src1, src2, NORM_L2);
        cout << " : " << dist << endl;
        //cout << "Dist " <<dist << endl;
        if (dist < closetFaceDist) {
            closetFaceDist = dist;

            closetFaceID = _loadedFacesID[i];
        }
    }
    //cout  << "id " << closetFaceID << endl;
    //cout << "Closet Distance: " << closetFaceDist << endl;
}

string FaceRecognizer_getClosetFaceID()
{
    return closetFaceID;
}

double FaceRecognizer_getClosetDist()
{
    return closetFaceDist;
}

void FaceRecognizer(Mat testImg, Mat avgVec, Mat eigenVec, Mat facesInEigen, vector<string> _loadedFacesID) {
    FaceRecognizer_prepareFace(testImg);
    FaceRecognizer_projectFace(testVec, avgVec, eigenVec);
    FaceRecognizer_recognize(testPrjFace, facesInEigen, _loadedFacesID);
    
}
int main(int argc, char const *argv[])
{
    /* code */
    vector<Mat> images = readList("C:\\Users\\Anwar\\Desktop\\CV Task 5\\images_list.txt");
    PCA_init(images);
    WriteTrainData();
    vector<string> loadedFacesID;
    Mat avgVec, eigenVec, facesInEigen;
    facesInEigen =  readFaces(int(images.size()), loadedFacesID);
    avgVec = readMean();
    eigenVec = readEigen(int(images.size()));
    Mat testImg = imread("C:\\Users\\Anwar\\Desktop\\CV Task 5\\cropped_faces\\s01_14.jpg", IMREAD_COLOR);
    //resize image
    resize(testImg, testImg, Size(100, 100));

    FaceRecognizer(testImg, avgVec, eigenVec, facesInEigen, loadedFacesID);
    // Show Result
    string faceID = FaceRecognizer_getClosetFaceID();

    cout << "Face ID: " << faceID << endl;
    return 0;
}


// MyPCA::~MyPCA() {}