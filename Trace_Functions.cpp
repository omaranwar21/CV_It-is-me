#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

Mat allFacesMatrix;         //done
Mat avgVector;              //done
Mat subFacesMatrix;         //done
Mat covarianceMatrix;       //done
Mat allEigenValues;         //done
Mat allEigenVectors;        //done
Mat K_eigen_vectors;
Mat eigenVector;        //done
Mat weights;        //done
int imgSize = -1;//Dimension of features
int imgRows = -1;//row# of image

#define EIGEN_VECTORS_NUMBER 35

// vector<Mat> readImages(vector<string> trainFacesPath)
// {
//     string folder_path = "E:\\SBME 6th Term\\Computer Vision\\Projects & Tasks\\CV Final Project\\CV_It-is-me\\Images\\Train";
//     vector<Mat> images;
//     for (const auto &filename : trainFacesPath)
//     {
//         string file_path = folder_path + "\\" + filename;
//         Mat img = imread(file_path);
//         resize(img, img, Size(100, 100));
//         // convert to gray
//         cvtColor(img, img, COLOR_BGR2GRAY);
//         images.push_back(img);
//     }
//     cout << "Number of images: " << images.size() << endl;
//     return images;
// }

// // read training list
// vector<Mat> readList(string listFilePath)
// {
//     vector<string> facesPath;
//     ifstream file(listFilePath.c_str(), ifstream::in);

//     if (!file)
//     {
//         cout << "Fail to open file: " << listFilePath << endl;
//         exit(0);
//     }

//     string line, path, id;
//     while (getline(file, line))
//     {
//         stringstream lines(line);
//         getline(lines, path);

//         path.erase(remove(path.begin(), path.end(), '\r'), path.end());
//         path.erase(remove(path.begin(), path.end(), '\n'), path.end());
//         path.erase(remove(path.begin(), path.end(), ' '), path.end());

//         facesPath.push_back(path);
//     }
//     vector<Mat> images = readImages(facesPath);
//     return images;
// }


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

Mat normalize_test_img(Mat test_image_vector , Mat avgVector )
{
        // subtract test_image_vector from avgVector
    Mat normalized_test_img = avgVector.clone();
    for (int i = 0; i< avgVector.rows; i++)
    {
        normalized_test_img.at<float>(i,0) = test_image_vector.at<float>(i,0) - avgVector.at<float>(i,0);
    }
    return normalized_test_img;
}
Mat calulate_test_weight(Mat normalized_test_img , Mat eigenVector )
{

// Mat diff = weights.clone();
// for (int r = 0 ; r< weights.rows; r++ )
// {
//     for (int c = 0 ; c< weights.cols; c++ )
//     {
//         diff.at<float>(r,c) = weights.at<float>(r,c) - test_weight.at<float>(r,0);
//     }
// }

    // // Mat diff = weights - test_weight;
    // cout << "diff size: " << diff.size() << endl;   
    // cout << diff.at<float>(0, 0) << endl;
    // // calculate norm vector
    // vector<float> Norm_Vector ;
    // float norm;
    // for (int r = 0 ; r< diff.rows; r++ )
    // {
    //     norm = 0;
    //     for (int c = 0 ; c< diff.cols; c++ )
    //     {
    //         norm += pow(diff.at<float>(r, c), 2);
    //     }
    //     norm = sqrt(norm);
    //     Norm_Vector.push_back(norm);
    // }




    Mat test_weight =  normalized_test_img.t() * eigenVector.t();
    return test_weight;
}

vector<double>  calulate_eucledien_distance(Mat weights , Mat test_weight )
{
    vector<double> eucledien_distance ;
    for(int i=0; i< 50 ; i++)
    {
        double dist = norm(weights.row(i), test_weight, NORM_L2);
        eucledien_distance.push_back(dist);
    }    
    return eucledien_distance;
}

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

Mat dot_product(Mat image1 , Mat image2)
{
    Mat dot_product_result(image1.rows, image2.cols, CV_32FC1);

    // Mat data_transpose = data.t();
    // cout << data_transpose.size() << endl;
    // cout << data.size() << endl;

    // Multiply matrix A and transposed matrix B
    // note image1 rows = image2 cols
    for (int i = 0; i < image1.rows; i++)
    {
        for (int j = 0; j < image2.cols; j++)
        {
            int sum = 0;
            for (int k = 0; k < image1.cols; k++)
            {
                sum += image1.at<uchar>(i, k) * image2.at<uchar>(k, j);
            }
            dot_product_result.at<float>(i, j) = sum;
        }
    }
    return dot_product_result;
}


void PCA_getBestEigenVectors(Mat covarianceMatrix)
{
    //Get all eigenvalues and eigenvectors from covariance matrix
    eigen(covarianceMatrix, allEigenValues, allEigenVectors);

    // select best k eigen vectors
    K_eigen_vectors = Mat::zeros(EIGEN_VECTORS_NUMBER, allEigenVectors.cols, CV_32FC1); 
    allEigenVectors.rowRange(Range(0, EIGEN_VECTORS_NUMBER)).copyTo(K_eigen_vectors); 

    // convert lower dimension to original dimension
    eigenVector = K_eigen_vectors * (subFacesMatrix.t());
    
    cout << "EigenVector size: " << eigenVector.size() << endl;
    cout << "K_eigen_vectors size: " << K_eigen_vectors.size() << endl;
    cout << "subFacesMatrix size: " << subFacesMatrix.size() << endl;


    // calculate weights
    weights =  subFacesMatrix.t() * eigenVector.t();
}
//PCA Algorithm
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

