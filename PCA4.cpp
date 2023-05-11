#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

Mat allFacesMatrix;         //done
Mat avgVector;              //done
Mat subFacesMatrix;         //done
Mat eigenVector;
Mat weights;
int imgSize = -1;//Dimension of features
int imgRows = -1;//row# of image


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
    testimage.reshape(1, imgSize).copyTo(test_image_vector);
    return test_image_vector;
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
    Mat allEigenValues, allEigenVectors;
    eigen(covarianceMatrix, allEigenValues, allEigenVectors);

    // print type of all eigen vectos
    cout << "EigenVector type: " << allEigenVectors.type() << endl; 

    cout << allEigenVectors.at<float>(0, 0) << endl;
    cout << "OLD EigenVector size: " << allEigenVectors.size() << endl;
    cout << "EigenValues size: " << allEigenValues.size() << endl;
    eigenVector = allEigenVectors * (subFacesMatrix.t());
    //Normalize eigenvectors
    for(int i = 0; i < eigenVector.rows; i++ )
    {
        Mat tempVec = eigenVector.row(i);
        normalize(tempVec, tempVec);
    }
    // eigenVector = subFacesMatrix * eigenVector;
    cout << "EigenVector size: " << eigenVector.size() << endl;
    cout << "subFacesMatrix size: " << subFacesMatrix.size() << endl;

    // calculate weights
    // Mat weights = dot_product( eigenVector ,  subFacesMatrix);
    weights =  eigenVector *  subFacesMatrix;
    // cout << weights.at<float>(0, 0) << endl;

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
    vector<Mat> images = readList( "E:\\SBME 6th Term\\Computer Vision\\Projects & Tasks\\CV Final Project\\CV_It-is-me\\Train_images_list.txt");
    PCA_init(images);

    cout << "subFacesMatrix size: " << subFacesMatrix.size() << endl;
    cout << "EigenVector size: " << eigenVector.size() << endl;
    cout << "weights size: " << weights.size() << endl;
    cout << weights.at<float>(0, 0) << endl;

    Mat test_image = read_test_image( "s01_14.jpg");
    // namedWindow("test_image", WINDOW_NORMAL);
    // imshow("test_image", test_image);
    // waitKey(0);

    Mat test_image_vector = test_image_to_vector( test_image);

    // print test_image_vector size
    cout << "test_image_vector size: " << test_image_vector.size() << endl;

    cout << "avgVector size: " << avgVector.size() << endl;    

    // cout<< (int)test_image_vector.at<uchar>(0,0) <<endl ;
    // cout<< (int)test_image_vector.at<uchar>(1,0) <<endl ;

    // cout<< (int)avgVector.at<float>(0,0) <<endl ;
    // cout<< (int)avgVector.at<float>(1,0) <<endl ;
    // subtract test_image_vector from avgVector
    Mat test_sub_avgVector = avgVector.clone();
    for (int i = 0; i< avgVector.cols; i++)
    {
        test_sub_avgVector.at<float>(i,0) = test_image_vector.at<uchar>(i,0) - avgVector.at<float>(i,0);
    }
    cout << "test_sub_avgVector size: " << test_sub_avgVector.size() << endl; 
    cout << "eigenVector size: " << eigenVector.size() << endl; 
    cout << "test_sub_avgVector Transpose size: " << test_sub_avgVector.t().size() << endl;

    // print data types of eigenvctor and test
    cout << "EigenVector type: " << eigenVector.type() << endl;
    cout << "test_sub_avgVector type: " << test_sub_avgVector.type() << endl;

    Mat test_weight = eigenVector * test_sub_avgVector;
    // Mat test_weight = eigenVector.dot(test_sub_avgVector)  ;

    cout << "test_weight size: " << test_weight.size() << endl;
    cout << "weights size: " << weights.size() << endl;

    // double distance = norm(test_weight, weights, NORM_L2);
    // cout << "distance size: " << distance.size() << endl;
    // cout << distance << endl;
    // weights - test weights 
    cout << "Before NORM" << endl;
    vector<float> Norm_Vector ;
    cout << "Weight COLS size: " << weights.col(1).size <<endl;
    cout << "Weight Rows size: " << weights.row(1).size <<endl;
    cout << "test_weight size: " << test_weight.size() << endl;

    Mat new_weights = weights.clone();
    new_weights.row(1).reshape(0, 50);
    cout << "New Weight Rows size: " << new_weights.row(1).size <<endl;
    

    // for(int i=0; i< 50 ; i++)
    // {
    //     weights.row(i).reshape(0, 50);
    //     double dist = norm(weights.col(i), test_weight, NORM_L2);
    //     // double dist = norm(weights.col(i), test_weight, NORM_L2);
    //     Norm_Vector.push_back(dist);
    // }

    cout << "AFTER NORM" << endl;


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

    // print norm size
    cout << "Norm_Vector size: " << Norm_Vector.size() << endl;
    // print norm vector

    // for (int i = 0; i < Norm_Vector.size(); i++)
    // {
    //     cout << Norm_Vector[i] << endl;
    // }
    auto it = min_element(Norm_Vector.begin(), Norm_Vector.end());
    int index = distance(Norm_Vector.begin(), it);
    cout << "INDEX = " << index <<endl;


    // cout << "diff: " << diff << endl;

    // cout << weights.at<float>(0, 0) << endl;
    // cout << weights.at<float>(0, 1) << endl;
    // cout << weights.at<float>(1, 0) << endl;

    // cout << test_weight.at<float>(0, 0) << endl;
    // cout << test_weight.at<float>(1, 0) << endl;

    // cout<< (int)test_image_vector.at<uchar>(0,0) <<endl ;
    // cout<< (int)avgVector.at<float>(0,0) <<endl ;
    // cout<< (int)test_sub_avgVector.at<float>(0,0);

    // cout << test_weight.at<float>(0, 0) << endl;

    // cout << weights.at<float>(0, 0) << endl;

    return 0;
}
