#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// #define EIGEN_VECTORS_NUMBER 35
#define PERSON_SAMPLES  10

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

vector<Mat> readImages(string folder_path ,vector<string> trainFacesPath)
{
    // string folder_path = "E:\\SBME 6th Term\\Computer Vision\\Projects & Tasks\\CV Final Project\\CV_It-is-me\\Images\\Train";
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

// function to return label of each training persom
vector<string> specify_labels (vector<string> images_files)
{
    vector<string> labels;
    int counter ;
    int person_number = 1;
    for (int i = 0; i < images_files.size(); i++)
    {
        counter = 0;
        string person_str = "Person : " + std::to_string(person_number);
        for (int counter = 0; counter < PERSON_SAMPLES; counter++)
        {
        labels.push_back(person_str);
        }
        person_number++;
    }
    return labels;
}

//put all face images to one matrix, order in column
Mat merge_images(vector<Mat>images)
{
    Mat allFacesMatrix;
    int col = int(images.size());
    Mat first_image = images[0];
    Size image_size = first_image.size();
    // Get the total number of pixels in an image
    int total_pixels = image_size.height * image_size.width;

    allFacesMatrix.create(total_pixels, col, CV_32FC1);

    for (int i = 0; i < col; i++) {
        Mat tmpMatrix = allFacesMatrix.col(i);
        //Load grayscale image 0
        Mat tmpImg;
        // Convert image to 32-bit floating-point format
        images[i].convertTo(tmpImg, CV_32FC1);
        //convert to 1D matrix
        tmpImg.reshape(1, total_pixels).copyTo(tmpMatrix);
    }
    //cout << "Merged Matix(Width, Height): " << allFacesMatrix.size() << endl;
    return allFacesMatrix;
}

//compute average face
Mat calculate_average_vector(Mat allFacesMatrix )
{
    //To calculate average face, 1 means that the matrix is reduced to a single column.
    //vector is 1D column vector, face is 2D Mat
    Mat average_face_vector;
    reduce(allFacesMatrix, average_face_vector, 1, REDUCE_AVG);
    return average_face_vector;
}

// function to transform the image column vector to image shape(100*100)
Mat columnToImage(Mat Mean)
{
    Mat original_image = Mat(100, 100, CV_8UC1);
    int counter = 0;
    for (int r = 0; r < original_image.rows; r++)
    {
        for (int c = 0; c < original_image.cols; c++)
        {
            original_image.at<uchar>(r, c) = Mean.col(0).at<float>(counter) ;

            counter++;
        }
    }
    return original_image;
}

// function to transform the image column vector to image shape(100*100)
Mat vectorToImage(vector<float> Mean)
{
    Mat meanImage = Mat(100, 100, CV_8UC1);
    int counter = 0;
    for (int r = 0; r < meanImage.rows; r++)
    {
        for (int c = 0; c < meanImage.cols; c++)
        {
            meanImage.at<uchar>(r, c) = Mean[counter];
            counter++;
        }
    }
    return meanImage;
}

Mat normalize_all_images(Mat allFacesMatrix , Mat mean_face_vector)
{
    Mat normalized_faces = allFacesMatrix.clone();

    // allFacesMatrix.copyTo(subFacesMatrix);
    for (int i = 0; i < normalized_faces.cols; i++) {
        subtract(normalized_faces.col(i), mean_face_vector, normalized_faces.col(i));
    }
    return normalized_faces;
    // Subtract the mean feature vector from each column of the images matrix
    // for (int i = 0; i < images.rows; i++)
    // {
    //     for (int j = 0; j < images.cols; j++)
    //     {
    //         centered.at<float>(i, j) = images.at<float>(i, j) - mean.at<float>(j);
    //     }
    // }
    // Mat normalized_faces = faces.clone();
    // // substract from each row in data_covarianceMatrix[1] o
    // for (int c = 0; c < normalized_faces.cols; c++)
    // {
    //     for (int r = 0; r < normalized_faces.rows; r++)
    //     {
    //         normalized_faces.at<uchar>(r, c) = normalized_faces.at<uchar>(r, c) - Mean[r];
    //     }
    // }           

}
//
Mat compute_K_eigenfaces(Mat normalized_images, int K = 400)
{
    cout << "Start Computing Eigen vectors " << endl;
    // calcualte covariance matrix
    Mat covarianceMatrix = (normalized_images.t()) * normalized_images;
    //Get all eigenvalues and eigenvectors from covariance matrix
    Mat allEigenValues ,allEigenVectors ;        
    eigen(covarianceMatrix, allEigenValues, allEigenVectors);

    // select best k eigen vectors
    // k = min(images.rows, images.cols);

    Mat K_eigen_vectors = Mat::zeros(K, allEigenVectors.cols, CV_32FC1); 
    allEigenVectors.rowRange(Range(0, K)).copyTo(K_eigen_vectors); 

    // convert lower dimension to original dimension (compute eigen faces)
    Mat eigen_faces = K_eigen_vectors * (normalized_images.t());
    
    cout << "allEigenVectors size: " << allEigenVectors.size() << endl;
    cout << "K_eigen_vectors size: " << K_eigen_vectors.size() << endl;
    cout << "eigen_faces size: " << eigen_faces.size() << endl;

    return eigen_faces;
}

Mat calculate_weights (Mat normalized_images, Mat eigen_faces )
{
    // calculate weights
    Mat weights = normalized_images.t() * eigen_faces.t();
    return weights;
}

//  ********************************* Recognition *************************************


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
    int total_pixels = testimage.rows * testimage.cols;
    testimage.convertTo(testimage, CV_32FC1);
    testimage.reshape(1, total_pixels).copyTo(test_image_vector);
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

vector<double> calulate_eucledien_distance(Mat weights , Mat test_weight )
{
    vector<double> eucledien_distance ;
    for(int i=0; i< 50 ; i++)
    {
        double dist = norm(weights.row(i), test_weight, NORM_L2);
        eucledien_distance.push_back(dist);
    }    
    return eucledien_distance;
}

// // Computes the distance between two image projections
// double euclideanDist(const Mat& A, const Mat& B)
// {
//     Mat diff = A - B;
//     diff = diff.mul(diff);  // element-wise multiplication
//     double sum = cv::sum(diff)[0];
//     return sqrt(sum);
// }







// Mat dot_product(Mat image1 , Mat image2)
// {
//     Mat dot_product_result(image1.rows, image2.cols, CV_32FC1);

//     // Mat data_transpose = data.t();
//     // cout << data_transpose.size() << endl;
//     // cout << data.size() << endl;

//     // Multiply matrix A and transposed matrix B
//     // note image1 rows = image2 cols
//     for (int i = 0; i < image1.rows; i++)
//     {
//         for (int j = 0; j < image2.cols; j++)
//         {
//             int sum = 0;
//             for (int k = 0; k < image1.cols; k++)
//             {
//                 sum += image1.at<uchar>(i, k) * image2.at<uchar>(k, j);
//             }
//             dot_product_result.at<float>(i, j) = sum;
//         }
//     }
//     return dot_product_result;
// }
