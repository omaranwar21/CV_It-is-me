#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <numeric>

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

#define EIGEN_VECTORS_NUMBER 3000

#define ROW_PIXELS 100
#define COL_PIXELS 100

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



// loading image and convert each image toa row (act as column)
Mat load_images(const vector<string> &image_files ,string folder_path )
{
    Mat images;

    string image_path = folder_path + "\\"  + image_files[0] ;

    // Load the first image and get its size
    Mat first_image = imread(image_path, IMREAD_GRAYSCALE);

    if (first_image.empty())
    {
        cerr << "Failed to load image " << image_files[0] << endl;
        return images;
    }

    resize(first_image, first_image, Size(ROW_PIXELS, COL_PIXELS));
    Size image_size = first_image.size();

    // Get the total number of pixels in an image
    int total_pixels = image_size.height * image_size.width;

    // Loop through all images
    for (const auto &filename : image_files)
    {
        string image_path = folder_path + "\\"  + filename ;
        Mat image = imread(image_path, IMREAD_GRAYSCALE);
        if (image.empty())
        {
            cerr << "Failed to load image " << image_path << endl;
            continue;
        }
        // Convert image to 32-bit floating-point format
        image.convertTo(image, CV_32F);

        // Resize the image to match the size of the first image
        if (image.size() != image_size)
        {
            resize(image, image, image_size);
        }
        // Create an empty column for this image
        Mat column(total_pixels, 1, image.type());

        // Reshape the image into a column vector and copy to the column
        Mat reshaped_image = image.reshape(1, total_pixels);
        reshaped_image.copyTo(column);

        // Add this column to the images matrix
        images.push_back(column.t());
    }

    // cout << " the first output images vectors ---------- : " << endl << images << endl;
    return images;
}

// Computes the mean face image and the eigenfaces from a matrix of images
Mat compute_eigenfaces(const Mat &images, Mat &mean, Mat &eigenvectors )
{

    // Calculate the mean vector
    // Mat mean;
    reduce(images, mean, 0, REDUCE_AVG, CV_32F);

    // Create a matrix to store the result
    Mat centered(images.rows, images.cols, CV_32F);

    // Subtract the mean feature vector from each column of the images matrix
    for (int i = 0; i < images.rows; i++)
    {
        for (int j = 0; j < images.cols; j++)
        {
            centered.at<float>(i, j) = images.at<float>(i, j) - mean.at<float>(j);
        }
    }

    // Compute covariance matrix
    Mat centered_trans = centered.t();
    Mat mean_trans = mean.t();

    Mat covar;
    Mat mean_test;

    calcCovarMatrix(centered, covar, mean_test, COVAR_NORMAL | COVAR_ROWS);

    // Compute eigenvalues and eigenvectors of the covariance matrix
    // the eigen vectors is already normalized
    Mat eigenvalues;
    eigen(covar, eigenvalues, eigenvectors);

    cout << "EigenVector size: " << eigenvectors.size() << endl;

    // Select the top k eigenvectors
    // Mat K_eigen_vectors = Mat::zeros(EIGEN_VECTORS_NUMBER, eigenvectors.cols, CV_32FC1);
    // eigenvectors.rowRange(Range(0, EIGEN_VECTORS_NUMBER)).copyTo(K_eigen_vectors);

    Mat eigenfaces = eigenvectors.rowRange(0, EIGEN_VECTORS_NUMBER).clone();

    Mat centered_converted;
    // convert the data type of centered to match the type of eigenvectors.t()
    centered.convertTo(centered_converted, eigenvectors.type()); 
    centered.convertTo(centered_converted, eigenfaces.type()); 


    Mat reconstruct;
    // Mat sum;

    for (int i = 0; i < centered_converted.rows; i++)
    { 
        Mat weights;
        weights.create(eigenvectors.rows,1,eigenvectors.type());
        for(int j = 0; j< eigenvectors.rows ; j++){
            weights.row(j) = eigenvectors.row(j) * centered_converted.row(i).t();
        }

        Mat weights_training = weights.t();

        reconstruct.push_back(weights_training);
    }
    return reconstruct;
    
}

// Projects an image onto the subspace spanned by the eigenfaces
Mat project_image(const Mat &image, const Mat &mean_face, const Mat &eigenfaces)
{

    // Convert image to 32-bit floating-point format

    // Resize gray to match size of mean_face
    Mat Test_image;
    resize(image, Test_image, Size(ROW_PIXELS, COL_PIXELS) );

    // Get the total number of pixels in an image
    int total_pixels = Test_image.rows * Test_image.cols;
    
    // Create an empty column for this image
    Mat column(total_pixels, 1, Test_image.type());

    // Reshape the image into a column vector and copy to the column
    Mat reshaped_image = Test_image.reshape(1, total_pixels);
    reshaped_image.copyTo(column);

    Mat mean_face_trans = mean_face.t();

    // Convert the data type of Test_image to match mean_face_trans
    Mat Test_image_converted;
    column.convertTo(Test_image_converted, mean_face_trans.type());

    // Subtract mean face image from input image
    Mat centered  = Test_image_converted - mean_face_trans;

    Mat covar;
    Mat centered_transpose = centered.t();

    Mat mean_test;

    calcCovarMatrix(centered, covar, mean_test, COVAR_ROWS);

    Mat eigenvalues;
    eigen(covar, eigenvalues, eigenfaces);

    Mat centered_converted;
    // convert the data type of centered to match the type of eigenvectors.t()
    centered_transpose.convertTo(centered_converted, eigenfaces.type()); 

    Mat weights;
        weights.create(eigenfaces.rows,1,eigenfaces.type());
        for(int j = 0; j< eigenfaces.rows ; j++){
            weights.row(j) = eigenfaces.row(j) * centered_converted.row(0).t();
        }
        // vertically concatenate the weights into a single matrix
    return weights;
}

// Computes the distance between two image projections
double euclideanDist(const Mat& A, const Mat& B)
{
    Mat diff = A - B;
    diff = diff.mul(diff);  // element-wise multiplication
    double sum = cv::sum(diff)[0];
    return sqrt(sum);
}


// Recognizes a face using PCA and a set of training images
int recognize_face( Mat &Training_images_weights, Mat &weights, vector<string> &labels)
{
    Mat input_weights = weights.t();

    // Find the closest match among the training images
    double min_distance = numeric_limits<double>::max();

    int min_index = -1;

    for (int i = 0; i < Training_images_weights.rows; i++)
    {
        double dist = euclideanDist(input_weights, Training_images_weights.row(i));

        if (dist < min_distance)
        {
            min_distance = dist;
            min_index = i;
        }
    }

    // Return the label of the closest match
    // return labels[min_index];
    return min_index;


}