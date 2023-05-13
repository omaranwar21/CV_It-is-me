#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <numeric>

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

#define EIGEN_VECTORS_NUMBER 3250

#define ROW_PIXELS 100
#define COL_PIXELS 100

// loading image and convert each image toa row (act as column)
Mat load_images(const vector<string> &image_files , string training_dir)
{
    Mat images;

    string train_path = training_dir +  "\\" + image_files[0];

    // Load the first image and get its size
    Mat first_image = imread(train_path, IMREAD_GRAYSCALE);

    if (first_image.empty())
    {
        cerr << "Failed to load image " << image_files[0] << endl;
        return images;
    }

    resize(first_image, first_image, Size(ROW_PIXELS, ROW_PIXELS));

    Size image_size = first_image.size();

    // Get the total number of pixels in an image
    int total_pixels = image_size.height * image_size.width;

    // Loop through all images
    for (const auto &filename : image_files)
    {
        string train_path = training_dir +  "\\" + filename;
        Mat image = imread(train_path, IMREAD_GRAYSCALE);
        if (image.empty())
        {
            cerr << "Failed to load image " << filename << endl;
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

    return images;
}

// Computes the mean face image and the eigenfaces from a matrix of images
Mat compute_eigenfaces(const Mat &images, Mat &mean, Mat &eigenvectors)
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

    Mat eigenfaces = eigenvectors.rowRange(0, EIGEN_VECTORS_NUMBER).clone();

    eigenvectors = eigenfaces;

    Mat centered_converted;
    centered.convertTo(centered_converted, eigenvectors.type()); // convert the data type of centered to match the type of eigenvectors.t()

    Mat reconstruct;
    // Mat sum;

    for (int i = 0; i < centered_converted.rows; i++)
    {
        Mat weights;
        weights.create(eigenvectors.rows, 1, eigenvectors.type());
        for (int j = 0; j < eigenvectors.rows; j++)
        {
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
    // Resize gray to match size of mean_face
    Mat Test_image;
    resize(image, Test_image, Size(ROW_PIXELS, ROW_PIXELS));

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
    Mat centered = Test_image_converted - mean_face_trans;

    Mat centered_transpose = centered.t();

    Mat centered_converted;
    centered_transpose.convertTo(centered_converted, eigenfaces.type()); // convert the data type of centered to match the type of eigenvectors.t()

    Mat weights;
    weights.create(eigenfaces.rows, 1, eigenfaces.type());
    for (int j = 0; j < eigenfaces.rows; j++)
    {
        weights.row(j) = eigenfaces.row(j) * centered_converted.row(0).t();
    }

    return weights;
}

// Computes the distance between two image projections
double euclideanDist(const Mat &A, const Mat &B)
{
    Mat diff = A - B;
    diff = diff.mul(diff); // element-wise multiplication
    double sum = cv::sum(diff)[0];
    return sqrt(sum);
}

// Recognizes a face using PCA and a set of training images
int recognize_face(Mat &Training_images_weights, Mat &weights, vector<string> &labels)
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
        if(counter == 9)
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


int main()
{
    // Set the directory containing the training images
    string training_dir = "E:\\SBME 6th Term\\Computer Vision\\Projects & Tasks\\CV Final Project\\CV_It-is-me\\cropped_faces";

    string training_script_file = "E:\\SBME 6th Term\\Computer Vision\\Projects & Tasks\\CV Final Project\\CV_It-is-me\\Train_Images.txt";

    // Combine the file name vectors into a single vector

    // vector<string> image_files = readList(training_script_file);

    vector<string> image_files;

    string file_name;
    for (int sub = 1; sub <= 9; sub++)
    {
        for (int sample = 1; sample <= 9; sample++)
        {
            // if(sub == 4)
            // {
            //     continue;
            // }
            file_name = "s0"+ std::to_string(sub)+ "_0"+  std::to_string(sample)+  +".jpg" ;
            image_files.push_back(file_name);
            cout << "Train path " << file_name << endl ;
        }
    }

    vector<string> labels = specify_labels(image_files);
    // print labels size
    cout << labels.size() << endl;
    // print labels
    for (int i = 0; i < labels.size(); i++)
    {
        cout << labels[i] << endl;
    }

    // Load the training images into a matrix
    Size image_size;

    Mat training_images = load_images(image_files , training_dir);

    cout << "********************************************************* 2- Train PCA Model *******************" << endl;
    // Compute the mean face image and the eigenfaces
    Mat mean_face, eigenfaces, Training_images_weights;

    Training_images_weights = compute_eigenfaces(training_images, mean_face, eigenfaces);

    // Set the directory containing the test images
    string test_dir = "E:\\SBME 6th Term\\Computer Vision\\Projects & Tasks\\CV Final Project\\CV_It-is-me\\cropped_faces";

    // string test_script_file = "E:\\SBME 6th Term\\Computer Vision\\Projects & Tasks\\CV Final Project\\CV_It-is-me\\Test_Images.txt";
    // vector<string> test_files = readList(test_script_file);

    vector<string> test_files ;
    string test_file_name;
    for (int sub = 1; sub <= 9; sub++)
    {
        for (int sample = 10; sample <= 15; sample++)
        {
            test_file_name = "s0"+ std::to_string(sub)+ "_"+  std::to_string(sample)+  +".jpg" ;
            test_files.push_back(test_file_name);
            cout << "Test path " << test_file_name << endl ;
        }
    }


    cout << "********************************************************* 4- Start Recognition*******************" << endl;

    vector<int> test_pass;

    // Load the test images and recognize faces
    for (int i = 0; i < test_files.size(); i++)
    {
        
        string test_path = test_dir +  "\\" + test_files[i];
        // cout << "Loading test image " << filename << endl;
        Mat image = imread(test_path, IMREAD_GRAYSCALE);
        if (image.empty())
        {
            cerr << "Failed to load image " << test_files[i] << endl;
            continue;
        }

        Mat weights = project_image(image, mean_face, eigenfaces);

        cout << "------------------------------------------------------------------" << endl;

                // string label = recognize_face(Training_images_weights, test_weights, labels);44
        int index = recognize_face(Training_images_weights, weights, labels);

        string test = string_split(test_files[i]);
        string predicted = string_split(image_files[index]);

        cout << "Test : " << test << " Predicted : " << predicted << endl;
        if (test == predicted)
        {
            cout << "TRUE " << endl;
            test_pass.push_back(1);
        }
        else
        {
            cout << "FALSE " << endl;
            test_pass.push_back(0);
            // cout << cut.compare(h) << endl ;
        }

        cout << "Recognized face in " << test_files[i] << " as  : "   <<  labels[index] << "  With index :  " << index << endl <<" Predicted as : "  <<image_files[index] << endl;

        cout <<"* *****************************************************"<< endl  ;
    }
        double accuracy = std::accumulate(test_pass.begin(), test_pass.end(), 0.0) / test_pass.size() * 100.0;
    cout << "Accuracy: " << accuracy << "%" << endl;


    cout <<"* *****************************************************"<< endl  ;
    return 0;
}
