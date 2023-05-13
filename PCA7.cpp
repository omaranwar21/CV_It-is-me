// #include <iostream>
// #include <fstream>
// #include <vector>
// #include <string>
// #include <opencv2/opencv.hpp>

// using namespace cv;
// using namespace std;

#include "Best_Funcrtions.cpp"


// Projects an image onto the subspace spanned by the eigenfaces
Mat project_image(const Mat &Test_image, const Mat &mean_face, const Mat &eigenfaces)
{
    // Reshape the image into a column vector 
    Mat test_image_vector = test_image_to_vector(Test_image);

    // Normalize test image (Subtract mean face image from input image)
    Mat normalized_test_img = normalize_test_img(test_image_vector , mean_face );

    // calculate test image weights 
    Mat test_weight =  normalized_test_img.t() * eigenfaces.t();
    
    return test_weight;
}

// Recognizes a face using PCA and a set of training images
string recognize_face( Mat &Training_images_weights, Mat &test_weights, vector<string> &labels)
{

    // Find the closest match among the training images
    // double min_distance = numeric_limits<double>::max();

    // int min_index = -1;

    // for (int i = 0; i < Training_images_weights.rows; i++)
    // {
    //     double dist = euclideanDist(input_weights, Training_images_weights.row(i));

    //     if (dist < min_distance)
    //     {
    //         min_distance = dist;
    //         min_index = i;
    //     }
    // }

    // calculate eucledien distance
    vector<double> eucledien_distance = calulate_eucledien_distance(Training_images_weights , test_weights );

    // git index of minimum eucledien distance
    auto it = min_element(eucledien_distance.begin(), eucledien_distance.end());
    int min_index = distance(eucledien_distance.begin(), it);
    cout << "INDEX = " << min_index <<endl;
    cout << "MIN DISTANCE = " << eucledien_distance[min_index] <<endl;

    // Return the label of the closest match
    return labels[min_index];
}


int main()
{
    // Set the directory and text file containing the training images
    string training_dir = "E:\\SBME 6th Term\\Computer Vision\\Projects & Tasks\\CV Final Project\\CV_It-is-me\\cropped_faces";
    string training_script_file = "E:\\SBME 6th Term\\Computer Vision\\Projects & Tasks\\CV Final Project\\CV_It-is-me\\Train_Images.txt";
    // read training list
    vector<string> train_files = readList(training_script_file);
    // read images fromt the list
    vector<Mat> images = readImages(training_dir,train_files);

    // Specify the labels for each person
    vector<string> labels = specify_labels(train_files) ;

    // Merge the training images into one matrix
    Mat training_images_matrix = merge_images(images);

    // calulate mean face
    Mat mean_face_vector = calculate_average_vector(training_images_matrix );
    // normalize images by mean face
    Mat normalized_training_images = normalize_all_images( training_images_matrix ,mean_face_vector );
    // calculate eigen faces
    Mat k_eigen_faces = compute_K_eigenfaces(normalized_training_images);
    // calculate weights
    Mat Training_images_weights = calculate_weights (normalized_training_images, k_eigen_faces);

//  ********************************* Recognition *************************************
    // Set the directory and text file containing the test images
    string test_dir = "E:\\SBME 6th Term\\Computer Vision\\Projects & Tasks\\CV Final Project\\CV_It-is-me\\cropped_faces";
    string test_script_file = "E:\\SBME 6th Term\\Computer Vision\\Projects & Tasks\\CV Final Project\\CV_It-is-me\\Test_Images.txt";
    // Automatically read the test images
    vector<string> test_files =  readList(test_script_file );
    vector<Mat> test_images = readImages(test_dir,test_files);

    // loop over test imags to start recognition
    // for (const auto &filename : test_files)
    for (int i = 0; i < test_files.size(); i++)
    {
        Mat test_weights = project_image(test_images[i], mean_face_vector, k_eigen_faces);

        cout << "------------------------------------------------------------------" << endl;

        string label = recognize_face(Training_images_weights, test_weights, labels);
        cout << "Recognized face in " << test_files[i] << " as " << label << endl;
    }
    return 0;
}

