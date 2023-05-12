#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

//------------------------------------------ loaded image function -------------------------------------


//
//
//
//
//
//
//
//
//
//
//



// loading image and convert each image toa row (act as column)
Mat load_images(const vector<string> &image_files)
{
    Mat images;

    // Load the first image and get its size
    Mat first_image = imread(image_files[0], IMREAD_GRAYSCALE);

    if (first_image.empty())
    {
        cerr << "Failed to load image " << image_files[0] << endl;
        return images;
    }


    resize(first_image, first_image, Size(30, 30));

    Size image_size = first_image.size();


    // Get the total number of pixels in an image
    int total_pixels = image_size.height * image_size.width;

    // Loop through all images
    for (const auto &filename : image_files)
    {
        Mat image = imread(filename, IMREAD_GRAYSCALE);
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

        // cout << " the output mean_column size : " << column<< endl;
        // cout << " the output images---------- : " << images<< endl;
    }

    // cout << " the first output images vectors ---------- : " << endl << images << endl;

    return images;
}

//---------------------------------------------------------------------------------------------

//
//
//
//
//
//
//
//
//
//
//


// Computes the mean face image and the eigenfaces from a matrix of images
Mat compute_eigenfaces(const Mat &images, Mat &mean, Mat &eigenvectors )
{

    // Calculate the mean vector
    // Mat mean;
    reduce(images, mean, 0, REDUCE_AVG, CV_32F);
    // the output mean :  [121.54167, 121.75, 106.91667, 95.604172]

    // cout << " the output mean      : " << mean << endl;
    // cout << " the output mean      : " << mean_face << endl;

    // cout << " the output mean rows : " << mean.rows << endl;
    // cout << " the output mean cols : " << mean.cols << endl;
    // std::cout << "First element of mean vector: " << mean.at<float>(0, 0) << std::endl;

    // cout << " the output images size : " << images.rows     << endl;
    // cout << " the output mean_column size : " << images.cols<< endl;
    // cout << "First element of images: " <<images.at<float>(0,0) <<endl;

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

    // cout << " the output centered matrix after subtraction: " << endl<< centered<< endl;

    // Compute covariance matrix
    Mat centered_trans = centered.t();
    Mat mean_trans = mean.t();

    // cout << " the output centered trans: "<< endl << centered_trans<< endl;
    // cout << " the output mean_trans      : " << endl<< mean_trans << endl;

    Mat covar;
    Mat mean_test;

    calcCovarMatrix(centered, covar, mean_test, COVAR_NORMAL | COVAR_ROWS);

    // cout << " the output covariance: " << endl << covar << endl;

    // cout << " the output mean after cov      : " << endl<< mean << endl;

    // Compute eigenvalues and eigenvectors of the covariance matrix
    // the eigen vectors is already normalized
    Mat eigenvalues;
    eigen(covar, eigenvalues, eigenvectors);

    // cout << " the output eigenvectors : "<< endl << eigenvectors<< endl;

    // cout << " the output eigenvalues : " << endl << eigenvalues<< endl;

    // Sort eigenvectors by decreasing eigenvalues
    // Mat sorted = eigenvectors.t();

    // cout << " eigenvectors.t : "<< endl << sorted<< endl;

    // Mat sorted_indices;
    // sortIdx(eigenvalues, sorted_indices, SORT_EVERY_COLUMN | SORT_DESCENDING);

    // cout << " the output sorted (2) : "<< endl << sorted_indices << endl;

    // Select the top k eigenvectors
    // int k = min(images.rows, images.cols);
    // if (k > sorted.rows)
    // {
    //     k = sorted.rows;
    // }
    // eigenfaces = sorted.rowRange(0, k).clone();

    Mat centered_converted;
    centered.convertTo(centered_converted, eigenvectors.type()); // convert the data type of centered to match the type of eigenvectors.t()

    // cout << " centered : "<< endl << centered << endl;

    // cout << " centered_rows : "<< endl << centered.rows << endl;
    // cout << " centered_columns : "<< endl << centered.cols << endl;

    // cout << " centered_converted : "<< endl << centered_converted.rows << endl;
    // cout << " centered_converted : "<< endl << centered_converted.cols << endl;

    // cout << " eigenvectors_rows : "<< endl << eigenvectors.rows << endl;
    // cout << " eigenvectors_cols : "<< endl << eigenvectors.cols << endl;

    // Mat weight1;
    // Mat weight2;
    // Mat weight3;
    // Mat weight4;

    Mat reconstruct;
    // Mat sum;

    for (int i = 0; i < centered_converted.rows; i++)
    { 
        Mat weights;
        weights.create(eigenvectors.rows,1,eigenvectors.type());
        for(int j = 0; j< eigenvectors.rows ; j++){
            weights.row(j) = eigenvectors.row(j) * centered_converted.row(i).t();
        }
     
        // cout << " centered_converted : "<< endl << centered_converted.rows << endl;


        // weight1 = eigenvectors.row(0) * centered_converted.row(i).t();
        // weight2 = eigenvectors.row(1) * centered_converted.row(i).t();
        // weight3 = eigenvectors.row(2) * centered_converted.row(i).t();
        // weight4 = eigenvectors.row(3) * centered_converted.row(i).t();

        // vertically concatenate the weights into a single matrix
        // cv::Mat weights;
        // cv::vconcat(weight1, weight2, weights);
        // cv::vconcat(weights, weight3, weights);
        // cv::vconcat(weights, weight4, weights);

        // cout << " weights : "<< endl <<weights << endl;

        // Mat weight11;
        // Mat weight22;
        // Mat weight33;
        // Mat weight44;

        // weight11 = eigenvectors.row(0).t() * weights.row(0);
        // weight22 = eigenvectors.row(1).t() * weights.row(1);
        // weight33 = eigenvectors.row(2).t() * weights.row(2);
        // weight44 = eigenvectors.row(3).t() * weights.row(3);

        //  get the reconstructed innage ---------------------------------
        // Convert mean_trans to the same data type as the weights
        // Mat mean_weight;
        // mean_trans.convertTo(mean_weight, weight11.type());

        // cv::add(weight11, weight22, sum);
        // cv::add(sum, weight33, sum);
        // cv::add(sum, weight44, sum);
        // cv::add(sum, mean_weight, sum);

        // Mat sum_colums = sum.t();



        // vertically concatenate the weights into a single matrix
        // cv::vconcat(weight1, weight2, weights);
        // cv::vconcat(weights, weight3, weights);
        // cv::vconcat(weights, weight4, weights);



        // cout << " sum : "<< endl << sum_colums << endl;

        Mat weights_training = weights.t();

        reconstruct.push_back(weights_training);

    }

    // cout << " reconstruct : "<< endl << reconstruct << endl;

    return reconstruct;

    // cout << " weights : "<< endl << weights << endl;
    // cout << " weight11 : "<< endl << weight11 << endl;
    // cout << " weight22 : "<< endl << weight22 << endl;
    // cout << " weight33 : "<< endl << weight33 << endl;
    // cout << " weight44 : "<< endl << weight44 << endl;
    
}



//
//
//
//
//
//
//
//
//
//
//


// Projects an image onto the subspace spanned by the eigenfaces
Mat project_image(const Mat &image, const Mat &mean_face, const Mat &eigenfaces)
{

    // Convert image to 32-bit floating-point format
    // image.convertTo(image, CV_32F);

    // cout << " image : "<< endl << image << endl;

   
    // Resize gray to match size of mean_face
    Mat Test_image;
    resize(image, Test_image, Size(30, 30) );

    // cout << " Test_image : "<< endl << Test_image << endl;


    // Get the total number of pixels in an image
    int total_pixels = Test_image.rows * Test_image.cols;

    // cout << " total_pixels : "<< endl << total_pixels << endl;

    
    // Create an empty column for this image
    Mat column(total_pixels, 1, Test_image.type());

    // Reshape the image into a column vector and copy to the column
    Mat reshaped_image = Test_image.reshape(1, total_pixels);
    reshaped_image.copyTo(column);

    // cout << " the output mean_column size : " << column<< endl;
    // cout << " the output mean : " << mean_face<< endl;



    Mat mean_face_trans = mean_face.t();
    // cout << " the output mean_face_trans : " << mean_face_trans<< endl;

    // Convert the data type of Test_image to match mean_face_trans
    Mat Test_image_converted;
    column.convertTo(Test_image_converted, mean_face_trans.type());

    // cout << " the output Test_image_converted : " << Test_image_converted.type()<< endl;
    // cout << " the output mean_face_trans : " << mean_face_trans.type()<< endl;


    // Subtract mean face image from input image
    Mat centered  = Test_image_converted - mean_face_trans;

    // cout << " the output centered : " << centered<< endl;

    Mat covar;
    Mat centered_transpose = centered.t();

    // cout << " the output centered_transpose: " << endl << centered_transpose << endl;

    Mat mean_test;

    calcCovarMatrix(centered, covar, mean_test, COVAR_ROWS);

    // cout << " the output covar: " << endl << covar << endl;


    Mat eigenvalues;
    eigen(covar, eigenvalues, eigenfaces);

    // cout << " the output eigenfaces: " << endl << eigenfaces << endl;
    // cout << " the output eigenfaces: " << endl << eigenfaces.type() << endl;
    // cout << " the output eigenfaces: " << endl << eigenfaces.cols << endl;

    // cout << " the output centered: " << endl << centered.type() << endl;

    Mat centered_converted;
    centered_transpose.convertTo(centered_converted, eigenfaces.type()); // convert the data type of centered to match the type of eigenvectors.t()


    // Mat weight1;
    // Mat weight2;
    // Mat weight3;
    // Mat weight4;
     Mat weights;
        weights.create(eigenfaces.rows,1,eigenfaces.type());
        for(int j = 0; j< eigenfaces.rows ; j++){
            weights.row(j) = eigenfaces.row(j) * centered_converted.row(0).t();
        }

    // weight1 = eigenfaces.row(0) * centered_converted.row(0).t();
    // weight2 = eigenfaces.row(1) * centered_converted.row(0).t();
    // weight3 = eigenfaces.row(2) * centered_converted.row(0).t();
    // weight4 = eigenfaces.row(3) * centered_converted.row(0).t();



        // vertically concatenate the weights into a single matrix
        // cv::Mat weights;
        // cv::vconcat(weight1, weight2, weights);
        // cv::vconcat(weights, weight3, weights);
        // cv::vconcat(weights, weight4, weights);

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
string recognize_face( Mat &Training_images_weights, Mat &weights, vector<string> &labels)
{
    Mat input_weights = weights.t();
    // // Project input image onto the subspace spanned by the eigenfaces
    // Mat input_weights = project_image(image, mean_face, eigenfaces);

    // cout << " the output input_weights: " << endl << input_weights << endl;
    // cout << " the output Training_images_weights: " << endl << Training_images_weights << endl;
    
    // cout << " the output labels: " << endl << labels[0] << endl;
    // cout << " the output labels: " << endl << labels[1] << endl;
    // cout << " the output labels: " << endl << labels[2] << endl;


    // Find the closest match among the training images
    double min_distance = numeric_limits<double>::max();

    int min_index = -1;

    for (int i = 0; i < Training_images_weights.rows; i++)
    {
        double dist = euclideanDist(input_weights, Training_images_weights.row(i));

        // cout << " the output dist: " << endl << dist << endl;

        if (dist < min_distance)
        {
            min_distance = dist;
            min_index = i;
        }
    }

    // Return the label of the closest match
    return labels[min_index];
}




int main()
{
    // Set the directory containing the training images
    string training_dir = "E:\\SBME 6th Term\\Computer Vision\\Projects & Tasks\\CV Final Project\\CV_It-is-me\\cropped_faces";

    // Manually specify the labels and file names for each person
    vector<string> person1_files = {
       training_dir + "\\s02_01.jpg" , 
       training_dir + "\\s02_02.jpg",
       training_dir + "\\s02_03.jpg",
       training_dir + "\\s02_04.jpg",
       training_dir + "\\s02_05.jpg",
       training_dir + "\\s02_06.jpg",
       training_dir + "\\s02_07.jpg",
       training_dir + "\\s02_08.jpg",
       training_dir + "\\s02_09.jpg",
       training_dir + "\\s02_10.jpg",
       training_dir + "\\s02_11.jpg",
       training_dir + "\\s02_12.jpg",
       training_dir + "\\s02_13.jpg",
       training_dir + "\\s02_14.jpg"};


    vector<string> person2_files = {
       training_dir + "\\s04_01.jpg" , 
       training_dir + "\\s04_02.jpg",
       training_dir + "\\s04_03.jpg",
       training_dir + "\\s04_04.jpg",
       training_dir + "\\s04_05.jpg",
       training_dir + "\\s04_06.jpg",
       training_dir + "\\s04_07.jpg",
       training_dir + "\\s04_08.jpg",
       training_dir + "\\s04_09.jpg",
       training_dir + "\\s04_10.jpg",
       training_dir + "\\s04_11.jpg",
       training_dir + "\\s04_12.jpg",
       training_dir + "\\s04_13.jpg",
       training_dir + "\\s04_14.jpg"};

   
    vector<string> person3_files = {
       training_dir + "\\s05_01.jpg" , 
       training_dir + "\\s05_02.jpg",
       training_dir + "\\s05_03.jpg",
       training_dir + "\\s05_04.jpg",
       training_dir + "\\s05_05.jpg",
       training_dir + "\\s05_06.jpg",
       training_dir + "\\s05_07.jpg",
       training_dir + "\\s05_08.jpg",
       training_dir + "\\s05_09.jpg",
       training_dir + "\\s05_10.jpg",
       training_dir + "\\s05_11.jpg",
       training_dir + "\\s05_12.jpg",
       training_dir + "\\s05_13.jpg",
       training_dir + "\\s05_14.jpg"};



    // Combine the file name vectors into a single vector
    vector<string> image_files;
    image_files.insert(image_files.end(), person1_files.begin(), person1_files.end());
    image_files.insert(image_files.end(), person2_files.begin(), person2_files.end());
    image_files.insert(image_files.end(), person3_files.begin(), person3_files.end());

    // Manually specify the labels for each person
    vector<string> labels;
    for (int i = 0; i < person1_files.size(); i++)
        labels.push_back("s02_01");
    for (int i = 0; i < person2_files.size(); i++)
        labels.push_back("s04_01");
    for (int i = 0; i < person3_files.size(); i++)
        labels.push_back("s05_01");

    // Load the training images into a matrix
    Size image_size;

    Mat training_images = load_images(image_files);

    // Compute the mean face image and the eigenfaces
    Mat mean_face, eigenfaces , Training_images_weights;

    Training_images_weights = compute_eigenfaces(training_images, mean_face, eigenfaces);


    // Set the directory containing the test images
    string test_dir = "E:\\SBME 6th Term\\Computer Vision\\Projects & Tasks\\CV Final Project\\CV_It-is-me\\cropped_faces";

    // Manually specify the file names for each test image
    vector<string> test_files = { test_dir + "\\s02_15.jpg",
                                  test_dir + "\\s04_15.jpg",
                                  test_dir + "\\s05_15.jpg" };

    // Load the test images and recognize faces
    for (const auto &filename : test_files)
    {
        cout << "Loading test image " << filename << endl;
        Mat image = imread(filename, IMREAD_GRAYSCALE);
        if (image.empty())
        {
            cerr << "Failed to load image " << filename << endl;
            continue;
        }

        Mat weights = project_image(image, mean_face, eigenfaces);

        cout << "------------------------------------------------------------------" << endl;

        string label = recognize_face(Training_images_weights, weights, labels);
        cout << "Recognized face in " << filename << " as " << label << endl;
    }
    return 0;
}

