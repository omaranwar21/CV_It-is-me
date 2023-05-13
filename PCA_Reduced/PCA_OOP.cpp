#include "PCA_OOP.h"

train_PCA::train_PCA(vector<Mat> images)
{
    this->imgSize = images[0].rows*images[0].cols;
    this->imgRows = images[0].rows;
    init(images);

}

void train_PCA::init(vector<Mat> images)
{
    mergeMatrix(images);
    computeAverageVector();
    normalizedFacesMatrix();
    this->covarianceMatrix = (this->normalizedMatrix.t()) * this->normalizedMatrix;
    computeBestEigenVectors();
}

void train_PCA::mergeMatrix(vector<Mat> images)
{
    int col = int(images.size());
    this->allFacesMatrix.create(imgSize, col, CV_32FC1);

    for (int i = 0; i < col; i++)
    {
        Mat tmpMatrix = this->allFacesMatrix.col(i);
        // Load grayscale image 0
        Mat tmpImg;
        images[i].convertTo(tmpImg, CV_32FC1);
        // convert to 1D matrix
        tmpImg.reshape(1, this->imgSize).copyTo(tmpMatrix);
    }
}

void train_PCA::computeAverageVector()
{
    // To calculate average face, 1 means that the matrix is reduced to a single column.
    // vector is 1D column vector, face is 2D Mat
    reduce(this->allFacesMatrix, this->avgVector, 1, REDUCE_AVG);
}

void train_PCA::normalizedFacesMatrix()
{
    this->allFacesMatrix.copyTo(this->normalizedMatrix);
    for (int i = 0; i < this->normalizedMatrix.cols; i++)
    {
        subtract(this->normalizedMatrix.col(i), this->avgVector, this->normalizedMatrix.col(i));
    }
}

void train_PCA::computeBestEigenVectors()
{
    // Get all eigenvalues and eigenvectors from covariance matrix
    eigen(this->covarianceMatrix, this->allEigenValues, this->allEigenVectors);

    // select best k eigen vectors
    this->K_eigen_vectors = Mat::zeros(EIGEN_VECTORS_NUMBER, this->allEigenVectors.cols, CV_32FC1);
    this->allEigenVectors.rowRange(Range(0, EIGEN_VECTORS_NUMBER)).copyTo(this->K_eigen_vectors);

    // convert lower dimension to original dimension
    this->eigenVector = this->K_eigen_vectors * (this->normalizedMatrix.t());

    this->weights = this->normalizedMatrix.t() * this->eigenVector.t();
}

Mat train_PCA::getAverageVector()
{
    return this->avgVector;
}

Mat train_PCA::getEigenVectors()
{
    return this->eigenVector;
}

Mat train_PCA::getWeights()
{
    return this->weights;
}

train_PCA::~train_PCA(){}

/*************************************************************************************************************************/

// int main(int argc, char const *argv[])
// {

//     /* ************ TRACING *********** /
//     /* 1 - READ IMAGES  */

//     // Set the directory and text file containing the training images
//     string training_dir = "C:\\Users\\Anwar\\Desktop\\CV Task 5\\cropped_faces";
//     string training_script_file = "C:\\Users\\Anwar\\Desktop\\CV Task 5\\Train_Images.txt";
//     // // read training list
//     vector<string> train_files = readList(training_script_file);
//     // read images fromt the list
//     vector<Mat> images = readImages(training_dir, train_files);
//     // cout << "IMAGES SIZE : "<< images.size()<<endl;
//     // Specify the labels for each person
//     // vector<string> labels = specify_labels(train_files);
//     // cout << "LABELS SIZE : " << labels.size() << endl;
//     // for (int i =0; i<labels.size(); i++)
//     // {
//     //     cout << labels[i] << endl ;
//     // }

//     /* 2 - Train PCA Model */
//     cout << "*********************************************************Train PCA *******************" << endl;
//     train_PCA pca(images);
//     cout << "********************************************************* PCA Trained*******************" << endl;

//     /* 3 - TEST Vectors */
//     cout << "*********************************************************STEP 3 Recognition*******************" << endl;

//     // Mat test_image = read_test_image("s21_14.jpg");
//     // Set the directory and text file containing the test images
//     // string test_dir = "C:\\Users\\Anwar\\Desktop\\CV Task 5\\cropped_faces";
//     // string test_script_file = "C:\\Users\\Anwar\\Desktop\\CV Task 5\\Test_Images.txt";
//     // // Automatically read the test images
//     // vector<string> test_files = readList(test_script_file);
//     // vector<Mat> test_images = readImages(test_dir, test_files);

//     // vector<int> test_pass;
//     // vector<Mat> data = readData(489);
//     // // loop over test imags to start recognition
//     // for (int i = 0; i < test_files.size(); i++)
//     // {

//     //     Mat test_weight = project_image(test_images[i], data[0], data[1]);
//     //     cout << "------------------------------------------------------------------" << endl;
//     //     /* 4 - Recognition  */
//     //     int index = recognize_face(data[2], test_weight);

//     //     // string label = recognize_face(Training_images_weights, test_weights, labels);44
//     //     string test = string_split(test_files[i]);
//     //     string predicted = string_split(train_files[index]);

//     //     cout << "Test : " << test << " Predicted : " << predicted << endl;
//     //     if (test == predicted)
//     //     {
//     //         cout << "TRUE " << endl;
//     //         test_pass.push_back(1);
//     //     }
//     //     else
//     //     {
//     //         cout << "FALSE " << endl;
//     //         test_pass.push_back(0);
//     //         // cout << cut.compare(h) << endl ;
//     //     }

//     //     cout << " Recognized face in " << test_files[i] << " as " << train_files[index] << " by index : " << index << endl;
//     // }

//     //     // test_pass.
//     // // float accuracy = test_pass.su
//     // double accuracy = std::accumulate(test_pass.begin(), test_pass.end(), 0.0) / test_pass.size() * 100.0;
//     // cout << "Accuracy: " << accuracy << "%" << endl;

//     return 0;
// }