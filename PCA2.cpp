#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

vector<Mat> readImages(vector<string> trainFacesPath)
{
    string folder_path = "C:\\Users\\Anwar\\Desktop\\CV Task 5\\cropped_faces";
    vector<Mat> images;

    for (const auto& filename : trainFacesPath) {
        string file_path = folder_path + "\\" + filename;
        Mat img = imread(file_path);
        resize(img, img, Size(100, 100));
        //convert to gray
        cvtColor(img, img, COLOR_BGR2GRAY);
        images.push_back(img);
    }

    cout << "Number of images: " << images.size() << endl;

    return images;
}

//read training list
vector<Mat> readList(string listFilePath)
{
    vector<string> facesPath;
    ifstream file(listFilePath.c_str(), ifstream::in);
    
    if (!file) {
        cout << "Fail to open file: " << listFilePath << endl;
        exit(0);
    }
    
    string line, path, id;
    while (getline(file, line)) {
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

float Average_Face(Mat image){
    // Mat mean; 
    float mean=0 ;
    // loop on image
    for (int c = 0; c < image.cols; c++){
        for (int r=0; r<image.rows; r++){
            mean += image.at<uchar>(r, c);
        }
    }
    mean = mean / (image.rows * image.cols);

    return mean;
}

Mat PCA_Matrix(vector<Mat> images){
    Mat data ;
    // average vector
    // vector<float> means;
    
    
    int n = (int)images.size();
    int d = images[0].rows * images[0].cols;
    // cout<< n <<"," << d << endl;
    data = Mat(d, n, images[0].type());
    // cout<< data.rows << endl;
    // cout<< data.cols << endl;
    // int counter = 0;

    for (int i = 0; i < n; i++){

        // means.pushback(Average_Face)
        float mean = Average_Face(images[i]);
        // imshow("image wo avg", images[i]);
        images[i] = images[i]- (int)mean;

        Mat image = images[i].clone().reshape(0, d);
        image.convertTo(image, images[0].type());
        image.copyTo(data.col(i));
    }
    // transpose data image
    // cout << data.t().size() << endl;
    // imshow("ttttt", data.t());
    // cout<< ((data)*(data.t())).size() << endl;

    // Mat covarianceMatrix = (Mat_<uchar>(10000, 10000));
    Mat covarianceMatrix(data.rows, data.rows, data.type());
    // covarianceMatrix = (data).cross((data.t()));

    Mat data_transpose = data.t();
    // cout << data_transpose.size() << endl;
    // cout << data.size() << endl;


    // Multiply matrix A and transposed matrix B
    for (int i = 0; i < data.rows; i++) {
        for (int j = 0; j < data.rows; j++) {
            int sum = 0;
            for (int k = 0; k < data.cols; k++) {
                sum += data.at<uchar>(i,k) * data_transpose.at<uchar>(k,j);
            }
            covarianceMatrix.at<uchar>(i,j) = sum;
        }
    }
    // cout<< covarianceMatrix.rows << endl;
    // data = data.t();
    return covarianceMatrix;
}






int main(int argc, char const *argv[])
{       
    string trainListFilePath = "C:\\Users\\Anwar\\Desktop\\CV Task 5\\images_list.txt";
    vector<Mat> images = readList(trainListFilePath);
    Mat data = PCA_Matrix(images);

    // get the mean of each column

    // cout<< data.rows << endl;
    // cout<< data.cols << endl;
    
    //resize image
    imshow("Image", data);
    waitKey(0);
    return 0;
}

