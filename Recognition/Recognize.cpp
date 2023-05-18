#include "Recognize.hpp"

#define IMAGE_SIZE 10000



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
    testimage.reshape(1, IMAGE_SIZE).copyTo(test_image_vector);
    return test_image_vector;
}

Mat normalize_test_img(Mat test_image_vector, Mat avgVector)
{
    // subtract test_image_vector from avgVector
    Mat normalized_test_img = avgVector.clone();
    for (int i = 0; i < avgVector.rows; i++)
    {
        normalized_test_img.at<float>(i, 0) = test_image_vector.at<float>(i, 0) - avgVector.at<float>(i, 0);
    }
    return normalized_test_img;
}
Mat calulate_test_weight(Mat normalized_test_img, Mat eigenVector)
{
    Mat test_weight = normalized_test_img.t() * eigenVector.t();
    return test_weight;
}


// Projects an image onto the subspace spanned by the eigenfaces
Mat project_image(Mat test_image, Mat avgVector, Mat eigenVector)
{
    Mat test_image_vector = test_image_to_vector(test_image);

    Mat normalized_test_img = normalize_test_img(test_image_vector, avgVector);

    Mat test_weight = normalized_test_img.t() * eigenVector.t();

    return test_weight;
}

vector<double> calculate_eucledien_distance(Mat weights, Mat test_weight)
{
    vector<double> eucledien_distance;
    for (int i = 0; i < weights.rows; i++)
    {
        double dist = norm(weights.row(i), test_weight, NORM_L2);
        eucledien_distance.push_back(dist);
    }
    return eucledien_distance;
}
vector<double> get_min_k_indexs (vector<double> eucledien_distance , int k)
{
    vector<double> min_k_indexs;
    vector<double> temp = eucledien_distance;
    sort(temp.begin(), temp.end());
    for (int i = 0; i < k; i++)
    {
        auto it = find(eucledien_distance.begin(), eucledien_distance.end(), temp[i]);
        int index = distance(eucledien_distance.begin(), it);
        min_k_indexs.push_back(index);
    }
    // print minimum distance
    cout << eucledien_distance[min_k_indexs[0]]<<endl;

    // if eucledien ,more than variable max
    // cout <<  2.00246e+07<<  endl;
    double in_group= 1;
    // in group

    if ((eucledien_distance[min_k_indexs[0]])  > 2.90246e+07 )
    {
        in_group=  0;
        cout << "OUT OF GROUP" << endl;
    }
    // min_k_indexs.push_back(in_group);


    return min_k_indexs;
}

// Recognizes a face using PCA and a set of training images
vector<double> recognize_face(Mat weights, Mat test_weight)
{
    // calculate eucledien distance
    vector<double> eucledien_distance = calculate_eucledien_distance(weights , test_weight );

int  k=  1;
    // get k min indexes 
    vector<double> min_indexes = get_min_k_indexs(eucledien_distance , k);



    // // git index of minimum eucledien distance
    // auto it = min_element(eucledien_distance.begin(), eucledien_distance.end());
    // int min_index = distance(eucledien_distance.begin(), it);
    // cout << "MIN DISTANCE = " << eucledien_distance[min_index] ;

    return min_indexes;
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

string string_split(string input, string delimiter )
{
    string token = input.substr(0, input.find(delimiter));
    // string token = input.substr(input.find(delimiter)+1,input.length());
    return token;
}