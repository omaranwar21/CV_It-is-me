#ifndef RECOGNIZE
#define RECOGNIZE

#include <Common/common.hpp>

Mat read_test_image(string test_image_name);

Mat test_image_to_vector(Mat testimage);

Mat normalize_test_img(Mat test_image_vector, Mat avgVector);

Mat calulate_test_weight(Mat normalized_test_img, Mat eigenVector);

Mat project_image(Mat test_image, Mat avgVector, Mat eigenVector);

vector<double> calculate_eucledien_distance(Mat weights, Mat test_weight);

vector<double> get_min_k_indexs (vector<double> eucledien_distance, int k);


vector<double> recognize_face(Mat weights, Mat test_weight);

vector<string> specify_labels (vector<string> images_files);

string string_split(string input,string delimiter = "_");




#endif
