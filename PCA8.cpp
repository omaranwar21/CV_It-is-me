#include "PCA8_Functions.cpp"

int main(int argc, char const *argv[])
{

    /* ************ TRACING *********** /
    /* 1 - READ IMAGES  */

    // Set the directory and text file containing the training images
    string training_dir = "E:\\SBME 6th Term\\Computer Vision\\Projects & Tasks\\CV Final Project\\CV_It-is-me\\cropped_faces";
    string training_script_file = "E:\\SBME 6th Term\\Computer Vision\\Projects & Tasks\\CV Final Project\\CV_It-is-me\\Train_Images.txt";
    // read training list
    vector<string> train_files = readList(training_script_file);
    // read images fromt the list
    vector<Mat> images = readImages(training_dir, train_files);
    // cout << "IMAGES SIZE : "<< images.size()<<endl;
    // Specify the labels for each person
    vector<string> labels = specify_labels(train_files);
    cout << "LABELS SIZE : " << labels.size() << endl;
    // for (int i =0; i<labels.size(); i++)
    // {
    //     cout << labels[i] << endl ;
    // }

    /* 2 - Train PCA Model */
    cout << "*********************************************************Train PCA *******************" << endl;
    PCA_init(images);
    cout << "********************************************************* PCA Trained*******************" << endl;

    /* 3 - TEST Vectors */
    cout << "*********************************************************STEP 3 Recognition*******************" << endl;

    // Mat test_image = read_test_image("s21_14.jpg");
    // Set the directory and text file containing the test images
    string test_dir = "E:\\SBME 6th Term\\Computer Vision\\Projects & Tasks\\CV Final Project\\CV_It-is-me\\cropped_faces";
    string test_script_file = "E:\\SBME 6th Term\\Computer Vision\\Projects & Tasks\\CV Final Project\\CV_It-is-me\\Test_Images.txt";
    // Automatically read the test images
    vector<string> test_files = readList(test_script_file);
    vector<Mat> test_images = readImages(test_dir, test_files);

    vector<int> test_pass;
    // loop over test imags to start recognition
    for (int i = 0; i < test_files.size(); i++)
    {
        cout << "------------------------------------------------------------------" << endl;

        Mat test_weight = project_image(test_images[i], avgVector, eigenVector);
        /* 4 - Recognition  */
        int index = recognize_face(weights, test_weight);

        // string label = recognize_face(Training_images_weights, test_weights, labels);44
        string test = string_split(test_files[i]);
        string predicted = string_split(train_files[index]);

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

        cout << " Recognized face in " << test_files[i] << " as " << train_files[index] << " by index : " << index << endl;
    }

        // test_pass.
    // float accuracy = test_pass.su
    double accuracy = std::accumulate(test_pass.begin(), test_pass.end(), 0.0) / test_pass.size() * 100.0;
    cout << "Accuracy: " << accuracy << "%" << endl;

    return 0;
}
