#include "PCA6_Functions.cpp"

int main()
{
    cout << "********************************************************* 1- Load Images  *******************" << endl;
    // Set the directory and text file containing the training images
    string training_dir = "E:\\SBME 6th Term\\Computer Vision\\Projects & Tasks\\CV Final Project\\CV_It-is-me\\cropped_faces";

    string training_script_file = "E:\\SBME 6th Term\\Computer Vision\\Projects & Tasks\\CV Final Project\\CV_It-is-me\\Train_images_list.txt";
    // read training list
    vector<string> train_files = readList(training_script_file);

    Mat training_images = load_images(train_files, training_dir);
    // Specify the labels for each person
    vector<string> labels = specify_labels(train_files);

    cout << training_images.size() << endl;

    cout << "********************************************************* 2- Train PCA Model *******************" << endl;

    // Compute the mean face image and the eigenfaces
    Mat mean_face, eigenfaces, Training_images_weights;
    Training_images_weights = compute_eigenfaces(training_images, mean_face, eigenfaces);

    cout << "********************************************************* 3- Read Test Images  *******************" << endl;

    // Set the directory and text file containing the test images
    string test_dir = "E:\\SBME 6th Term\\Computer Vision\\Projects & Tasks\\CV Final Project\\CV_It-is-me\\cropped_faces";
    string test_script_file = "E:\\SBME 6th Term\\Computer Vision\\Projects & Tasks\\CV Final Project\\CV_It-is-me\\Test_Images.txt";

    // Automatically read the test images
    vector<string> test_files = readList(test_script_file);

    cout << "********************************************************* 4- Start Recognition*******************" << endl;

    vector<int> test_pass;

    // Load the test images and recognize faces
    for (int i = 0; i < test_files.size(); i++)
    {
        string file_path = test_dir + "\\" + test_files[i];
        // cout << "Loading test image " << filename << endl;
        Mat image = imread(file_path, IMREAD_GRAYSCALE);
        if (image.empty())
        {
            cerr << "Failed to load image " << test_files[i] << endl;
            continue;
        }

        cout << "------------------------------------------------------------------" << endl;
        Mat weights = project_image(image, mean_face, eigenfaces);
        int index = recognize_face(Training_images_weights, weights, labels);

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
        }

        cout << " Recognized face in " << test_files[i] << " as " << train_files[index] << " by index : " << index << " as " << labels[index] << endl;
    }

    double accuracy = std::accumulate(test_pass.begin(), test_pass.end(), 0.0) / test_pass.size() * 100.0;
    cout << "Accuracy: " << accuracy << "%" << endl;

    return 0;
}
