#include "PCA_Reduced//PCA_OOP.hpp"
#include "ReadWrite_Files/ReadWrite.hpp"
#include "Recognition//Recognize.hpp"

#include "common.hpp"



int main(int argc, char const *argv[])
{

    /* ************ TRACING *********** /
    /* 1 - READ IMAGES  */

    // Set the directory and text file containing the training images
    // string training_dir = "C:\\Users\\Anwar\\Desktop\\CV Task 5\\cropped_faces";
    // string training_script_file = "C:\\Users\\Anwar\\Desktop\\CV Task 5\\Train_Images.txt";

    // string training_dir  = "E:\\SBME 6th Term\\Computer Vision\\Projects & Tasks\\CV Final Project\\CV_It-is-me\\cropped_faces" ;
    // string training_script_file =  "E:\\SBME 6th Term\\Computer Vision\\Projects & Tasks\\CV Final Project\\CV_It-is-me\\Train_Images.txt" ;

    string training_dir  = "E:\\SBME 6th Term\\Computer Vision\\Projects & Tasks\\CV Final Project\\CV_It-is-me\\Images\\Team_Data\\Train" ;
    string training_script_file =  "E:\\SBME 6th Term\\Computer Vision\\Projects & Tasks\\CV Final Project\\CV_It-is-me\\image_lists\\team_train.txt" ;

    string write_path = "E:\\SBME 6th Term\\Computer Vision\\Projects & Tasks\\CV Final Project\\CV_It-is-me\\TrainedData";
    ReadWrite r1(write_path) ;
    // // read training list
    vector<string> train_files = r1.readList(training_script_file);
    // read images fromt the list
    vector<Mat> images = r1.readImages(training_dir, train_files);
    // cout << "IMAGES SIZE : "<< images.size()<<endl;
    // Specify the labels for each person
    // vector<string> labels = specify_labels(train_files);
    // cout << "LABELS SIZE : " << labels.size() << endl;
    // for (int i =0; i<labels.size(); i++)
    // {
    //     cout << labels[i] << endl ;
    // }

    /* 2 - Train PCA Model */
    cout << "*********************************************************Train PCA *******************" << endl;
    train_PCA pca(images,r1);
    cout << "********************************************************* PCA Trained*******************" << endl;

    /* 3 - TEST Vectors */
    cout << "*********************************************************STEP 3 Recognition*******************" << endl;

    // Mat test_image = read_test_image("s21_14.jpg");
    // Set the directory and text file containing the test images
    // string test_dir = "C:\\Users\\Anwar\\Desktop\\CV Task 5\\cropped_faces";
    // string test_script_file = "C:\\Users\\Anwar\\Desktop\\CV Task 5\\Test_Images.txt";

    string test_dir  = "E:\\SBME 6th Term\\Computer Vision\\Projects & Tasks\\CV Final Project\\CV_It-is-me\\Images\\Team_Data\\Test" ;
    string test_script_file =  "E:\\SBME 6th Term\\Computer Vision\\Projects & Tasks\\CV Final Project\\CV_It-is-me\\image_lists\\team_test.txt" ;

    // // Automatically read the test images
    vector<string> test_files = r1.readList(test_script_file);
    vector<Mat> test_images = r1.readImages(test_dir, test_files);

    


    vector<int> test_pass;
    // loop over test imags to start recognition
    for (int i = 0; i < test_files.size(); i++)
    {
        cout << "------------------------------------------------------------------" << endl;

        
        // string label = recognize_face(Training_images_weights, test_weights, labels);44
        string test = string_split(test_files[i]);

        Mat test_weight = project_image(test_images[i], pca.getAverageVector(), pca.getEigenVectors());
        /* 4 - Recognition  */
        vector<double> k_min_indexes = recognize_face(pca.getWeights(), test_weight);

        // map 
        map<string, int> myMap;
        for (int i = 0; i < k_min_indexes.size(); i++)
        {
            string label = string_split(train_files[k_min_indexes[i]]);
            myMap[label]++;
        }   
        // print my map keya nd values

        
        for (auto it = myMap.begin(); it != myMap.end(); ++it)
        {
            cout << it->first << " " << it->second << endl;
        }

        // get the key of the max value
        string max_key = "";
        int max_value = 0;
        for (auto it = myMap.begin(); it != myMap.end(); ++it)
        {
            if (it->second > max_value)
            {
                max_value = it->second;
                max_key = it->first;
            }
        }

        string predicted = max_key;

        // string predicted = string_split(train_files[index]);

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

        cout << " Recognized face in " << test_files[i] << " as " << predicted  << endl;
    }

        // test_pass.
    // float accuracy = test_pass.su
    double accuracy = std::accumulate(test_pass.begin(), test_pass.end(), 0.0) / test_pass.size() * 100.0;
    cout << "Accuracy: " << accuracy << "%" << endl;



    // // print test pass
    // for (int i = 0; i < test_pass.size(); i++){
    //     cout << test_pass[i] << endl;
    // }

    // // print test pass size
    // cout << test_pass.size() << endl;
    // // print accumulate
    // cout << std::accumulate(test_pass.begin(), test_pass.end(), 0.0) << endl;






    // vector<int> test_pass;
    // vector<Mat> data = r1.readData();
    // // loop over test imags to start recognition
    // for (int i = 0; i < test_files.size(); i++)
    // {

    //     string test = string_split(test_files[i]);
    //     Mat test_weight = project_image(test_images[i], data[0], data[1]);
    //     cout << "------------------------------------------------------------------" << endl;
    //     /* 4 - Recognition  */
    //     vector<double> k_min_indexes = recognize_face(data[2], test_weight);

    //     // map 
    //     map<string, int> myMap;
    //     for (int i = 0; i < k_min_indexes.size(); i++)
    //     {
    //         string label = string_split(train_files[k_min_indexes[i]]);
    //         myMap[label]++;
    //     }   
    //     // print my map keya nd values

        
    //     for (auto it = myMap.begin(); it != myMap.end(); ++it)
    //     {
    //         cout << it->first << " " << it->second << endl;
    //     }

    //     // get the key of the max value
    //     string max_key = "";
    //     int max_value = 0;
    //     for (auto it = myMap.begin(); it != myMap.end(); ++it)
    //     {
    //         if (it->second > max_value)
    //         {
    //             max_value = it->second;
    //             max_key = it->first;
    //         }
    //     }

    //     string predicted = max_key;


    //     // // string label = recognize_face(Training_images_weights, test_weights, labels);44
    //     // string test = string_split(test_files[i]);
    //     // string predicted = string_split(train_files[index]);

    //     // cout << "Test : " << test << " Predicted : " << predicted << endl;
    //     cout << "Test : " << test << " Predicted : " << predicted << endl;
    //     if (test == predicted)
    //     {
    //         cout << "TRUE " << endl;
    //         test_pass.push_back(1);
    //     }
    //     else
    //     {
    //         cout << "FALSE " << endl;
    //         test_pass.push_back(0);
    //         // cout << cut.compare(h) << endl ;
    //     }

    //     // cout << " Recognized face in " << test_files[i] << " as " << train_files[index] << " by index : " << index << endl;
    // }

    //     // test_pass.
    // // float accuracy = test_pass.su
    // double accuracy = std::accumulate(test_pass.begin(), test_pass.end(), 0.0) / test_pass.size() * 100.0;
    // cout << "Accuracy: " << accuracy << "%" << endl;

    // Debugging

    // vector<Mat> data = r1.readData();

    // Mat model_mean =   pca.getAverageVector() ;
    // Mat model_eigen =  pca.getEigenVectors() ;
    // Mat model_weights = pca.getWeights();

    // Mat files_mean =   data[0] ;
    // Mat files_eigen =  data[1] ;
    // Mat files_weights = data[2];

    // cout << "ENDED" << endl ;




    return 0;
}