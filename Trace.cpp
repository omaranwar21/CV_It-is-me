#include "Trace_Functions.cpp"

int main(int argc, char const *argv[])
{

    /* ************ TRACING *********** /
    /* 1 - READ IMAGES  */
    vector<Mat> images = readList( "E:\\SBME 6th Term\\Computer Vision\\Projects & Tasks\\CV Final Project\\CV_It-is-me\\Train_images_list.txt");
    // cout << "IMAGES SIZE : "<< images.size()<<endl;
    
    PCA_init(images);

    /* 2 - IMAGES  Vectors */
    cout << "*************STEP 2*******************" << endl ;
    // **************************************** allFacesMatrix DONE* ******************************
    cout << "allFacesMatrix size: " << allFacesMatrix.size() << endl;
    // // cout << "allFacesMatrix COL 1: " << allFacesMatrix.col(0) << endl;
    // cout << "allFacesMatrix COL 1 pixel 1: " << allFacesMatrix.col(0).at<float>(0) << endl;
    // cout << "allFacesMatrix COL 1 pixel 2: " << allFacesMatrix.col(0).at<float>(1) << endl;
    // cout << "allFacesMatrix COL 1 pixel 2: " << allFacesMatrix.col(0).at<float>(9998) << endl;
    // cout << "allFacesMatrix COL 1 pixel 2: " << allFacesMatrix.col(0).at<float>(9999) << endl;
    // cout << "allFacesMatrix COL 1 pixel 1 : " << allFacesMatrix.at<float>(0, 0) << endl;
    // cout << "allFacesMatrix COL 1 pixel 2: " << allFacesMatrix.at<float>(1, 0) << endl;

    // **************************************** Average vector DONE* ******************************
    cout << "avgVector size: " << avgVector.size() << endl;
    // cout << "avgVector COL 1 pixel 1: " << avgVector.col(0).at<float>(0) << endl;
    // cout << "avgVector COL 1 pixel 2: " << avgVector.col(0).at<float>(1) << endl;
    // cout << "avgVector COL 1 pixel 2: " << avgVector.col(0).at<float>(9998) << endl;
    // cout << "avgVector COL 1 pixel 2: " << avgVector.col(0).at<float>(9999) << endl;
    // cout << "avgVector col 1: " << avgVector.col(0) << endl;


    // **************************************** subFacesMatrix Semi Done ******************************
    cout << "subFacesMatrix size: " << subFacesMatrix.size() << endl;
    // cout << "subFacesMatrix COL 1 pixel 1: " << subFacesMatrix.col(0).at<float>(0) << endl;
    // cout << "subFacesMatrix COL 1 pixel 2: " << subFacesMatrix.col(0).at<float>(1) << endl;
    // cout << "subFacesMatrix COL 1 pixel 9998: " << subFacesMatrix.col(0).at<float>(9998) << endl;
    // cout << "subFacesMatrix COL 1 pixel 9999: " << subFacesMatrix.col(0).at<float>(9999) << endl;
    // cout << "subFacesMatrix COL 2 pixel 1: " << subFacesMatrix.col(1).at<float>(0) << endl;
    // cout << "subFacesMatrix COL 2 pixel 2: " << subFacesMatrix.col(1).at<float>(1) << endl;
    // cout << "subFacesMatrix COL 2 pixel 9998: " << subFacesMatrix.col(1).at<float>(9998) << endl;
    // cout << "subFacesMatrix COL 2 pixel 9999: " << subFacesMatrix.col(1).at<float>(9999) << endl;
    // cout << "avgVector col 1: " << avgVector.col(0) << endl;

    // ****************************************  Covariance matrix  Done  ******************************
    cout << "covarianceMatrix size: " << covarianceMatrix.size() << endl;
    // cout << "covarianceMatrix COL 1 pixel 1: " << covarianceMatrix.col(0).at<float>(0) << endl;
    // cout << "covarianceMatrix COL 1 pixel 2: " << covarianceMatrix.col(0).at<float>(1) << endl;
    // cout << "covarianceMatrix COL 1 pixel 48: " << covarianceMatrix.col(0).at<float>(48) << endl;
    // cout << "covarianceMatrix COL 1 pixel 49: " << covarianceMatrix.col(0).at<float>(49) << endl;
    // cout << "covarianceMatrix COL 2 pixel 1: " << covarianceMatrix.col(1).at<float>(0) << endl;
    // cout << "covarianceMatrix COL 2 pixel 2: " << covarianceMatrix.col(1).at<float>(1) << endl;
    // cout << "covarianceMatrix COL 2 pixel 48: " << covarianceMatrix.col(1).at<float>(48) << endl;
    // cout << "covarianceMatrix COL 2 pixel 49: " << covarianceMatrix.col(1).at<float>(49) << endl;
    // cout << "covarianceMatrix col 1: " << covarianceMatrix.col(0) << endl;

    // ****************************************  allEigenValues Done ******************************
    cout << "allEigenValues COL 1 pixel 1: " << allEigenValues.col(0).at<float>(0) << endl;    
    // cout << "allEigenValues size: " << allEigenValues.size() << endl;

    // ****************************************  EigenVector Step 1 DONE ******************************
    cout << "allEigenVectors size: " << allEigenVectors.size() << endl;

    // cout << "allEigenVectors COL 1 pixel 1: " << allEigenVectors.col(0).at<float>(0) << endl;
    // cout << "allEigenVectors COL 1 pixel 2: " << allEigenVectors.col(0).at<float>(1) << endl;
    // cout << "allEigenVectors COL 1 pixel 3: " << allEigenVectors.col(0).at<float>(2) << endl;
    // cout << "allEigenVectors COL 1 pixel 48: " << allEigenVectors.col(0).at<float>(48) << endl;
    // cout << "allEigenVectors COL 1 pixel 49: " << allEigenVectors.col(0).at<float>(49) << endl;
    // cout << "allEigenVectors COL 2 pixel 1: " << allEigenVectors.col(1).at<float>(0) << endl;
    // cout << "allEigenVectors COL 2 pixel 2: " << allEigenVectors.col(1).at<float>(1) << endl;
    // cout << "allEigenVectors COL 1 pixel 3: " << allEigenVectors.col(1).at<float>(2) << endl;
    // cout << "allEigenVectors COL 2 pixel 48: " << allEigenVectors.col(1).at<float>(48) << endl;
    // cout << "allEigenVectors COL 2 pixel 49: " << allEigenVectors.col(1).at<float>(49) << endl;
    // // cout << "allEigenVectors col 1: " << allEigenVectors.col(0) << endl;

    // ****************************************  EigenVector Step 2  DONE ******************************
    cout << "eigenVector 2 size: " << eigenVector.size() << endl;
    // cout << "eigenVector COL 1 pixel 1: " << eigenVector.col(0).at<float>(0) << endl;
    // cout << "eigenVector COL 1 pixel 2: " << eigenVector.col(0).at<float>(1) << endl;
    // cout << "eigenVector COL 1 pixel 48: " << eigenVector.col(0).at<float>(48) << endl;
    // cout << "eigenVector COL 1 pixel 49: " << eigenVector.col(0).at<float>(49) << endl;
    // cout << "eigenVector COL 2 pixel 1: " << eigenVector.col(1).at<float>(0) << endl;
    // cout << "eigenVector COL 2 pixel 2: " << eigenVector.col(1).at<float>(1) << endl;
    // cout << "eigenVector COL 2 pixel 48: " << eigenVector.col(1).at<float>(48) << endl;
    // cout << "eigenVector COL 2 pixel 49: " << eigenVector.col(1).at<float>(49) << endl;
    // cout << "eigenVector col 1: " << eigenVector.col(0) << endl;


    // ****************************************  weights   ******************************
    cout << "weights  size: " << weights.size() << endl;
    cout << "weights COL 1 pixel 1: " << weights.col(0).at<float>(0) << endl;
    cout << "weights COL 1 pixel 2: " << weights.col(0).at<float>(1) << endl;
    cout << "weights COL 1 pixel 48: " << weights.col(0).at<float>(48) << endl;
    cout << "weights COL 1 pixel 49: " << weights.col(0).at<float>(49) << endl;
    cout << "weights COL 2 pixel 1: " << weights.col(1).at<float>(0) << endl;
    cout << "weights COL 2 pixel 2: " << weights.col(1).at<float>(1) << endl;
    cout << "weights COL 2 pixel 48: " << weights.col(1).at<float>(48) << endl;
    cout << "weights COL 2 pixel 49: " << weights.col(1).at<float>(49) << endl;
    // cout << "weights col 1: " << weights.col(0) << endl;

    // cout << "EigenVector size: " << eigenVector.size() << endl;
    // cout << "weights size: " << weights.size() << endl;
    // cout << weights.at<float>(0, 0) << endl;

    // Mat test_image = read_test_image( "s01_12.jpg");
    // // namedWindow("test_image", WINDOW_NORMAL);
    // // imshow("test_image", test_image);
    // // waitKey(0);

    // Mat test_image_vector = test_image_to_vector( test_image);

    // // print test_image_vector size
    // cout << "test_image_vector size: " << test_image_vector.size() << endl;

    // cout << "avgVector size: " << avgVector.size() << endl;    

    // // cout<< (int)test_image_vector.at<uchar>(0,0) <<endl ;
    // // cout<< (int)test_image_vector.at<uchar>(1,0) <<endl ;

    // // cout<< (int)avgVector.at<float>(0,0) <<endl ;
    // // cout<< (int)avgVector.at<float>(1,0) <<endl ;
    // // subtract test_image_vector from avgVector
    // Mat test_sub_avgVector = avgVector.clone();
    // for (int i = 0; i< avgVector.cols; i++)
    // {
    //     test_sub_avgVector.at<float>(i,0) = test_image_vector.at<uchar>(i,0) - avgVector.at<float>(i,0);
    // }
    // cout << "test_sub_avgVector size: " << test_sub_avgVector.size() << endl; 
    // cout << "eigenVector size: " << eigenVector.size() << endl; 
    // cout << "test_sub_avgVector Transpose size: " << test_sub_avgVector.t().size() << endl;

    // // print data types of eigenvctor and test
    // cout << "EigenVector type: " << eigenVector.type() << endl;
    // cout << "test_sub_avgVector type: " << test_sub_avgVector.type() << endl;

    // Mat test_weight =  eigenVector* test_sub_avgVector;
    // // Mat test_weight = eigenVector.dot(test_sub_avgVector)  ;

    // cout << "test_weight size: " << test_weight.size() << endl;
    // cout << "weights size: " << weights.size() << endl;

    // // double distance = norm(test_weight, weights, NORM_L2);
    // // cout << "distance size: " << distance.size() << endl;
    // // cout << distance << endl;
    // // weights - test weights 
    // cout << "Before NORM" << endl;
    // vector<double> Norm_Vector ;
    // cout << "Weight COLS size: " << weights.col(1).size <<endl;
    // cout << "Weight Rows size: " << weights.row(1).size <<endl;
    // cout << "test_weight size: " << test_weight.size() << endl;

    // // Mat new_weights ;
    // // new_weights.row(1).reshape(0, 50).copyTo(new_weights);
    // // cout << "New Weight Rows size: " << new_weights.size <<endl;
    

    // for(int i=0; i< 50 ; i++)
    // {
    // //     new_weights =weights.clone();
    // //     new_weights.row(i).reshape(0, 50).copyTo(new_weights);

    // //     double dist = norm(new_weights, test_weight, NORM_L2);
    
    //     double dist = norm(weights.col(i), test_weight, NORM_L2);
    //     // double dist = norm(weights.col(i), test_weight, NORM_L2);
    //     Norm_Vector.push_back(dist);
    // }
    // // print test_weight
    // // cout << "test_weight size: " << test_weight.size() << endl;
    // // cout << test_weight << endl;

    // cout << "AFTER NORM" << endl;
    // cout << weights.col(30)<<endl;

    // cout<< "New COL" << endl;

    // cout << weights.col(49)<<endl;


    // // Mat diff = weights.clone();
    // // for (int r = 0 ; r< weights.rows; r++ )
    // // {
    // //     for (int c = 0 ; c< weights.cols; c++ )
    // //     {
    // //         diff.at<float>(r,c) = weights.at<float>(r,c) - test_weight.at<float>(r,0);
    // //     }
    // // }

    // // // Mat diff = weights - test_weight;
    // // cout << "diff size: " << diff.size() << endl;   
    // // cout << diff.at<float>(0, 0) << endl;
    // // // calculate norm vector
    // // vector<float> Norm_Vector ;
    // // float norm;
    // // for (int r = 0 ; r< diff.rows; r++ )
    // // {
    // //     norm = 0;
    // //     for (int c = 0 ; c< diff.cols; c++ )
    // //     {
    // //         norm += pow(diff.at<float>(r, c), 2);
    // //     }
    // //     norm = sqrt(norm);
    // //     Norm_Vector.push_back(norm);
    // // }

    // // print norm size
    // cout << "Norm_Vector size: " << Norm_Vector.size() << endl;
    // // print norm vector

    // for (int i = 0; i < Norm_Vector.size(); i++)
    // {
    //     cout << Norm_Vector[i] << endl;
    // }
    // auto it = min_element(Norm_Vector.begin(), Norm_Vector.end());
    // int index = distance(Norm_Vector.begin(), it);
    // cout << "INDEX = " << index <<endl;


    // // cout << "diff: " << diff << endl;

    // // cout << weights.at<float>(0, 0) << endl;
    // // cout << weights.at<float>(0, 1) << endl;
    // // cout << weights.at<float>(1, 0) << endl;

    // // cout << test_weight.at<float>(0, 0) << endl;
    // // cout << test_weight.at<float>(1, 0) << endl;

    // // cout<< (int)test_image_vector.at<uchar>(0,0) <<endl ;
    // // cout<< (int)avgVector.at<float>(0,0) <<endl ;
    // // cout<< (int)test_sub_avgVector.at<float>(0,0);

    // // cout << test_weight.at<float>(0, 0) << endl;

    // // cout << weights.at<float>(0, 0) << endl;

    return 0;
}
