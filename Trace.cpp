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
    // cout << "allEigenValues size: " << allEigenValues.size() << endl;
    // cout << "allEigenValues COL 1 pixel 1: " << allEigenValues.col(0).at<float>(0) << endl;    

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


    // ****************************************  weights Done  ******************************
    cout << "weights  size: " << weights.size() << endl;
    // cout << "weights COL 1 pixel 1: " << weights.col(0).at<float>(0) << endl;
    // cout << "weights COL 1 pixel 2: " << weights.col(0).at<float>(1) << endl;
    // cout << "weights COL 1 pixel 48: " << weights.col(0).at<float>(48) << endl;
    // cout << "weights COL 1 pixel 49: " << weights.col(0).at<float>(49) << endl;
    // cout << "weights COL 2 pixel 1: " << weights.col(1).at<float>(0) << endl;
    // cout << "weights COL 2 pixel 2: " << weights.col(1).at<float>(1) << endl;
    // cout << "weights COL 2 pixel 48: " << weights.col(1).at<float>(48) << endl;
    // cout << "weights COL 2 pixel 49: " << weights.col(1).at<float>(49) << endl;
    // cout << "weights col 1: " << weights.col(0) << endl;

    /* 3 - TEST Vectors */
    cout << "*********************************************************STEP 3*******************" << endl ;

    Mat test_image = read_test_image( "s05_14.jpg");
    // // namedWindow("test_image", WINDOW_NORMAL);
    // // imshow("test_image", test_image);
    // // waitKey(0);
    Mat test_image_vector = test_image_to_vector( test_image);

    // ****************************************  test_image_vector  DONE ******************************
    cout << "test_image_vector  size: " << test_image_vector.size() << endl;
    // cout << "test_image_vector COL 1 pixel 1: " << test_image_vector.col(0).at<float>(0) << endl;
    // cout << "test_image_vector COL 1 pixel 2: " << test_image_vector.col(0).at<float>(1) << endl;
    // cout << "test_image_vector COL 1 pixel 9998: " << test_image_vector.col(0).at<float>(9998) << endl;
    // cout << "test_image_vector COL 1 pixel 9999: " << test_image_vector.col(0).at<float>(9999) << endl;

    // ****************************************  normalized_test_img DONE   ******************************
    Mat normalized_test_img = normalize_test_img(test_image_vector , avgVector );
    
    cout << "normalized_test_img  size: " << normalized_test_img.size() << endl;
    // cout << "normalized_test_img COL 1 pixel 1: " << normalized_test_img.col(0).at<float>(0) << endl;
    // cout << "normalized_test_img COL 1 pixel 2: " << normalized_test_img.col(0).at<float>(1) << endl;
    // cout << "normalized_test_img COL 1 pixel 9998: " << normalized_test_img.col(0).at<float>(9998) << endl;
    // cout << "normalized_test_img COL 1 pixel 9999: " << normalized_test_img.col(0).at<float>(9999) << endl;


    cout << "test_sub_avgVector Transpose size: " << normalized_test_img.t().size() << endl;
    cout << "eigenVector Transpose size: " << eigenVector.t().size() << endl;

    // ****************************************  test_weight Done  ******************************
    Mat test_weight =  normalized_test_img.t() * eigenVector.t();

    cout << "test_weight  size: " << test_weight.size() << endl;
    // cout << "test_weight ROW 1 pixel 1: " << test_weight.row(0).at<float>(0) << endl;
    // cout << "test_weight ROW 1 pixel 2: " << test_weight.row(0).at<float>(1) << endl;
    // cout << "test_weight ROW 1 pixel 3: " << test_weight.row(0).at<float>(2) << endl;
    // cout << "test_weight ROW 1 pixel 48: " << test_weight.row(0).at<float>(48) << endl;
    // cout << "test_weight ROW 1 pixel 49: " << test_weight.row(0).at<float>(49) << endl;

    /* 4 - Recognition  */
    cout << "*************STEP 4 Recognition*******************" << endl ;

    // ****************************************  Eucledien distance  DONE ******************************

    vector<double> eucledien_distance ;

    for(int i=0; i< weights.rows ; i++)
    {
        double dist = norm(weights.row(i), test_weight, NORM_L2);
        eucledien_distance.push_back(dist);
    }

    // cout << "eucledien_distance pixel 1: " << eucledien_distance[0] << endl;
    // cout << "eucledien_distance pixel 2: " << eucledien_distance[1] << endl;
    // cout << "eucledien_distance pixel 3: " << eucledien_distance[2] << endl;
    // cout << "eucledien_distance pixel 48: " << eucledien_distance[48] << endl;
    // cout << "eucledien_distance pixel 49: " << eucledien_distance[49] << endl;

    // ****************************************  INDEX   ******************************

    auto it = min_element(eucledien_distance.begin(), eucledien_distance.end());
    int index = distance(eucledien_distance.begin(), it);
    cout << "INDEX = " << index <<endl;

    return 0;
}
