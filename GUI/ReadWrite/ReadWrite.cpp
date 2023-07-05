#include "ReadWrite/ReadWrite.hpp"


ReadWrite::ReadWrite(string write_folder_path){
    this->noOfRows = 0;
    this->write_folder_path = write_folder_path;
}

vector<string> ReadWrite::readList(QString listFilePath){
    std::vector<std::string> facesPath;

    QFile file(listFilePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        qDebug() << "Failed to open file:" << listFilePath;
        return facesPath;
    }

    QTextStream in(&file);
    while (!in.atEnd())
    {
        QString line = in.readLine();
        QString path = line.trimmed();

        std::string pathStr = path.toStdString();
        pathStr.erase(std::remove(pathStr.begin(), pathStr.end(), '\r'), pathStr.end());
        pathStr.erase(std::remove(pathStr.begin(), pathStr.end(), '\n'), pathStr.end());

        facesPath.push_back(pathStr);
    }

    file.close();

    return facesPath;
}

vector<Mat> ReadWrite::readImages(string folder_path, vector<string> trainFacesPath){
    vector<Mat> images;
    for (const auto &filename : trainFacesPath)
    {
        string file_path = folder_path + "\\" + filename;
        Mat img = imread(file_path);
        resize(img, img, Size(100, 100));
        // convert to gray
        cvtColor(img, img, COLOR_BGR2GRAY);
        images.push_back(img);
    }
    this->noOfRows = (int)images.size();
//    cout << "Number of images: " << images.size() << endl;
    return images;
}

void ReadWrite::writeData(Mat avgVector, Mat eigenVector, Mat weights){
    writeMean(avgVector);
    writeEigenVectors(eigenVector);
    writeWeights(weights);
}

void ReadWrite::writeMean(Mat avg)
{
    string meanPath = this -> write_folder_path  + "//" + "mean.txt";
    ofstream writeMeanFile(meanPath.c_str(), ofstream::out | ofstream::trunc);
    if (!writeMeanFile) {
        cout << "Fail to open file: " << meanPath << endl;
    }
    
    writeMeanFile << this->noOfRows;
    writeMeanFile << " ";
    for (int i = 0; i < avg.rows; i++) {
        writeMeanFile << avg.at<float>(i);
        writeMeanFile << " ";
    }
    
    writeMeanFile.close();
}

void ReadWrite::writeEigenVectors(Mat eigen)
{
    string eigenPath = this->write_folder_path  + "//" + "eigen.txt";
    ofstream writeEigenFile(eigenPath.c_str(), ofstream::out | ofstream::trunc);
    if (!writeEigenFile) {
        cout << "Fail to open file: " << eigenPath << endl;
    }
    
    for (int i = 0; i < eigen.rows; i++) {
        for (int j = 0; j < eigen.cols; j++) {
            writeEigenFile << eigen.row(i).at<float>(j);
            writeEigenFile << " ";
        }
        writeEigenFile << "\n";
    }
    
    writeEigenFile.close();
}

void ReadWrite::writeWeights(Mat weights)
{
    string weightsPath = this->write_folder_path  + "//" + "weights.txt";
    ofstream writeWeightsFile(weightsPath.c_str(), ofstream::out | ofstream::trunc);
    if (!writeWeightsFile) {
        cout << "Fail to open file: " << weightsPath << endl;
    }
    
    for (int i = 0; i < weights.rows; i++) {
        for (int j = 0; j < weights.cols; j++) {
            writeWeightsFile << weights.row(i).at<float>(j);
            writeWeightsFile << " ";
        }
        writeWeightsFile << "\n";
    }
    
    writeWeightsFile.close();
}

vector<Mat> ReadWrite::readData(){
    vector<Mat> data(3);
    data[0] = readMean();
    data[1] = readEigen();
    data[2]= readWeights();
    return data;
}

Mat ReadWrite::readMean()
{
    cv::Mat mean = cv::Mat::zeros(10000, 1, CV_32FC1);

    QString filePath = ":/TrainedData/mean.txt";
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        qDebug() << "Failed to open file:" << filePath;
        return mean;
    }

    QTextStream in(&file);
    QString line = in.readLine();
    QStringList dataList = line.split(' ');
    this->noOfRows = (int)(dataList.value(0).toFloat());

    for (int i = 1; i <= mean.rows; i++)
    {
        float value = dataList.value(i).toFloat();
        mean.at<float>(i - 1, 0) = value;
    }

    file.close();

    return mean;
}

Mat ReadWrite::readEigen()
{
    cv::Mat eigen = cv::Mat::zeros(this->noOfRows, 10000, CV_32FC1);
    QString filePath = ":/TrainedData/eigen.txt";
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        qDebug() << "Failed to open file:" << filePath;
        return eigen;
    }

    QTextStream in(&file);
    int i = 0;
    while (!in.atEnd() && i < noOfRows)
    {
        QString line = in.readLine();
        QStringList dataList = line.split(' ');

        for (int j = 0; j < eigen.cols; j++)
        {
            float value = dataList.value(j).toFloat();
            eigen.at<float>(i, j) = value;
        }

        i++;
    }

    file.close();

    return eigen;
}

Mat ReadWrite::readWeights()
{
    int noOfCols = this->noOfRows;

    cv::Mat weights = cv::Mat::zeros(this->noOfRows, noOfCols, CV_32FC1);
    QString filePath = ":/TrainedData/weights.txt";
    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        qDebug() << "Failed to open file:" << filePath;
        return weights;
    }

    QTextStream in(&file);
    int i = 0;
    while (!in.atEnd() && i < noOfRows)
    {
        QString line = in.readLine();
        QStringList dataList = line.split(' ');

        for (int j = 0; j < noOfCols; j++)
        {
            float value = dataList.value(j).toFloat();
            weights.at<float>(i, j) = value;
        }

        i++;
    }

    file.close();

    return weights;
}

