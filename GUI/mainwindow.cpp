#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    ui->liveView->setScene(new QGraphicsScene(this));

    loadTrainedModel();
    ReadPCAData();

}

MainWindow::~MainWindow()
{
    delete ui;
}



void MainWindow::on_UploadDetectionImage_clicked()
{
    uploadImage(inputImage, inputDetectionImageMat, detectionPath);

    if(inputImage.isNull()) return;

    updateImage(inputDetectionImageMat, ui->DetectionInputImage, 1);

}

void MainWindow::on_DetectButton_clicked()
{
    if(inputImage.isNull()) return;

    if(checkBox){
        faces_detection(inputDetectionImageMat, outputDetectionImageMat);
    }
    else{
        outputDetectionImageMat = inputDetectionImageMat.clone();
        cv::resize(outputDetectionImageMat, outputDetectionImageMat, Size(512, 512));
        cv::putText(outputDetectionImageMat, predictFaces(inputDetectionImageMat), cv::Point(outputDetectionImageMat.rows / 3, 100), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(0, 255, 0), 3);
    }

    updateImage(outputDetectionImageMat, ui->DetectionOutputImage, 1);

    outputDetectionImageMat = Mat::zeros(1, 1, CV_64F);

}

/*
 * ***********************************************************
 *          Change tabs using next & previous button
 * ***********************************************************
*/
void MainWindow::on_nextButton_clicked()
{
    if(tabIndex + 1 > ui->stackedWidget->count() - 1){
        return;
    }
    else{
        tabIndex++;
        ui->stackedWidget->setCurrentIndex(tabIndex);
    }
}
void MainWindow::on_backButton_clicked()
{
    if(tabIndex - 1 < 0){
        return;
    }
    else{
        tabIndex--;
        ui->stackedWidget->setCurrentIndex(tabIndex);
    }
}

/*
 * ************************************************************************
 *          Open live camera function for detection and recognition
 * ************************************************************************
*/
void MainWindow::on_openCameraButton_clicked()
{
    if(video.isOpened())
        {
            ui->openCameraButton->setIcon(QIcon(":/Icons/Icons/camera.png"));
            video.release();

            ui->liveView->scene()->removeItem(&pixmap);
            ui->liveView->scene()->update();
            return;
        }

    if(!video.open(0))
    {
        QMessageBox::critical(this,
                              "Camera Error",
                              "Make sure you entered a correct camera index,"
                              "<br>or that the camera is not being accessed by another program!");
        return;
    }

    ui->liveView->scene()->addItem(&pixmap);
    ui->openCameraButton->setIcon(QIcon(":/Icons/Icons/stop-button.png"));

    Mat frame;

    while(video.isOpened())
    {

       video >> frame;

       if(!frame.empty())
       {
           faces_detection(frame, frame);

           QImage qimg(frame.data,
                       frame.cols,
                       frame.rows,
                       frame.step,
                       QImage::Format_RGB888);

           pixmap.setPixmap( QPixmap::fromImage(qimg.rgbSwapped()) );
           ui->liveView->fitInView(&pixmap, Qt::KeepAspectRatio);

           timer.setInterval(10000);
       }
       qApp->processEvents();
    }


    ui->openCameraButton->setIcon(QIcon(":/Icons/Icons/camera.png"));
}
void MainWindow::closeEvent(QCloseEvent* event)
{
    if (video.isOpened())
    {
        QMessageBox::warning(this,
            "Warning",
            "Stop the video before closing the application!");
        event->ignore();
    }
    else
    {
        event->accept();
    }
}

/*
 * ************************************************************************
 *          Upload Image and Update the UI QLabel
 * ************************************************************************
*/
void MainWindow::uploadImage(QImage &image, Mat &imageMat, QString &imgPath)
{
    imgPath = QFileDialog::getOpenFileName(this,tr("Open image"));

    reader.setFileName(imgPath);
    image = reader.read();

    image = image.convertToFormat(QImage::Format_BGR888);
    imageMat = Mat(image.height(), image.width(), CV_8UC3, image.bits(), image.bytesPerLine());

    if(image.isNull()) return;
}
void MainWindow::updateImage(Mat &inputMat,  QLabel* image, bool rgb_flag){

    Mat clonedMat = inputMat.clone();
    cv::resize(clonedMat, clonedMat, Size(512,512));
    if(rgb_flag){
        image->setPixmap(QPixmap::fromImage(QImage(clonedMat.data, clonedMat.cols, clonedMat.rows, clonedMat.step, QImage::Format_BGR888)));
    }
    else{
        image->setPixmap(QPixmap::fromImage(QImage(clonedMat.data, clonedMat.cols, clonedMat.rows, clonedMat.step, QImage::Format_Grayscale8)));
    }
}

/*
 * ************************************************************************
 *          Predict faces of the input matrix using PCA Model
 * ************************************************************************
*/
string MainWindow::predictFaces(Mat &inputMat){

    Mat resizedInputMat = inputMat.clone();
    cv::resize(resizedInputMat, resizedInputMat, Size(100,100));

    Mat testWeights = project_image(resizedInputMat, dataMat[0], dataMat[1]);
    vector<double> min_indexes = recognize_face(dataMat[2], testWeights);

    // map
    map<string, int> myMap;
    for (int i = 0; i < min_indexes.size(); i++)
    {
        string label = string_split(trainImagesNames[min_indexes[i]]);
        myMap[label]++;
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
    return max_key;
}
void MainWindow::faces_detection(Mat &image, Mat &resultImage){

    std::vector<cv::Rect> faces;
    resultImage = image.clone();
    face_cascade.detectMultiScale(resultImage, faces, 1.1, 2, 0|cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));

    for (const auto face : faces) {

            cv::rectangle(resultImage, face, cv::Scalar(0, 255, 0), 2);

            // add text on the frame
            Mat faceROI = resultImage(face).clone();
            cv::resize(faceROI, faceROI, Size(100,100));
            cv::putText(resultImage, predictFaces(faceROI), cv::Point(face.x, face.y - 5), cv::FONT_HERSHEY_PLAIN, 2.5, cv::Scalar(0, 255, 0), 3);
     }
}
void MainWindow::loadTrainedModel(){
    // Read trained model
    QDir dir(QDir::currentPath());
    dir.cdUp();
    QString absolutePath = dir.absoluteFilePath("CV_It-is-me/TrainedModel/haarcascade_frontalface_alt.xml");
    face_cascade.load(absolutePath.toStdString());
}
void MainWindow::ReadPCAData(){
    ReadWrite obj("");
    dataMat = obj.readData();
    trainImagesNames = obj.readList(":/ImagesLists/team_train.txt");
}
void MainWindow::on_faceDetectionCheckBox_stateChanged(int arg1)
{
    checkBox = arg1;

    qDebug() << QString::number(checkBox);
}

