#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    ui->liveView->setScene(new QGraphicsScene(this));

    // Read trained model
    QDir dir(QDir::currentPath());
    dir.cdUp();
    QString absolutePath = dir.absoluteFilePath("TrainedModel/haarcascade_frontalface_alt.xml");
    face_cascade.load(absolutePath.toStdString());


    ReadWrite obj("");
    dataMat = obj.readData();
    trainImagesNames = obj.readList(":/ImagesLists/team_train.txt");

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
           std::vector<cv::Rect> faces;

           face_cascade.detectMultiScale(frame, faces, 1.1, 2, 0|cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));

           for (const auto& face : faces) {

                   cv::rectangle(frame, face, cv::Scalar(0, 255, 0), 2);
                   // add text on the frame
                   Mat faceROI = frame(face).clone();
                   cv::resize(faceROI, faceROI, Size(100,100));
                   cv::putText(frame, predictFaces(faceROI), cv::Point(face.x, face.y - 5), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(0, 255, 0), 2);
            }

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

void MainWindow::uploadImage(QImage &image, Mat &imageMat, QString &imgPath)
{
    imgPath = QFileDialog::getOpenFileName(this,tr("Open image"));

    reader.setFileName(imgPath);
    image = reader.read();

    image = image.convertToFormat(QImage::Format_BGR888);
    imageMat = Mat(image.height(), image.width(), CV_8UC3, image.bits(), image.bytesPerLine());

    if(image.isNull()) return;
    cv::resize(imageMat, imageMat, cv::Size(512,512), 0, 0);

}

void MainWindow::updateImage(Mat &inputMat,  QLabel* image, bool rgb_flag){

    if(rgb_flag){
        image->setPixmap(QPixmap::fromImage(QImage(inputMat.data, inputMat.cols, inputMat.rows, inputMat.step, QImage::Format_BGR888)));
    }
    else{
        image->setPixmap(QPixmap::fromImage(QImage(inputMat.data, inputMat.cols, inputMat.rows, inputMat.step, QImage::Format_Grayscale8)));
    }
}


string MainWindow::predictFaces(Mat &inputMat){

    Mat testWeights = project_image(inputMat, dataMat[0], dataMat[1]);
    vector<double> min_indexes = recognize_face(dataMat[2], testWeights);

    // map
    map<string, int> myMap;
    for (int i = 0; i < min_indexes.size(); i++)
    {
        string label = string_split(trainImagesNames[min_indexes[i]]);
        myMap[label]++;
    }


    // print my map keya nd values
//    for (auto it = myMap.begin(); it != myMap.end(); ++it)
//    {
//        cout << it->first << " " << it->second << endl;
//    }

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

