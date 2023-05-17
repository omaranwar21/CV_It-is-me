#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <Recgonition/Recognize.hpp>
#include <ReadWrite/ReadWrite.hpp>


using namespace cv;
using namespace std;

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT


public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();



private slots:
    void on_UploadDetectionImage_clicked();

    void on_nextButton_clicked();

    void on_backButton_clicked();

    void on_openCameraButton_clicked();

    void uploadImage(QImage &image, Mat &imageMat, QString &imgPath);

    void updateImage(Mat &inputMat,  QLabel* image, bool rgb_flag);

    void closeEvent(QCloseEvent *event);

    string predictFaces(Mat &inputMat);

    void loadTrainedModel();

    void ReadPCAData();

    void on_DetectButton_clicked();

    void faces_detection(Mat &image, Mat &resultImage);

private:

    Ui::MainWindow *ui;
    QImageReader reader;
    QImage inputImage;
    QPixmap defaultBackground;

    // ----------- Offline Settings ----------
    Mat inputDetectionImageMat = Mat::zeros(1, 1, CV_64F);
    QString detectionPath;
    Mat outputDetectionImageMat = Mat::zeros(1, 1, CV_64F);

    // ----------- Live Settings -------------
    bool startRecording = false;
    QGraphicsPixmapItem pixmap;
    cv::VideoCapture video;
    QTimer timer;

    // ----------- Model Settings ------------
    cv::CascadeClassifier face_cascade;

    // dataMat[0] returns avgVector, dataMat[1] return eigenVectors, dataMat[2] returns weights
    vector<Mat> dataMat;
    vector<string> trainImagesNames;



    int tabIndex = 0; // stackedWidget tab index
;};
#endif // MAINWINDOW_H
