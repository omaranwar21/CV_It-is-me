# Recognize faces Application

## Table of contents:
- [Introduction](#introduction)
- [Project Features](#project-features)
- [Project Structure](#project-structure)
- [Quick Preview](#quick-preview)
- [Requirements to run](#Requirements-to-run)
- [Run the project](#Run-the-project)
- [Try a demo](#Try-a-demo)
- [Team]()


### Introduction
Our application uses C++ language that focuses on face detection and recognition using PCA/Eigen analysis. This project aims to provide robust and efficient methods for detecting and recognizing faces in color and grayscale images. Our application leverages the power of C++ to deliver high-performance algorithms and accurate results.

### Project Features
In this project:
- [x] Face Detection:
  - Our application employs advanced techniques to detect faces in images. It can analyze both color and grayscale images, making it versatile 
    and adaptable to various scenarios. The face detection algorithm efficiently identifies facial features, enabling accurate localization of 
    faces within the images.
- [x] Face Recognition with PCA/Eigen Analysis:
  - Once the faces are detected, our application utilizes PCA/Eigen analysis for face recognition. This method extracts the essential facial 
    features and represents them in a lower-dimensional space. By comparing these features, the application can recognize and match faces with a 
    high degree of accuracy.
- [x] Performance Reporting:
  - Our application provides comprehensive performance reports, allowing users to evaluate the effectiveness of the face detection and 
    recognition algorithms. The reports include metrics such as precision, providing insights into the application's performance.
- [x] ROC Curve Plotting:
  - In addition to performance metrics, our application generates Receiver Operating Characteristic (ROC) curves. These curves plot the true 
    positive rate against the false positive rate at various thresholds, providing a visual representation of the algorithm's performance. ROC 
    curves are widely used in evaluating the effectiveness of face recognition systems.



### Project Structure
The Application is built using:
- C++/Opencv:
  - Opencv 14/15/16 versions

- QT framework:
  - QT 6.4.2 version

```
├─ GUI
│  ├─  Data/Test
│  ├─  PCA
│  ├─  ReadWrite
│  ├─  Recognition
│  ├─  TrainedData
│  ├─  TrainedModel
│  ├─  BackgroundImage
│  ├─  ImagesLists
│  ├─  Icons
│  ├─  Common
│  ├─  main
│  └─  mainwindow
├─ Model_Training
│  ├─  Detection
│  ├─  PCA_Reduced
│  ├─  ReadWrite_Files
│  ├─  Recognition
│  ├─  TrainedData
│  ├─  image_lists
│  ├─  images_script
│  ├─  main_PCA
│  └─  Common
README.md
```

### Quick Preview

#### Application.
![app](https://github.com/omaranwar21/CV_It-is-me/assets/94166833/935b3ea3-98a0-4b93-a00a-3a10b973f597)
#### Face Detection.
![Face Detection](https://github.com/omaranwar21/CV_It-is-me/assets/94166833/ab0a9c45-f256-404b-8ca2-b4ceac5458c9)
#### Offline Face Detection and Recognition.
![offline detection   recognition](https://github.com/omaranwar21/CV_It-is-me/assets/94166833/5db74e4e-b3d3-41cb-bcc6-518965d5539e)
#### Live Face Detection and Recognition.
![live recognition](https://github.com/omaranwar21/CV_It-is-me/assets/94166833/a0d888fe-ca68-4736-a850-6f3b9a3c3ed9)

### Requirements to run 

[ Qt Setup and openCV ](https://github.com/Dinahussam/Impro-App/files/10972282/Qt.Setup.and.openCV.pdf)


### Run the project


### Try a demo

[ Download Here !]


### Team

Second Semester - Biomedical Computer Vision (SBE3230) class project created by:

| Team Members' Names                                  | Section | B.N. |
|------------------------------------------------------|:-------:|:----:|
| [Dina Hussam](https://github.com/Dinahussam)         |    1    |  28  |
| [Omar Ahmed ](https://github.com/omaranwar21)        |    2    |  2   |
| [Omar saad ](https://github.com/Omar-Saad-ELGharbawy)|    2    |  3   |
| [Mohamed Ahmed](https://github.com/MohamedAIsmail)   |    2    |  16  |
| [Neveen Mohamed](https://github.com/NeveenMohamed)   |    2    |  49  |
