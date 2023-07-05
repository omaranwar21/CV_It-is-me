QT       += core gui
#QT       += multimedia
QT       += xml

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17


INCLUDEPATH += C:\Users\Anwar\Desktop\OpenCv\opencv\release\install\include

LIBS += C:\Users\Anwar\Desktop\OpenCv\opencv\release\bin\libopencv_core470.dll
LIBS += C:\Users\Anwar\Desktop\OpenCv\opencv\release\bin\libopencv_highgui470.dll
LIBS += C:\Users\Anwar\Desktop\OpenCv\opencv\release\bin\libopencv_imgcodecs470.dll
LIBS += C:\Users\Anwar\Desktop\OpenCv\opencv\release\bin\libopencv_imgproc470.dll
LIBS += C:\Users\Anwar\Desktop\OpenCv\opencv\release\bin\libopencv_calib3d470.dll
LIBS += C:\Users\Anwar\Desktop\OpenCv\opencv\release\bin\libopencv_calib3d470.dll
LIBS += C:\Users\Anwar\Desktop\OpenCv\opencv\release\bin\libopencv_videoio470.dll
LIBS += C:\Users\Anwar\Desktop\OpenCv\opencv\release\bin\libopencv_features2d470.dll
LIBS += C:\Users\Anwar\Desktop\OpenCv\opencv\release\bin\libopencv_objdetect470.dll

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
#    PCA/PCA_Original/PCA_original.cpp \
#    PCA/PCA_Reduced/PCA_Reduced.cpp \
#    PCA/main_PCA.cpp \
    ReadWrite/ReadWrite.cpp \
#    ReadWrite/ReadWrite.cpp \
#    Recgonition/Recognize.cpp \
    Recgonition/Recognize.cpp \
#    main.cpp \
    main.cpp \
#    mainwindow.cpp \
    mainwindow.cpp

HEADERS += \
    Common/common.hpp \
#    Common/common.hpp \
#    PCA/PCA_Reduced/PCA_Reduced.hpp \
    PCA/PCA_Reduced/PCA_Reduced.hpp \
#    ReadWrite/ReadWrite.hpp \
    ReadWrite/ReadWrite.hpp \
#    Recgonition/Recognize.hpp \
    Recgonition/Recognize.hpp \
#    mainwindow.h \
    mainwindow.h

FORMS += \
#    mainwindow.ui \
    mainwindow.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

RESOURCES += \
    Background.qrc \
    Data.qrc \
    Icons.qrc \
    ImageLists.qrc \
    TrainedData.qrc \
    TrainedModel.qrc

DISTFILES += \
    TrainedModel/haarcascade_frontalface_alt.xml
