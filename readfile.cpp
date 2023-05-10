#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include <sstream>
#include <string>

#include <fstream>
// #include <unistd.h>


using namespace std;
using namespace cv;
//read training list
void readList(string& listFilePath, vector<string>& facesPath)
{
    ifstream file(listFilePath.c_str(), ifstream::in);
    
    if (!file) {
        cout << "Fail to open file: " << listFilePath << endl;
        exit(0);
    }
    
    string line, path, id;
    while (getline(file, line)) {
        stringstream lines(line);
        // getline(lines, id, ';');
        getline(lines, path);
        
        path.erase(remove(path.begin(), path.end(), '\r'), path.end());
        path.erase(remove(path.begin(), path.end(), '\n'), path.end());
        path.erase(remove(path.begin(), path.end(), ' '), path.end());
        
        facesPath.push_back(path);
        //facesID.push_back(atoi(id.c_str()));
        // facesID.push_back(id);
    }
}

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    string trainListFilePath = "E:\\SBME 6th Term\\Computer Vision\\Projects & Tasks\\CV Final Project\\Repo\\CV_It-is-me\\images_list.txt";
    vector<string> trainFacesPath;
    vector<string> trainFacesID;
    // vector<string> loadedFacesID;
    //read training list and ID from txt file
    readList(trainListFilePath, trainFacesPath);

    // print trainFacesPath 
    for (int i = 0; i < trainFacesPath.size(); i++)
    {
        cout << trainFacesPath[i] << endl;
    }

    return 0;

}