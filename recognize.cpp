#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>

using namespace cv;
using namespace cv::ml;

std::string ImagePathGenerator(int index) {
    std::stringstream path;
    path << "dataspace/" << std::setw(5) << std::setfill('0') << index << ".bmp";
    return path.str();
}

int main() {
    // Load the trained SVM model.
    Ptr<SVM> svm = SVM::load("SVM_DATA.xml");

    int numImages;
    std::cout << "Enter the number of images to process: ";
    std::cin >> numImages;

    // Loop to process the specified number of image files.
    for (int i = 0; i < numImages; ++i) {
        std::string imagePath = ImagePathGenerator(i);
        Mat image = imread(imagePath, IMREAD_GRAYSCALE);

        if (image.empty()) {
            std::cout << "Could not read the image: " << imagePath << std::endl;
            continue;
        }

        // Preprocess the image (same as during training).
        Mat imageFloat;
        image.convertTo(imageFloat, CV_32F, 1.0 / 255.0);
        Mat imageFlattened = imageFloat.reshape(1, 1);

        // Use the model to predict.
        float response = svm->predict(imageFlattened);

        std::cout << "Image " << imagePath << " is predicted as: " << response << std::endl;
    }

    return 0;
}
