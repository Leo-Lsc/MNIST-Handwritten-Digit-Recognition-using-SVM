#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include "MNIST_READER.cpp"

using namespace cv;
using namespace cv::ml;

void train(Mat& trainingImagesMat, Mat& trainingLabelsMat, Mat& testImagesMat, Mat& testLabelsMat);

void train(Mat& trainingImages, Mat& trainingLabels, Mat& testImages, Mat& testLabels) {
    Ptr<SVM> svm = SVM::create();

    int C = 10; // Value of penlaty parameter C.
    double gamma = 0.09; // Value of gamma.

    // Set SVM parameters.
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::RBF);
    svm->setC(C);
    svm->setGamma(gamma);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 500, FLT_EPSILON));

    std::cout << "Training parameters:" << std::endl;
    std::cout << "C(Penlaty parameter) = " << C << std::endl;
    std::cout << "Gamma = " << gamma << std::endl;

    svm->train(trainingImages, ROW_SAMPLE, trainingLabels);

    Mat predictedLabels;
    svm->predict(testImages, predictedLabels);

    int correctPredictions = 0;
    for (int i = 0; i < testLabels.rows; i++) {
        if (predictedLabels.at<float>(i, 0) == testLabels.at<int>(i, 0)) {
            correctPredictions++;
        }
    }
    double accuracy = static_cast<double>(correctPredictions) / testLabels.rows;
    std::cout << "The accuracy of the model is: " << accuracy << std::endl;

    // Save the model
    svm->save("SVM_DATA.xml");
}

int main() {
    std::cout << "Loading training and test data..." << std::endl;

    // Load training data.
    std::vector<std::vector<unsigned char>> trainImages = readImages("train-images.idx3-ubyte");
    std::vector<unsigned char> trainLabels = readLabels("train-labels.idx1-ubyte");

    // Prepare training data.
    Mat trainingImagesMat(trainImages.size(), trainImages[0].size(), CV_32F);
    Mat trainingLabelsMat(trainLabels.size(), 1, CV_32S);

    for (size_t i = 0; i < trainImages.size(); ++i) {
        for (size_t j = 0; j < trainImages[i].size(); ++j) {
            trainingImagesMat.at<float>(i, j) = static_cast<float>(trainImages[i][j]) / 255.0f;
        }
        trainingLabelsMat.at<int>(i, 0) = static_cast<int>(trainLabels[i]);
    }

    // Load test data.
    std::vector<std::vector<unsigned char>> testImages = readImages("t10k-images.idx3-ubyte");
    std::vector<unsigned char> testLabels = readLabels("t10k-labels.idx1-ubyte");

    // Prepare test data.
    Mat testImagesMat(testImages.size(), testImages[0].size(), CV_32F);
    Mat testLabelsMat(testLabels.size(), 1, CV_32S);

    for (size_t i = 0; i < testImages.size(); ++i) {
        for (size_t j = 0; j < testImages[i].size(); ++j) {
            testImagesMat.at<float>(i, j) = static_cast<float>(testImages[i][j]) / 255.0f;
        }
        testLabelsMat.at<int>(i, 0) = static_cast<int>(testLabels[i]);
    }

    std::cout << "Start training, this process may take several minutes..." << std::endl;
    train(trainingImagesMat, trainingLabelsMat, testImagesMat, testLabelsMat);
    std::cout << "Trainning completed." << std::endl;

    return 0;
}
