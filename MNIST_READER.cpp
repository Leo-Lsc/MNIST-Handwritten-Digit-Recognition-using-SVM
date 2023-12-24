#include <iostream>
#include <fstream>
#include <vector>

std::vector<std::vector<unsigned char>> readImages(const std::string& file);

std::vector<unsigned char> readLabels(const std::string& file);

/* Function to read images from file and return a 2-dimensional matrix storing image data. */
std::vector<std::vector<unsigned char>> readImages(const std::string& file){
    std::ifstream f(file, std::ios::binary);

    char head[16];
    f.read(head, 16);

    int numOfImages = (static_cast<unsigned char>(head[4]) << 24) |
                      (static_cast<unsigned char>(head[5]) << 16) |
                      (static_cast<unsigned char>(head[6]) << 8) |
                      static_cast<unsigned char>(head[7]);

    int numOfRows = (static_cast<unsigned char>(head[8]) << 24) |
                    (static_cast<unsigned char>(head[9]) << 16) |
                    (static_cast<unsigned char>(head[10]) << 8) |
                    static_cast<unsigned char>(head[11]);

    int numOfColumns = (static_cast<unsigned char>(head[12]) << 24) |
                       (static_cast<unsigned char>(head[13]) << 16) |
                       (static_cast<unsigned char>(head[14]) << 8) |
                       static_cast<unsigned char>(head[15]);

    std::vector<std::vector<unsigned char>> images (numOfImages, std::vector<unsigned char>(numOfRows * numOfColumns));

    for (int i = 0; i < numOfImages; i++) {
        f.read(reinterpret_cast<char*>(images[i].data()), numOfRows * numOfColumns);
    }

    f.close();
    return images;
}

/* Function to read labels from file and return a vector storing the labels. */
std::vector<unsigned char> readLabels(const std::string& file) {
    std::ifstream f(file, std::ios::binary);

    char head[8];
    f.read(head, 8);

    int numOfLabels = (static_cast<unsigned char>(head[4]) << 24) |
                      (static_cast<unsigned char>(head[5]) << 16) |
                      (static_cast<unsigned char>(head[6]) << 8) |
                      static_cast<unsigned char>(head[7]);

    std::vector<unsigned char> labels(numOfLabels);

    for (int i = 0; i < numOfLabels; i++) {
        f.read(reinterpret_cast<char*>(&labels[i]), 1);
    }

    f.close();
    return labels;
}
