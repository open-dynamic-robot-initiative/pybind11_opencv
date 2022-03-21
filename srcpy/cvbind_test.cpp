// Created  by ausk<jincsu#126.com> @ 2019.11.23
// Modified by ausk<jincsu#126.com> @ 2020.03.01
// Based on
// https://github.com/ausk/keras-unet-deploy/tree/master/cpp/libunet/cvbind.cpp

#include <pybind11_opencv/cvbind.hpp>

cv::Point addpt(cv::Point& lhs, cv::Point& rhs)
{
    return cv::Point(lhs.x + rhs.x, lhs.y + rhs.y);
}

cv::Rect addrect(cv::Rect& lhs, cv::Rect& rhs)
{
    int x = std::min(lhs.x, rhs.x);
    int y = std::min(lhs.y, rhs.y);
    int width = std::max(lhs.x + lhs.width - 1, rhs.x + rhs.width - 1) - x + 1;
    int height =
        std::max(lhs.y + lhs.height - 1, rhs.y + rhs.height - 1) - y + 1;
    return cv::Rect(x, y, width, height);
}

cv::Mat addmat(cv::Mat& lhs, cv::Mat& rhs)
{
    return lhs + rhs;
}

cv::Mat imread(std::string fpath)
{
    return cv::imread(fpath);
}

void imwrite(std::string fpath, const cv::Mat& img)
{
    cv::imwrite(fpath, img);
}

PYBIND11_MODULE(cvbind_test, m)
{
    m.def("addpt", &addpt, "add two point");
    m.def("addrect", &addrect, "add two rect");
    m.def("addmat", &addmat, "add two matrix");
    m.def("imread", &imread, "read the file into np.ndarray/cv::Mat");
    m.def("imwrite", &imwrite, "write np.ndarray/cv::Mat into the file");
}
