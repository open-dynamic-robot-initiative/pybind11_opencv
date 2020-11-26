pybind11_opencv
===============

Adds pybind11 support for basic OpenCV types:

- cv::Mat <=> np.array (for types float32, int32, uint8)
- cv::Point <=> tuple (x, y)
- cv::Rect <=> tuple (x, y, width, height)

This is based on
https://github.com/ausk/keras-unet-deploy/blob/master/cpp/libunet/cvbind.{h,cpp}
(commit 83d3f85).


Usage
-----

Simply add `#include <pybind11_opencv/cvbind.hpp>` in your code to enable the
conversion of the types listed above.
