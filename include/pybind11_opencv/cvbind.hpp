// Created  by ausk<jincsu#126.com> @ 2019.11.23
// Based on
// https://github.com/ausk/keras-unet-deploy/tree/master/cpp/libunet/cvbind.h

#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// Convert cv::Point, cv::Rest, cv::Mat
namespace pybind11
{
namespace detail
{
//! cv::Point <=> tuple(x,y)
template <>
struct type_caster<cv::Point>
{
    PYBIND11_TYPE_CASTER(cv::Point, _("tuple_xy"));

    // Convert from Python to C++.
    // Convert the Python tuple object to C++ cv::Point type, and return false
    // if the conversion fails.
    // The second argument indicates whether implicit conversions should be
    // applied.
    bool load(handle obj, bool)
    {
        // Ensure that the passed parameter is of tuple type
        if (!pybind11::isinstance<pybind11::tuple>(obj))
        {
            std::logic_error("Point(x,y) should be a tuple!");
            return false;
        }

        // Extract the tuple object from the handle and ensure its length is 2.
        pybind11::tuple pt = reinterpret_borrow<pybind11::tuple>(obj);
        if (pt.size() != 2)
        {
            std::logic_error("Point(x,y) tuple should be size of 2");
            return false;
        }

        // Convert a tuple of length 2 to cv::Point.
        value = cv::Point(pt[0].cast<int>(), pt[1].cast<int>());
        return true;
    }

    // Convert from C++ to Python.  Convert C++ cv::Mat object to tuple,
    // parameter 2 and parameter 3 are ignored
    static handle cast(const cv::Point& pt, return_value_policy, handle)
    {
        return pybind11::make_tuple(pt.x, pt.y).release();
    }
};

// cv::Rect <=> tuple(x,y,w,h)
template <>
struct type_caster<cv::Rect>
{
    PYBIND11_TYPE_CASTER(cv::Rect, _("tuple_xywh"));

    bool load(handle obj, bool)
    {
        if (!pybind11::isinstance<pybind11::tuple>(obj))
        {
            std::logic_error("Rect should be a tuple!");
            return false;
        }

        pybind11::tuple rect = reinterpret_borrow<pybind11::tuple>(obj);
        if (rect.size() != 4)
        {
            std::logic_error("Rect (x,y,w,h) tuple should be size of 4");
            return false;
        }

        value = cv::Rect(rect[0].cast<int>(),
                         rect[1].cast<int>(),
                         rect[2].cast<int>(),
                         rect[3].cast<int>());
        return true;
    }

    static handle cast(const cv::Rect& rect, return_value_policy, handle)
    {
        return pybind11::make_tuple(rect.x, rect.y, rect.width, rect.height)
            .release();
    }
};

// Convert between cv::Mat and numpy.ndarray.
//
// Python supports a general buffer protocol for data exchange between plugins.
// Let the type expose a buffer view, this allows direct access to the original
// internal data, often used in matrix types.
//
// Pybind11 provides the pybind11::buffer_info type to map the Python buffer
// protocol (buffer protocol).
//
// struct buffer_info {
//    void* ptr;                      /* Pointer to buffer */
//    ssize_t itemsize;               /* Size of one scalar */
//    std::string format;             /* Python struct-style format descriptor
//    */ ssize_t ndim;                /* Number of dimensions */
//    std::vector<ssize_t> shape;     /* Buffer dimensions */
//    std::vector<ssize_t> strides;   /* Strides (in bytes) for each index */
//};

template <>
struct type_caster<cv::Mat>
{
public:
    PYBIND11_TYPE_CASTER(cv::Mat, _("numpy.ndarray"));

    //! 1. cast numpy.ndarray to cv::Mat
    bool load(handle obj, bool)
    {
        array b = reinterpret_borrow<array>(obj);
        buffer_info info = b.request();

        int nh = 1;
        int nw = 1;
        int nc = 1;
        int ndims = info.ndim;
        if (ndims == 2)
        {
            nh = info.shape[0];
            nw = info.shape[1];
        }
        else if (ndims == 3)
        {
            nh = info.shape[0];
            nw = info.shape[1];
            nc = info.shape[2];
        }
        else
        {
            throw std::logic_error("Only support 2d, 2d matrix");
            return false;
        }

        int dtype;
        if (info.format == format_descriptor<unsigned char>::format())
        {
            dtype = CV_8UC(nc);
        }
        else if (info.format == format_descriptor<int>::format())
        {
            dtype = CV_32SC(nc);
        }
        else if (info.format == format_descriptor<float>::format())
        {
            dtype = CV_32FC(nc);
        }
        else
        {
            throw std::logic_error(
                "Unsupported type, only support uchar, int32, float");
            return false;
        }
        value = cv::Mat(nh, nw, dtype, info.ptr);
        return true;
    }

    //! Cast cv::Mat to numpy.ndarray
    static handle cast(const cv::Mat& mat,
                       return_value_policy,
                       handle /*defval*/)
    {
        std::string format = format_descriptor<unsigned char>::format();
        size_t elemsize = sizeof(unsigned char);
        int nw = mat.cols;
        int nh = mat.rows;
        int nc = mat.channels();
        int depth = mat.depth();
        int type = mat.type();
        int dim = (depth == type) ? 2 : 3;
        if (depth == CV_8U)
        {
            format = format_descriptor<unsigned char>::format();
            elemsize = sizeof(unsigned char);
        }
        else if (depth == CV_32S)
        {
            format = format_descriptor<int>::format();
            elemsize = sizeof(int);
        }
        else if (depth == CV_32F)
        {
            format = format_descriptor<float>::format();
            elemsize = sizeof(float);
        }
        else
        {
            throw std::logic_error(
                "Unsupport type, only support uchar, int32, float");
        }

        std::vector<size_t> bufferdim;
        std::vector<size_t> strides;
        if (dim == 2)
        {
            bufferdim = {(size_t)nh, (size_t)nw};
            strides = {elemsize * (size_t)nw, elemsize};
        }
        else if (dim == 3)
        {
            bufferdim = {(size_t)nh, (size_t)nw, (size_t)nc};
            strides = {(size_t)elemsize * nw * nc,
                       (size_t)elemsize * nc,
                       (size_t)elemsize};
        }
        return array(buffer_info(
                         mat.data, elemsize, format, dim, bufferdim, strides))
            .release();
    }
};
}  // namespace detail
}  // namespace pybind11
