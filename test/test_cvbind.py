#!/usr/bin/env python3
"""Test bindings of some basic functions using OpenCV types."""
import pathlib
import tempfile
import unittest

import numpy as np
import cv2

from pybind11_opencv import cvbind_test


class TestCvBind(unittest.TestCase):

    test_img_path = str(pathlib.Path(__file__).parent / "testimg.png")

    def test_addpt(self):
        p1 = (1, 2)
        p2 = (10, 20)

        res = cvbind_test.addpt(p1, p2)

        self.assertEqual(res, (11, 22))

    def test_addrect(self):
        # rect is a tuple (x, y, width, height)
        r1 = (10, 20, 20, 10)  # lt(10, 20) => rb(30, 30)
        r2 = (5, 5, 20, 20)  # lt(5, 5)   => rb(25, 25)
        res = cvbind_test.addrect(r1, r2)  # lt(5, 5), rb(30, 30) => wh(25, 25)

        self.assertEqual(res, (5, 5, 25, 25))

    def _test_addmat(self, dtype):
        m1 = np.arange(9, dtype=dtype).reshape(3, 3)
        m2 = np.array(
            [[10, 100, 1000], [20, 200, 2000], [30, 300, 3000]], dtype=dtype
        )

        res = cvbind_test.addmat(m1, m2)

        self.assertEqual(res.dtype, dtype)
        np.testing.assert_array_equal(res, m1 + m2)

    def test_addmat_float32(self):
        self._test_addmat(np.float32)

    def test_addmat_int32(self):
        self._test_addmat(np.int32)

    def test_addmat_uint8(self):
        self._test_addmat(np.uint8)

    def test_imread(self):
        expected = cv2.imread(self.test_img_path)
        actual = cvbind_test.imread(self.test_img_path)

        np.testing.assert_array_equal(actual, expected)

    def test_imwrite(self):
        orig_img = cv2.imread(self.test_img_path)

        with tempfile.NamedTemporaryFile(suffix=".png") as temp_file:
            # write the image with the bound function then use cv2 to load it
            # again
            cvbind_test.imwrite(temp_file.name, orig_img)
            written_img = cv2.imread(temp_file.name)

        np.testing.assert_array_equal(written_img, orig_img)


if __name__ == "__main__":
    unittest.main()
