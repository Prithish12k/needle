#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>
#include <algorithm>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE

    auto matmul = [](const float *A, const float *B, float* M, size_t r, size_t k, size_t c)
    {
        std::fill(M, M + (r*c), 0.0f);

        for (size_t i {}; i < r; i++)
        {
            for (size_t l {}; l < k; l++)
            {
                for (size_t j {}; j < c; j++)
                {
                    size_t ij = i*c + j;
                    size_t il = i*k + l;
                    size_t lj = l*c + j;

                    M[ij] += A[il]*B[lj];
                }
            }
        }
    };

    auto transpose = [](const float *A, float *M, size_t r, size_t c)
    {
        for (size_t i {}; i < r; i++)
        {
            for (size_t j {}; j < c; j++)
            {
                size_t ij = i*c + j;
                size_t ji = j*r + i;

                M[ji] = A[ij];
            }
        }
    };
/*
    auto EWiseMul = [](const float *A, const float* B, float *M, size_t n)
    {
        for (size_t i {}; i < n; i++)
        {
            M[i] = A[i]*B[i];
        }
    };

    auto relu = [](const float *A, float *M, size_t n)
    {
        for (size_t i {}; i < n; i++)
        {
            M[i] = (A[i] > 0) ? A[i] : 0.0f;
        }
    };

    auto D_relu = [](const float *A, float *M, size_t n)
    {
        for (size_t i {}; i < n; i++)
        {
            M[i] = (A[i] > 0) ? 1.0f : 0.0f;
        }
    };
*/
    auto softmax = [](const float *A, float *M, size_t r, size_t c)
    {
        for (size_t i {}; i < r; i++)
        {
            size_t row_offset {i*c};

            float max_val = A[row_offset];

            for (size_t j {}; j < c; j++)
            {
                max_val = std::max(A[row_offset + j], max_val);
            }

            float sum_val {0.0f};

            for (size_t j {}; j < c; j++)
            {
                float e = std::exp(A[row_offset + j] - max_val);
                M[row_offset + j] = e;
                sum_val += e;
            }

            for (size_t j{}; j < c; j++)
            {
                M[row_offset + j] /= sum_val;
            }
        }
    };

    auto one_hot = [](const unsigned char *y, float *Y, size_t batch, size_t k)
    {
        for (size_t i {}; i < batch; i++)
        {
            for (size_t j {}; j < k; j++)
            {
                size_t ij = i*k + j;
                Y[ij] = (j == y[i]) ? 1.0f : 0.0f;
            }
        }
    };

    float *X_batch = new float[batch*n];
    float *X_batch_T = new float[n*batch];
    unsigned char *y_batch = new unsigned char[batch];
    float *logits = new float[batch*k];
    float *S_batch = new float[batch*k];
    float *Y_batch = new float[batch*k] {};
    float *rate = new float[n*k];

    for (size_t i {}; i < m; i += batch)
    {
        if ((i + batch) > m) break;

        for (size_t j {n*i}; j < n*(i + batch); j++)
        {
            X_batch[j - n*i] = X[j]; 
        }

        transpose(X_batch, X_batch_T, batch, n);

        for (size_t j {i}; j < i + batch; j++)
        {
            y_batch[j - i] = y[j];
        }

        matmul(X_batch, theta, logits, batch, n, k);
        softmax(logits, S_batch, batch, k);
        one_hot(y_batch, Y_batch, batch, k);

        for (size_t j {}; j < batch*k; j++)
        {
            S_batch[j] -= Y_batch[j];
        }

        matmul(X_batch_T, S_batch, rate, n, batch, k);

        for (size_t j {}; j < n*k; j++)
        {
            theta[j] -= (lr/batch)*rate[j];
        }
    }

    delete[] X_batch;
    delete[] X_batch_T;
    delete[] y_batch;
    delete[] logits;
    delete[] S_batch;
    delete[] Y_batch;
    delete[] rate;
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
