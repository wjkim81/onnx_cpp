#ifndef ONNX_MODEL_H_
#define ONNX_MODEL_H_

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <algorithm>  // std::generate
#include <assert.h>
#include <sstream>

// Header for onnxruntime
// #include <onnxruntime_cxx_api.h>
#include <experimental_onnxruntime_cxx_api.h>

#define VERBOSE
//#define TIME_PROFILE

#ifdef TIME_PROFILE
using clock_time = std::chrono::system_clock;
using sec = std::chrono::duration<double>;
#endif

/**
 * @brief Compute the product over all the elements of a vector
 * @tparam T
 * @param v: input vector
 * @return the product
 */
template <typename T>
size_t vectorProduct(const std::vector<T>& v) {
    return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

// pretty prints a shape dimension vector
// std::string print_shape(const std::vector<int64_t>& v) {
//     std::stringstream ss("");
//     for (size_t i = 0; i < v.size() - 1; i++)
//         ss << v[i] << "x";
//     ss << v[v.size() - 1];
//     return ss.str();
// }

#ifdef VERBOSE
/**
 * @brief Operator overloading for printing vectors
 * @tparam T
 * @param os
 * @param v
 * @return std::ostream&
 */
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) {
  os << "[";
  for (int i = 0; i < v.size(); ++i) {
    os << v[i];
    if (i != v.size() - 1) {
      os << ", ";
    }
  }
  os << "]";
  return os;
}
#endif


std::string print_shape(const std::vector<int64_t>&);
int calculate_product(const std::vector<int64_t>&);


class Network {
    public:
        /**
         * @brief Constructor
         * @param modelFilepath: path to the .onnx file
         */
        Network(const std::string&);
        ~Network();

        /**
         * @brief Perform inference on a single image
         * @param imageFilepath: path to the image
         * @return the index of the predicted class
         */
        int Inference(const std::vector<float>, float**);
        std::vector<std::vector<int64_t>> getInputShapes();
        std::vector<std::vector<int64_t>> getOutputShapes();
        /**
         * @brief Create a tensor from an input image
         * @param img: the input image
         * @param inputTensorValues: the output tensor
         */
        void create_tensor_from_image(const cv::Mat&, std::vector<float>&);
                                  
    private:
        // ORT Environment
        Ort::Env env;

        // Session
        // std::shared_ptr<Ort::Experimental::Session> mSession;
        // Ort::Experimental::Session *session;
        Ort::Experimental::Session *mSession;

        // Inputs
        std::vector<std::string> input_names;
        std::vector<std::vector<int64_t>> input_shapes;

        // Outputs
        std::vector<std::string> output_names;
        std::vector<std::vector<int64_t>> output_shapes;


        
};

#endif  // IMAGE_ENHANCEMENT_H_
