#include "dirent.h"
#include "onnx_model.h"

/**
 * @brief Get all the image filenames in a specified directory
 * @param img_dir: the input directory
 * @param img_names: the vector storing all the image filenames
 */
void getAllImageFiles(const std::string &img_dir,
                      std::vector<std::string> &img_names) {
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(img_dir.c_str())) != NULL) {
       while ((ent = readdir(dir)) != NULL) {
          std::string filename(ent->d_name);
          if (filename == "." || filename == "..") continue;
          img_names.push_back(filename);
        }
        closedir(dir);
      } else {
        // Failed to open directory
        perror("");
        exit(EXIT_FAILURE);
    }
}

// pretty prints a shape dimension vector
// std::string print_shape(const std::vector<int64_t>& v) {
//   std::stringstream ss("");
//   for (size_t i = 0; i < v.size() - 1; i++)
//     ss << v[i] << "x";
//   ss << v[v.size() - 1];
//   return ss.str();
// }


// int calculate_product(const std::vector<int64_t>& v) {
//     int total = 1;
//     for (auto& i : v) total *= i;
//     return total;
// }

// Create a tensor from the input image

int main(int argc, char **argv) {
    // Create image classifier
    std::string model_file = "../model/net.onnx";
    Network net(model_file);

    // Load images in the input directory
    std::string img_dir("../genoray_samples/");
    std::string dst_dir ("../genoray_results/");

    std::vector<std::string> img_names;
    getAllImageFiles(img_dir, img_names);

    for (int i = 0; i < img_names.size(); i++) {
        std::cout << img_names[i] << std::endl;
        // Load an input image
        cv::Mat image = cv::imread(img_dir + img_names[i], cv::IMREAD_UNCHANGED);

        auto input_shape = net.getInputShapes()[0];
        int total_number_elements = calculate_product(input_shape);
        std::vector<float> input_arr(total_number_elements);
        net.create_tensor_from_image(image, input_arr);

        float* out_arr;
        net.Inference(input_arr, &out_arr);

        cv::Mat out_img = cv::Mat(image.rows, image.cols, CV_32FC1, out_arr);
        cv2:imwrite(dst_dir + img_names[i], out_img);

        image.release();
        out_img.release();
    }

    return 0;
}
