#include "onnx_model.h"

using namespace std;

// pretty prints a shape dimension vector
std::string print_shape(const std::vector<int64_t>& v) {
    std::stringstream ss("");
    for (size_t i = 0; i < v.size() - 1; i++)
        ss << v[i] << "x";
    ss << v[v.size() - 1];
    return ss.str();
}

int calculate_product(const std::vector<int64_t>& v) {
    int total = 1;
    for (auto& i : v) total *= i;
    return total;
}


// Constructor
Network::Network(const std::string& modelFilepath) {
    /**************** Create ORT environment ******************/
    std::string instanceName{"Network inference"};
    env = Ort::Env(
        OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
        instanceName.c_str()
    );

    /**************** Create ORT session ******************/
    // Set up options for session
    Ort::SessionOptions session_options;
    // Enable CUDA
    session_options.AppendExecutionProvider_CUDA(OrtCUDAProviderOptions{});
    // Sets graph optimization level (Here, enable all possible optimizations)
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    std::string model_file = modelFilepath;
    // Create session by loading the onnx model
    std::cout << "Loading model from " << model_file << std::endl;
    // Ort::Experimental::Session mSession = Ort::Experimental::Session(env, model_file, session_options);
    // mSession = shared_ptr<Ort::Experimental::Session>(env, model_file, session_options);
    mSession = new Ort::Experimental::Session(env, model_file, session_options);

    // Get the name of the input
    // 0 means the first input of the model
    // The example only has one input, so use 0 here
    input_names = mSession->GetInputNames();
    input_shapes = mSession->GetInputShapes();

#ifdef VERBOSE
    for (size_t i = 0; i < input_names.size(); i++) {
        cout << "\t" << input_names[i] << " : " << print_shape(input_shapes[i]) << endl;
    }
#endif

    // Get the name of the output
    // 0 means the first output of the model
    // The example only has one output, so use 0 here
    output_names = mSession->GetOutputNames();
    output_shapes = mSession->GetOutputShapes();

#ifdef VERBOSE
    cout << "Output Node Name/Shape (" << output_names.size() << "):" << endl;
    for (size_t i = 0; i < output_names.size(); i++) {
        cout << "\t" << output_names[i] << " : " << print_shape(output_shapes[i]) << endl;
    }
#endif

    assert(input_names.size() == 1 && output_names.size() == 1);

    // Inputs
    std::vector<int64_t> mInputDims;

    // Outputs
    char* mOutputName;
    std::vector<int64_t> mOutputDims;


    // Get the type of the input
    // 0 means the first input of the model
    auto inputTypeInfo = mSession->GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
#ifdef VERBOSE
    std::cout << "Input Type: " << inputType << std::endl;
#endif

  /**************** Input info ******************/
  // Get the number of input nodes
  size_t numInputNodes = mSession->GetInputCount();
#ifdef VERBOSE
  std::cout << "******* Model information below *******" << std::endl;
  std::cout << "Number of Input Nodes: " << numInputNodes << std::endl;
#endif

  // Get the shape of the input
  mInputDims = inputTensorInfo.GetShape();
#ifdef VERBOSE
  std::cout << "Input Dimensions: " << mInputDims << std::endl;
#endif
}

Network::~Network() {
    delete mSession;
}

std::vector<std::vector<int64_t>> Network::getInputShapes() {
    return input_shapes;
}

std::vector<std::vector<int64_t>> Network::getOutputShapes() {
    return output_shapes;
}


void Network::create_tensor_from_image(
    /**************** Preprocessing ******************/
    // Create input tensor (including size and value) from the loaded input image
    const cv::Mat& img, std::vector<float>& inputTensorValues) {

    cv::Mat preprocessedImage;

    /******* Preprocessing *******/
    cv::dnn::blobFromImage(img, preprocessedImage);

    // Assign the input image to the input tensor
    inputTensorValues.assign(
        preprocessedImage.begin<float>(),
        preprocessedImage.end<float>()
    );
}


// Perform inference for a given image
int Network::Inference(std::vector<float> input_arr, float** out_arr) {
#ifdef TIME_PROFILE
    const auto before = clock_time::now();
#endif

    std::vector<Ort::Value> input_tensors;
    auto input_shape = input_shapes[0];
    input_tensors.push_back(
        Ort::Experimental::Value::CreateTensor<float>(input_arr.data(), input_arr.size(), input_shape)
    );

    // double-check the dimensions of the input tensor
    assert(input_tensors[0].IsTensor() &&
         input_tensors[0].GetTensorTypeAndShapeInfo().GetShape() == input_shape
    );


    std::vector<Ort::Value> output_tensors;
    std::cout << "\ninput_tensor shape: " << print_shape(input_tensors[0].GetTensorTypeAndShapeInfo().GetShape()) << std::endl;

    std::cout << "Running model..." << std::endl;
    try {
        output_tensors = mSession->Run(mSession->GetInputNames(), input_tensors, mSession->GetOutputNames());
        cout << "done" << endl;

        // double-check the dimensions of the output tensors
        // NOTE: the number of output tensors is equal to the number of output nodes specifed in the Run() call
        assert(output_tensors.size() == session.GetOutputNames().size() &&
           output_tensors.front().IsTensor());
        cout << "output_tensor_shape: " << print_shape(output_tensors.front().GetTensorTypeAndShapeInfo().GetShape()) << endl;

    } catch (const Ort::Exception& exception) {
        cout << "ERROR running model inference: " << exception.what() << endl;
        exit(-1);
    }
#ifdef TIME_PROFILE
    const sec duration1 = clock_time::now() - before1;
    std::cout << "The inference takes " << duration1.count() << "s" << std::endl;
#endif

    /**************** Postprocessing the output result ******************/
    cout << "Postprocessing" << endl;
#ifdef TIME_PROFILE
    const auto before2 = clock_time::now();
#endif
    // Get the inference result
    *out_arr = output_tensors.front().GetTensorMutableData<float>();


#ifdef TIME_PROFILE
    const sec duration2 = clock_time::now() - before2;
    std::cout << "The postprocessing takes " << duration2.count() << "s"
              << std::endl;
#endif
}