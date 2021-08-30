// Handle linking to VPI libraries
#include <vpi/Array.h>
#include <vpi/Image.h>
#include <vpi/ImageFormat.h>
#include <vpi/Pyramid.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/CUDAInterop.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/OpticalFlowDense.h>
#define CHECK_STATUS(STMT)                                    \
    do                                                        \
    {                                                         \
        VPIStatus status = (STMT);                            \
        if (status != VPI_SUCCESS)                            \
        {                                                     \
            char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH];       \
            vpiGetLastStatusMessage(buffer, sizeof(buffer));  \
            std::ostringstream ss;                            \
            ss << vpiStatusGetName(status) << ": " << buffer; \
            throw std::runtime_error(ss.str());               \
        }                                                     \
    } while (0);

// Handle linking to torch libraries
#include <torch/torch.h>
#include <torch/extension.h>
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


// VPI metadata
VPIStream __stream;
VPIOpticalFlowQuality __qualityNative;
VPIImageFormat __formatNative;
VPIPixelType __pixelType;
VPIImagePlane __prevPlanes[VPI_MAX_PLANE_COUNT];
VPIImagePlane __currPlanes[VPI_MAX_PLANE_COUNT];
VPIImagePlane __movePlanes[VPI_MAX_PLANE_COUNT];

// VPI images and payloads for dense optical flow
VPIImage __prevInter, __currInter;
VPIImage __prevCUDA, __currCUDA, __prevBL, __currBL;
VPIImage __motionBL;
VPIImage __prev, __curr;
VPIPayload __payload;


// Torch metadata for constructing result
auto __torchOpts = torch::TensorOptions()
    .dtype(torch::kInt16)
    .layout(torch::kStrided)
    .device(torch::kCUDA, 0)
    .requires_grad(false);

// Transfer data from a torch Tensor to the VPI
VPIImageData __transfer(torch::Tensor* x, VPIImagePlane* planeData) {
    // Hard limit the amount of planes going in to the VPI-set limit of 6
    size_t planeCount;
    if (x->size(0) > 6) planeCount = 6;
    else planeCount = x->size(0);
    
    // Transfer each frame
    for (int i = 0; i < x->size(0); i++) {
        auto plane = &(planeData[i]);

        plane->data = x->data_ptr();
        plane->height = x->size(1);
        plane->width = x->size(2);
        plane->pitchBytes = 1; // TODO: I feel like I'm wrong here
        plane->pixelType = __pixelType;
    }

    // Construct result for return
    auto result = VPIImageData();
    result.format = __formatNative;
    result.numPlanes = (int)x->size(0);
    memccpy(result.planes, planeData, planeCount, sizeof(VPIImagePlane));

    return result;
};


// Take a set of tensors from torch, detach, and perform dense optical flow on them
torch::Tensor denseFlow(torch::Tensor prevImg, torch::Tensor currImg, 
        std::string quality = "high", std::string format = "rgb",
        bool upscale = false) {
    /// Precheck ///
    // The correct backend is being used
    CHECK_INPUT(prevImg);
    CHECK_INPUT(currImg);

    // Extract the native quality setting from the input string
    if (quality == "low") __qualityNative = VPI_OPTICAL_FLOW_QUALITY_LOW;
    else if (quality == "med") __qualityNative = VPI_OPTICAL_FLOW_QUALITY_MEDIUM;
    else if (quality == "high") __qualityNative = VPI_OPTICAL_FLOW_QUALITY_HIGH;
    else TORCH_CHECK(false, quality, " is not implemented in the VPI backend.");

    // Extract the format from the input string
    if (format == "nv12") __formatNative = VPI_IMAGE_FORMAT_NV12_ER;
    else if (format == "rgb") __formatNative = VPI_IMAGE_FORMAT_RGB8;
    else if (format == "bgr") __formatNative = VPI_IMAGE_FORMAT_BGR8;
    else TORCH_CHECK(false, format, " is not implemented for conversion from torch.");


    /// Transfer Contexts ///
    __pixelType = (__formatNative == VPI_IMAGE_FORMAT_NV12_ER) ? VPI_PIXEL_TYPE_S16 : VPI_PIXEL_TYPE_3U8;

    // Start up stream to VPI
    if (__stream == nullptr)
        CHECK_STATUS(vpiStreamCreate(VPI_BACKEND_CUDA | VPI_BACKEND_NVENC | VPI_BACKEND_VIC, &__stream));

    // Pull out of gradient from Torch
    auto prevRaw = prevImg.detach();
    auto prevDat = __transfer(&prevRaw, __prevPlanes);
    auto currRaw = currImg.detach();
    auto currDat = __transfer(&currRaw, __currPlanes);

    // Make VPI compatible without leaving CUDA context
    CHECK_STATUS(vpiImageCreateCUDAMemWrapper(&prevDat, VPI_BACKEND_CUDA, &__prevCUDA));
    CHECK_STATUS(vpiImageCreateCUDAMemWrapper(&currDat, VPI_BACKEND_CUDA, &__currCUDA));

    // Get ready for the VIC
    if (__formatNative != VPI_IMAGE_FORMAT_NV12_ER) {
        // Set up the computation
        CHECK_STATUS(vpiImageCreate(prevRaw.size(1), prevRaw.size(2), 
            VPI_IMAGE_FORMAT_NV12_ER, VPI_BACKEND_CUDA, &__prev));
        CHECK_STATUS(vpiImageCreate(currRaw.size(1), currRaw.size(2),
            VPI_IMAGE_FORMAT_NV12_ER, VPI_BACKEND_CUDA, &__curr));
        
        // Change the format of the images
        CHECK_STATUS(vpiSubmitConvertImageFormat(__stream, VPI_BACKEND_CUDA, __prevCUDA, __prev, nullptr));
        CHECK_STATUS(vpiSubmitConvertImageFormat(__stream, VPI_BACKEND_CUDA, __currCUDA, __curr, nullptr));
    }
    else {
        // Move images over
        __prev = __prevCUDA;
        __curr = __currCUDA;
    }
    
    // Handle final run-up to NVENC
    CHECK_STATUS(vpiImageCreate(prevRaw.size(1), prevRaw.size(2),
        VPI_IMAGE_FORMAT_NV12_ER_BL, VPI_BACKEND_VIC, &__prevInter));
    CHECK_STATUS(vpiImageCreate(currRaw.size(1), currRaw.size(2),
        VPI_IMAGE_FORMAT_NV12_ER_BL, VPI_BACKEND_VIC, &__currInter));

    // Convert image to block linear for processing
    CHECK_STATUS(vpiSubmitConvertImageFormat(__stream, VPI_BACKEND_VIC, __prev, __prevInter, nullptr));
    CHECK_STATUS(vpiSubmitConvertImageFormat(__stream, VPI_BACKEND_VIC, __curr, __currInter, nullptr));


    /// Compute ///
    // Create motion vector
    auto resultRaw = torch::zeros({prevRaw.size(0), prevRaw.size(1) / 4, prevRaw.size(2) / 4, 2}, __torchOpts);
    auto resultDat = __transfer(&resultRaw, __movePlanes);
    CHECK_STATUS(vpiImageCreateCUDAMemWrapper(&resultDat, VPI_BACKEND_CUDA, &__motionBL));

    // Set up and compute dense optical flow
    CHECK_STATUS(vpiCreateOpticalFlowDense(VPI_BACKEND_NVENC, 
        prevRaw.size(1), prevRaw.size(2), VPI_IMAGE_FORMAT_NV12_ER_BL, 
        __qualityNative, &__payload));
    CHECK_STATUS(vpiSubmitOpticalFlowDense(__stream, VPI_BACKEND_NVENC, __payload, 
        __prevInter, __currInter, __motionBL));


    /// Return ///
    // Convert back to torch and return
    CHECK_STATUS(vpiStreamSync(__stream));
    auto result = resultRaw.to(torch::kFloat16, true, false).div(1 << 5);

    // Clean up
    vpiStreamDestroy(__stream);
    vpiPayloadDestroy(__payload);

    vpiImageDestroy(__prevInter);
    vpiImageDestroy(__currInter);
    vpiImageDestroy(__prevCUDA);
    vpiImageDestroy(__currCUDA);
    vpiImageDestroy(__prevBL);
    vpiImageDestroy(__currBL);
    vpiImageDestroy(__motionBL);
    vpiImageDestroy(__curr);
    vpiImageDestroy(__prev);

    // Done.
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("denseFlow", &denseFlow, "NV VPI Dense Optical Flow");
}
