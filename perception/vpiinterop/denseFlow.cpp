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
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define TORCH_EXTENSION_NAME vpiinterop

#include <iostream>

// VPI metadata
VPIStream __stream = nullptr;
VPIOpticalFlowQuality __qualityNative;
VPIImageFormat __formatNative;
VPIPixelType __pixelType;
VPIImagePlane __prevPlanes[VPI_MAX_PLANE_COUNT];
VPIImagePlane __currPlanes[VPI_MAX_PLANE_COUNT];
VPIImagePlane __movePlanes[VPI_MAX_PLANE_COUNT];

// VPI images and payloads for dense optical flow
VPIImage __prevInter, __currInter;
VPIImage __prevCUDA, __currCUDA, __prevBL, __currBL;
VPIImage __motion, __motionBL;
VPIImage __prev, __curr;
VPIPayload __payload;


// Torch metadata for constructing result
auto __torchOpts = torch::TensorOptions()
    .dtype(torch::kInt16)
    .layout(torch::kStrided)
    .device(torch::kCUDA, 0)
    .requires_grad(false);

// Transfer data from a torch Tensor to the VPI
#define LAZYASS_DEBUG true
VPIImageData* __transfer(torch::Tensor* x, VPIImagePlane* planeData, 
        const VPIPixelType* pType=nullptr, const size_t* pitchWidth=nullptr, 
        const VPIImageFormat* iFormat=nullptr) {
    // Hard limit the amount of planes going in to the VPI-set limit of 6
    size_t planeCount;
    if (x->size(0) > 6) planeCount = 6;
    else planeCount = x->size(0);
    
    // Transfer each frame
    for (int i = 0; i < x->size(0); i++) {
        auto plane = &(planeData[i]);

        plane->data = x->data_ptr();
        plane->height = x->size(1);
        if (LAZYASS_DEBUG) std::cout << "Height: " << x->size(1) << std::endl;
        plane->width = x->size(2);
        if (LAZYASS_DEBUG) std::cout << "Width: " << x->size(2) << std::endl;
        plane->pitchBytes = plane->width * ((pitchWidth == nullptr) ? 3 : *pitchWidth);
        plane->pixelType = (pType == nullptr) ? __pixelType : *pType;
    }

    // Construct result for return
    auto result = new VPIImageData();
    result->format = (iFormat == nullptr) ? __formatNative : *iFormat;
    result->numPlanes = (int)x->size(0);
    for (int i = 0; i < result->numPlanes; i++) {
        result->planes[i] = planeData[i];
    }

    return result;
};


// Take a set of tensors from torch, detach, and perform dense optical flow on them
torch::Tensor denseFlow(torch::Tensor prevImg, torch::Tensor currImg, 
        std::string quality = "high", std::string format = "rgb",
        bool upscale = false, bool keepAliveStream = true) {
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
        CHECK_STATUS(vpiStreamCreate(VPI_BACKEND_ALL, &__stream));

    // Pull out of gradient from Torch
    auto prevRaw = prevImg.detach();
    auto prevDat = __transfer(&prevRaw, __prevPlanes);
    auto currRaw = currImg.detach();
    auto currDat = __transfer(&currRaw, __currPlanes);

    // Make VPI compatible without leaving CUDA context
    CHECK_STATUS(vpiImageCreateCUDAMemWrapper(prevDat, VPI_BACKEND_CUDA, &__prevCUDA));
    if (LAZYASS_DEBUG) std::cout << "prevDat was successfully wrapped in a VPI Image." << std::endl;
    CHECK_STATUS(vpiImageCreateCUDAMemWrapper(currDat, VPI_BACKEND_CUDA, &__currCUDA));
    if (LAZYASS_DEBUG) std::cout << "CUDA Wrapped!" << std::endl;

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
        if (LAZYASS_DEBUG) {
            CHECK_STATUS(vpiStreamSync(__stream));
            std::cout << "Image converted." << std::endl;
        }
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
    if (LAZYASS_DEBUG) {
        CHECK_STATUS(vpiStreamSync(__stream));
        std::cout << "Converted to BL." << std::endl;
    }

    /// Compute ///
    // Set up and compute dense optical flow
    CHECK_STATUS(vpiImageCreate(prevRaw.size(1) / 4, prevRaw.size(2) / 4, VPI_IMAGE_FORMAT_2S16_BL, VPI_BACKEND_NVENC, &__motionBL));
    CHECK_STATUS(vpiCreateOpticalFlowDense(VPI_BACKEND_NVENC, 
        prevRaw.size(1), prevRaw.size(2), VPI_IMAGE_FORMAT_NV12_ER_BL, 
        __qualityNative, &__payload));
    if (LAZYASS_DEBUG) {
        CHECK_STATUS(vpiStreamSync(__stream));
        std::cout << "Sync'd before the dense calc." << std::endl;
    }
    CHECK_STATUS(vpiSubmitOpticalFlowDense(__stream, VPI_BACKEND_NVENC, __payload, 
        __prevInter, __currInter, __motionBL));
    if (LAZYASS_DEBUG) std::cout << "Dense calc'd." << std::endl;


    /// Return ///
    // Final sync-up for the VPI prior to falling back into torch
    CHECK_STATUS(vpiStreamSync(__stream));
    if (LAZYASS_DEBUG) std::cout << "Ready for final conversion." << std::endl;

    // Handle conversion from NVidia's inhouse chosen S10.5 format
    // https://docs.nvidia.com/vpi/algo_optflow_dense.html
    VPIImageData resultDat;
    CHECK_STATUS(vpiImageLock(__motionBL, VPI_LOCK_READ, &resultDat));

    auto result = torch::zeros({prevRaw.size(0), prevRaw.size(1) / 4, prevRaw.size(2) / 4, 2}, __torchOpts);
    for (int i = 0; i < prevRaw.size(0); i++) {
        result[i] = torch::from_blob(resultDat.planes[i].data, {prevRaw.size(1), prevRaw.size(2), 2}, __torchOpts);
    }
    
    CHECK_STATUS(vpiImageUnlock(__motion));

    // Clean up
    delete prevDat;
    delete currDat;

    if (!keepAliveStream) vpiStreamDestroy(__stream);

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

#include <pybind11/pybind11.h>
namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Runs a torch tensor through the nvidia vpi";
    m.def("denseFlow", &denseFlow, "NV VPI Dense Optical Flow",
        py::arg("prevImage"),
        py::arg("currImage"),
        py::arg("quality") = "high",
        py::arg("format") = "rgb",
        py::arg("upscale") = false,
        py::arg("keepAliveStream") = true);
}
