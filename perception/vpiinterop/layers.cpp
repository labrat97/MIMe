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

#include <torch/extension.h>
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")


VPIStream __stream;
VPIOpticalFlowQuality __qualityNative;
VPIImageFormat __formatNative;
VPIPixelType __pixelType;
VPIImagePlane __prevPlanes[VPI_MAX_PLANE_COUNT];
VPIImagePlane __currPlanes[VPI_MAX_PLANE_COUNT];

VPIImage __prevInter, __currInter;
VPIImage __prevCUDA, __currCUDA, __prevBL, __currBL;
VPIImage __motionBL, __motion;

VPIImageData __transfer(torch::Tensor* x, VPIImagePlane* planes) {
        for (int i = 0; i < x->size(0); i++) {
            auto plane = &(planes[i]);

            plane->data = x->data_ptr();
            plane->height = x->size(1);
            plane->width = x->size(2);
            plane->pitchBytes = 1;
            plane->pixelType = __pixelType;
        }

        return {
            .format = __formatNative,
            .numPlanes = x->size(0),
            .planes = planes
        };
    };

torch::Tensor denseFlow(torch::Tensor prevImg, torch::Tensor currImg, 
        std::string quality = "med", std::string format = "rgb",
        bool upscale = false) {
    /// Precheck ///
    // The correct backend is being used
    CHECK_CUDA(prevImg);
    CHECK_CUDA(currImg);

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

    // Get ready for NVENC with the VIC
    VPIImage prev, curr;
    if (__formatNative != VPI_IMAGE_FORMAT_NV12_ER) {
        // Convert the images
        CHECK_STATUS(vpiImageCreate(prevRaw.size(1), prevRaw.size(2), 
            VPI_IMAGE_FORMAT_NV12_ER, VPI_BACKEND_CUDA, &__prevInter));
        CHECK_STATUS(vpiImageCreate(currRaw.size(1), currRaw.size(2),
            VPI_IMAGE_FORMAT_NV12_ER, VPI_BACKEND_CUDA, &__currInter));
        
        // Format to something that works with the NVENC
        VPIConvertImageFormatParams toNV12;
        toNV12.scale = 1; toNV12.offset = 0;
        toNV12.flags = VPI_BACKEND_CUDA;
        toNV12.policy = VPI_CONVERSION_CLAMP;
        CHECK_STATUS(vpiInitConvertImageFormatParams(&toNV12));
        CHECK_STATUS(vpiSubmitConvertImageFormat())
    }
}
