if __name__ == '__main__':
    import cv2 as cv
import numpy as np
import vpi
from numba import cuda as nb


@nb.autojit
def pred(previousImage, currentImage, quality=vpi.OptFlowQuality.HIGH, upscale:bool = False) -> np.float16:
    # Do the main image conversion through the CUDA cores
    asshole:vpi.Backend = vpi.Backend.CUDA
    pimg:vpi.Image = previousImage if previousImage is vpi.Image else vpi.asimage(previousImage, format=vpi.Format.BGR8, backend=asshole)
    pimg = pimg.convert(vpi.Format.NV12_ER, backend=asshole)
    cimg:vpi.Image = currentImage if currentImage is vpi.Image else vpi.asimage(currentImage, format=vpi.Format.BGR8, backend=asshole)
    cimg = cimg.convert(vpi.Format.NV12_ER, backend=asshole)
    
    # Get vision processor to handle some image conversion
    asshole = vpi.Backend.VIC
    pimg = pimg.convert(vpi.Format.NV12_ER_BL)
    cimg = cimg.convert(vpi.Format.NV12_ER_BL)

    # Pipe to encoder to offload work from GPU
    asshole = vpi.Backend.NVENC
    motion:vpi.Image = vpi.optflow_dense(pimg, cimg, quality=quality, backend=asshole)
    # Convert from S10.5 format as according to the docs
    # https://docs.nvidia.com/vpi/sample_optflow_dense.html
    if not upscale:
        flow:np.float16 = np.float16(motion.rlock().cpu())/(1<<5)
    else:
        flowRaw:np.int16 = np.int16(motion.rlock().cpu())
    
    # Exit early if not upscaling the motion vector to the original resolution
    if not upscale:
        return flow


    flows = []
    for idx in range(flowRaw.shape[-1]):
        # Convert the original image into something that the VIC can handle to avoid
        # using the precious CUDA cores for resampling
        asshole = vpi.Backend.CUDA
        flowSlice:np.int16 = np.array(np.int16(flowRaw[:,:,idx]))
        workingImg:vpi.Image = vpi.asimage(flowSlice, format=vpi.Format.S16)
        scaledImg:vpi.Image = workingImg.convert(vpi.Format.Y16_ER, backend=asshole)

        # Scale and interpolate
        asshole = vpi.Backend.VIC
        # Everything but the initial size is basically a requirement of the VPI
        scaledImg = scaledImg.rescale((cimg.width, cimg.height), \
            interp=vpi.Interp.LINEAR, border=vpi.Border.CLAMP, backend=asshole)
        
        # Append for end product
        flows.append(scaledImg)
    
    # Turn into one image with the same cartesean coordinate positioning
    cpuFlows = []
    for flow in flows:
        # Convert from S10.5 format as according to the docs
        # https://docs.nvidia.com/vpi/sample_optflow_dense.html
        current:np.float16 = np.float16(flow.rlock().cpu())/(1<<5)
        
        # Set up the arrays to be concatenated into one final cpu bounded image
        river:np.float16 = np.expand_dims(current, axis=-1)
        cpuFlows.append(river)
    
    # Perform the final merge
    return np.concatenate(cpuFlows, axis=-1)


if __name__ == "__main__":
    import sys
    import gc

    import syscamera as camera

    def __listret__(frames):
        result:bool = True
        for frame in frames:
            ret = frame
            result = result and ret
        return result

    CAM_COUNT:int = int(sys.argv[-1]) if len(sys.argv) > 1 else 1
    CACHE_MOD_CLEAR:int = 64

    cams = [camera.configurationString(idx, 3264, 2464, 21) for idx in range(2)]
    caps = [cv.VideoCapture(cams[idx], cv.CAP_GSTREAMER) for idx in range(2)]
    previousFrames = [caps[idx].read() for idx in range(2)]

    ret:bool = __listret__(previousFrames)
    cacheItr:int = 0
    while ret:
        if cacheItr % CACHE_MOD_CLEAR == 0:
            gc.collect()
            vpi.clear_cache()
        currentFrames = [caps[idx].read() for idx in range(2)]
        if not __listret__(currentFrames):
            break
        motions = [pred(previousFrames[idx][1], currentFrames[idx][1], upscale=True) for idx in range(CAM_COUNT)]

        print(f'frame: {currentFrames[0][1].shape}')
        print(f'motion: {motions[0].shape}\n')

        previousFrames = currentFrames
        ret = __listret__(previousFrames)