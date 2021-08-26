if __name__ == '__main__':
    import cv2 as cv
import vpi
import torch


@torch.jit.script
def pred(previousImage, currentImage, quality=vpi.OptFlowQuality.MEDIUM, upscale:bool = False) -> torch.Tensor:
    # Do the main image conversion through the CUDA cores
    asshole = vpi.Backend.CUDA
    pimg = previousImage if previousImage is vpi.Image else vpi.asimage(previousImage, format=vpi.Format.BGR8)
    pimg = pimg.convert(vpi.Format.NV12_ER, backend=asshole)
    cimg = currentImage if currentImage is vpi.Image else vpi.asimage(currentImage, format=vpi.Format.BGR8)
    cimg = cimg.convert(vpi.Format.NV12_ER, backend=asshole)
    
    # Get vision processor to handle some image conversion
    asshole = vpi.Backend.VIC
    pimg = pimg.convert(vpi.Format.NV12_ER_BL, backend=asshole)
    cimg = cimg.convert(vpi.Format.NV12_ER_BL, backend=asshole)

    # Pipe to encoder to offload work from GPU
    asshole = vpi.Backend.NVENC
    motion = vpi.optflow_dense(pimg, cimg, quality=quality, backend=asshole)

    # Convert from S10.5 format as according to the docs
    # https://docs.nvidia.com/vpi/sample_optflow_dense.html
    if not upscale:
        #flow = np.float16(motion.cpu())/(1<<5)
        flow = torch.HalfTensor(motion.rlock().cpu()).detach().cuda()
        flow.type = torch.float16
        flow.div_(1<<5)
    else:
        #flowRaw = np.int16(motion.cpu())
        flowRaw = torch.ShortTensor(motion.rlock().cpu()).detach().cuda()
        flowRaw.type = torch.int16
    
    # Exit early if not upscaling the motion vector to the original resolution
    if not upscale:
        return flow


    # Upsample using hardware acceleration
    flows = []
    for idx in range(flowRaw.shape[-1]):
        # The VIC is too slow for how much this could be called
        asshole = vpi.Backend.CUDA
        # Casting needed, not quite sure why but we are rollin with it
        flowSlice = np.int16(flowRaw[:,:,idx].cpu().numpy())
        wimg = vpi.asimage(flowSlice, format=vpi.Format.S16) \
            .convert(format=vpi.Format.Y16_ER, backend=asshole)
        
        asshole = vpi.Backend.VIC
        wimg = wimg.rescale((cimg.width, cimg.height), \
                interp=vpi.Interp.LINEAR, border=vpi.Border.CLAMP, backend=asshole)
        
        # Append for end product
        flows.append(wimg)
    
    # Turn into one image with the same cartesean coordinate positioning
    plasmaFlows = []
    for flow in flows:
        # Convert from S10.5 format as according to the docs
        # https://docs.nvidia.com/vpi/sample_optflow_dense.html
        current = torch.HalfTensor(np.int16(flow.rlock().cpu())).detach().cuda()
        current.type = torch.float16
        current.div_(1<<5)
        
        # Set up the arrays to be concatenated into one final cpu bounded image
        #river = np.expand_dims(current, axis=-1)
        river = torch.unsqueeze(current, dim=-1)
        plasmaFlows.append(river)
    
    # Perform the final merge
    return torch.cat(plasmaFlows, dim=-1)


if __name__ == "__main__":
    import sys
    import gc

    import syscamera as camera
    import numpy as np
    import time

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
    cacheItr:int = 4
    UPSCALE:bool = False

    try:
        capTimes = []
        startTimes = []
        stopTimes = []
        while ret:
            if cacheItr % CACHE_MOD_CLEAR != 0:
                startTimes.append(time.time())

            currentFrames = [caps[idx].read() for idx in range(2)]
            if cacheItr % CACHE_MOD_CLEAR != 0:
                capTimes.append(time.time())

            ret = __listret__(currentFrames)
            if not ret:
                break
            motions = [pred(previousFrames[idx][1], currentFrames[idx][1], upscale=UPSCALE) for idx in range(CAM_COUNT)]

            # Scoot the frames on back
            previousFrames = currentFrames

            # Clear cache at zero and clear cache every CACHE_MOD_CLEAR steps
            if cacheItr % CACHE_MOD_CLEAR != 0:
                stopTimes.append(time.time())
            cacheItr = cacheItr + 1
            if cacheItr % CACHE_MOD_CLEAR == 0:
                gc.collect()
                vpi.clear_cache()

                startT = torch.DoubleTensor(startTimes).detach().cpu()
                capT = torch.DoubleTensor(capTimes).detach().cpu()
                stopT = torch.DoubleTensor(stopTimes).detach().cpu()

                outerLoopAvg = torch.mean(stopT-startT, axis=0)
                capTimeAvg = torch.mean(capT-startT, axis=0)
                postProcAvg = torch.mean(stopT-capT, axis=0)

                torch.cuda.synchronize()
                print(f"outer-loop-itr (sec):\t{float(outerLoopAvg)}" \
                    + f" fps: {int(1/outerLoopAvg)} Hz")
                print(f"cap-time (sec): \t{float(capTimeAvg)}" \
                    + f" fps: {int(1/capTimeAvg)} Hz")
                print(f"post-proc (sec): \t{float(postProcAvg)}" \
                    + f" fps: {int(1/postProcAvg)} Hz")
                print(f"upscale (bool): \t{UPSCALE}\n")

                startTimes.clear()
                capTimes.clear()
                stopTimes.clear()
    except InterruptedError:
        [n.release() for n in caps]
