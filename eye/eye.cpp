#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#define _USE_MATH_DEFINES
#include <cmath>

#define IRIS_MISS_MESSAGE "Invalid iris layer provided."

#include "eye.hpp"

namespace eye {
    float __blend(float a, float b, float x) {
        return ((b - a) * x) + a;
    }
    float __scale(float x, float originLow, float originHigh, float targetLow, float targetHigh) {
        float originRange = originHigh - originLow;
        float targetRange = targetHigh - targetLow;
        
        float preTarget = (x - originLow) / originRange;
        return (x * targetRange) + targetLow;
    }
    void IrisContour::draw(cv::Mat& img, const cv::Mat& occupancyGrid, float x, float y, 
    float intensity) {
        /**
         * A couple of steps are needed to arrive at the appropraitely formatted
         * data for the iris's defining functions.
         * 
         * 1.   Figure out fundamental parameters for siniosoidal functions.
         * 2.   Create initial vertices in polar coordinates.
         * 3.   Convert coordinates to opencv screen space coordinates.
         * 4.   Draw filled contour.
         */

        /* 1 */

        // Conver the x and y coordinates into screen space coordinates
        size_t sx, sy;
        if (img.rows > img.cols) {
            sx = __scale(x, -1, 1, 0, img.cols);
            sy = __scale(y, -1, 1, 0, img.cols);
        }
        else {
            sx = __scale(x, -1, 1, 0, img.rows);
            sy = __scale(y, -1, 1, 0, img.rows);
        }

        // The maximum physical eye size in pixels
        float maxOccupancy = occupancyGrid.at<float>(y, x);

        // The current size based off of the intensity provided
        float size;
        if (this->_sizeFunction == nullptr) {
            size = __blend(this->_sizeRange[(int)IRIS_INNER], 
                this->_sizeRange[(int)IRIS_OUTER], intensity);
        }
        else {
            size = this->_sizeFunction(intensity).continuous;
        }

        // The current amplitude based off of the intensity provided
        float amplitude;
        if (this->_amplitudeFunction == nullptr) {
            amplitude = __blend(this->_amplitudeRange[(int)IRIS_INNER],
                this->_amplitudeRange[(int)IRIS_OUTER], intensity);
        }
        else {
            amplitude = this->_amplitudeFunction(intensity).continuous;
        }

        // Account for extraneous functional implementations
        float phaseShift = this->_phaseFunction == nullptr ? this->_phaseShift 
            : this->_phaseFunction(intensity).continuous;
        RGB color = this->_colorFunction == nullptr ? this->_color
            : this->_colorFunction(intensity).color;
        
        
        // The actual occupancy of the layer
        float occupancy = maxOccupancy * size;

        // Finally, the sinusoidal frequency of the contour
        float omega = 2 * M_PI / (this->_vertices / this->_harmonic);

        /* 2 */

        // Apply each vertex
        // TODO: Threading
        float magnitude;
        for (int i = 0; i < this->_vertices; i++) {
            // Grab magnitude of the vector from the occupancy
            magnitude = occupancy;

            // Subtract the sinusoidal waveform from the eye, peak at 0.0
            magnitude -= amplitude * (sin(omega * (x + phaseShift)) - 1.0);

            /* 3 */

            // Convert to screen space coordinates
            cv::Point2i* target = &this->__vertexBuffer[i];
            target->x = magnitude * cos(2.0 * M_PI * ((double)i / this->_vertices));
            target->y = magnitude * sin(2.0 * M_PI * ((double)i / this->_vertices));
            *target += cv::Point2i(x, y);
        }

        /* 4 */

        // Assume BGR colorspace
        cv::Scalar polygonColor = cv::Scalar(color.B, color.G, color.R);
        // Draw polygon
        cv::fillPoly(img, &this->__vertexBuffer, &this->_vertices, 1, polygonColor);
    }

    void Iris::draw(cv::Mat& img, const cv::Mat& occupancyGrid, float x, float y, float intensity) {
        /**
         * To transfer the structure of the iris down to the lower layer contours,
         * a couple of steps must be done. The innermost contour of the iris will
         * actually be used as a helper to figure out how to mask the outer
         * contour.
         */

        // Running data
        cv::Mat baseMask , floatMask, floatImg, floatIris;
        baseMask = cv::Mat(img.rows, img.cols, cv::CV_8UC3);
        baseMask.setTo(0);

        // Draw the inner chunk of the iris on the inner contour/mask
        this->_inner->setColor(this->__MASK_COLOR);
        this->_inner->draw(baseMask, occupancyGrid, x, y, intensity);

        // Convert the masks to something more scalable
        baseMask.convertTo(floatMask, cv::CV_32FC3, (float)1/255);
        img.convertTo(floatImg, cv::CV_32FC3);

        // Draw the outer chunk of the iris normally
        this->_inner->draw(img, occupancyGrid, x, y, intensity);
        img.convertTo(floatIris, cv::CV_32FC3);

        // Mask inside of pupil
        floatImg = floatImg.mul(floatMask);

        // Mask outside of pupil
        floatMask = 1.0 - floatMask;
        floatIris = floatIris.mul(floatMask);

        // Combine and save
        floatImg += floatIris;
        floatImg.convertTo(img, img.type());
    }
}
