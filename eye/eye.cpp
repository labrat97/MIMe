#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>

#define IRIS_MISS_MESSAGE "Invalid iris layer provided."

#include "eye.hpp"

namespace eye {
    float __blend(float a, float b, float x) {
        return ((b - a) * x) + a;
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

        // The maximum physical eye size in pixels
        float maxOccupancy = occupancyGrid.at<float>(y, x);

        // The current size based off of the intensity provided
        float size;
        if (this->_sizeFunction == nullptr) {
            size = __blend(this->_sizeRange[(int)IRIS_INNER], 
                this->_sizeRange[(int)IRIS_OUTER], intensity);
        }
        else {
            size = this->_sizeFunction(intensity);
        }

        // The current amplitude based off of the intensity provided
        float amplitude;
        if (this->_amplitudeFunction == nullptr) {
            amplitude = __blend(this->_amplitudeRange[(int)IRIS_INNER],
                this->_amplitudeRange[(int)IRIS_OUTER], intensity);
        }
        else {
            amplitude = this->_amplitudeFunction(intensity);
        }

        // Account for extraneous functional implementations
        float phaseShift = this->_phaseFunction == nullptr ? this->_phaseShift 
            : this->_phaseFunction(intensity);
        RGB color = this->_colorFunction == nullptr ? this->_color
            : this->_colorFunction(intensity);
        
        
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
}
