#ifndef MIME_EYE_MAIN_HPP
#define MIME_EYE_MAIN_HPP

#include <cstdint>
#include <opencv2/core.hpp>

namespace eye {
    /**
     * This was made to make a solid datatype for a color.
     * TODO: Refine later to maybe opencv version.
     */
    typedef struct RGB {
        uint8_t R;
        uint8_t G;
        uint8_t B;
    } RGB;

    typedef enum IRIS_LEVEL {
        IRIS_INNER=0,
        IRIS_OUTER=1
    } IRIS_LEVEL;

    /**
     * Defines the logic for a single contour in the iris of the robot. Multiple
     * instances of these contours will be combined to produce the final iris.
     */
    class IrisContour {
        float _amplitudeRange[2] = {0.0, 0.25};
        float _sizeRange[2] = {0.5, 1.0};
        float _phaseShift = 0.0;
        RGB _color = {255, 255, 255};
        uint16_t _harmonic;

        const uint16_t _vertices;

    private:
        cv::Point2i* __vertexBuffer;

    public:
        IrisContour(uint16_t harmonic=9, uint16_t vertices=512)
        : _vertices(vertices), _harmonic(harmonic) {
            __vertexBuffer = new cv::Point2i[_vertices];
        }
        ~IrisContour() {
            delete __vertexBuffer;
        }

        /**
         * Gets/sets the color for the contour.
         */
        void setColor(RGB color) { _color = color; }
        RGB getColor() { return _color; }

        /**
         * Gets/sets the harmonic for the contour's animation.
         */
        void setHarmonic(uint16_t harmonic) { _harmonic = harmonic; }
        uint16_t getHarmonic() { return _harmonic; }

        /**
         * Sets the ampiltude range of the contour. This range is used for
         * the siniosoidal functionality of the contour, adding a sort of animation
         * to the class.
         */
        void setAmplitudeRange(float upperRange, float lowerRange=0.0) {
            _amplitudeRange[(int)IRIS_INNER] = lowerRange;
            _amplitudeRange[(int)IRIS_OUTER] = upperRange;
        }

        /**
         * Sets the peak to peak range of the contour. This is wrapper around the
         * ::setAmplitudeRange() function contained in this class.
         */
        void setPeakToPeakRange(float upperRange, float lowerRange=0.0) {
            _amplitudeRange[(int)IRIS_INNER] = lowerRange / 2.0;
            _amplitudeRange[(int)IRIS_OUTER] = upperRange / 2.0;
        }

        /**
         * Sets the base range of the iris size (the baseline of the sinusoidal
         * function) to the range provided. This range also adds some sort of
         * animation to the contour.
         */
        void setSizeRange(float upperRange, float lowerRange) {
            _sizeRange[(int)IRIS_INNER] = lowerRange;
            _sizeRange[(int)IRIS_OUTER] = upperRange;
        }

        /**
         * Gets/sets the phase shift of the sinusoidal function of the Iris
         * contour.
         */
        void setPhaseShift(float phaseShift) { _phaseShift = phaseShift; }
        float getPhaseShift() { return _phaseShift; }

        /**
         * Draws the iris onto the provided cv::Mat at the approximate positioning
         * given by arguments x and y and the occupancy matrix. The smaller of the
         * two dimensions, x and y, will be ranged between [-1.0, 1.0] as where the
         * larger of the two will retain the height-to-width ratio. The intensity
         * parameter operates in the range of [0.0, 1.0] and applies the intensity-based
         * amplitudes ranges accordingly.
         */
        void draw(cv::Mat& img, const cv::Mat& occupancyGrid, float x, float y, float intensity);
    };

    class IrisLayer {
        IrisContour* _outer;
        IrisContour* _inner;

    private:
        cv::Mat* mask = nullptr;

    public:
        IrisLayer(uint16_t layerVertices=512) {
            _outer = new IrisContour(9, layerVertices);
            
            _inner = new IrisContour(7, layerVertices);
        }
    };
}

#endif
