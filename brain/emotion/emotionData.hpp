#ifndef MIME_EMOTION_DATA_HPP
#define MIME_EMOTION_DATA_HPP

#include <vector>
#include <cstdint>

/**
 * The way that the emotions are encoded onto the MIMe is according to polar
 * coordinates. This is done to follow the emotional wheel model, and allows
 * some sort of emotional convolution to be computed in a reasonable time (as
 * apposed to making a whole network for emotion mixing).
 * 
 * The emotions to be used are:
 * {Anger, Disgust, Fear, Joy, Sadness, Surprise}
 */
namespace emotion {
    /**
     * This ad-hoc enum holds the lowest order emotional positions available for
     * evaluation.
     */
    class DefaultEmotions {
        float _angle;
        DefaultEmotions(float angle) : _angle(angle) {}
    public:
        static const DefaultEmotions Anger;
        static const DefaultEmotions Disgust;
        static const DefaultEmotions Fear;
        static const DefaultEmotions Joy;
        static const DefaultEmotions Sadness;
        static const DefaultEmotions Surprise;
        static const DefaultEmotions Interest;
        static const DefaultEmotions Comfort;
        operator float() const { return _angle; }
    };
    const DefaultEmotions DefaultEmotions::Anger(180.0f);
    const DefaultEmotions DefaultEmotions::Disgust(235.0f);
    const DefaultEmotions DefaultEmotions::Fear(0.0f);
    const DefaultEmotions DefaultEmotions::Joy(90.0f);
    const DefaultEmotions DefaultEmotions::Sadness(270.0f);
    const DefaultEmotions DefaultEmotions::Surprise(315.0f);
    const DefaultEmotions DefaultEmotions::Interest(135.0f);
    const DefaultEmotions DefaultEmotions::Comfort(55.0f);

    /**
     * To express the above emotions, an intensity is needed.
     */
    typedef struct Emotion {
        float position; // in degrees
        float intensity; // [0.0, 1.0]
    } Emotion;

    /**
     * Collect the above emotions into a ring for analysis.
     */
    class EmotionalRing : public std::vector<Emotion> {
    public:
        const std::vector<EmotionalRing> evaluateHigherOrderEmotions(
            std::size_t maxDepth=0);
    };
    const std::vector<EmotionalRing> EmotionalRing::evaluateHigherOrderEmotions(
        std::size_t maxDepth=0) {
        // Running data
        std::vector<EmotionalRing> result;
        EmotionalRing workingSet;
        EmotionalRing resultSet;
        std::size_t bundleSize = this->size();

        // Figure out pyramid depth
        if (maxDepth == 0 || maxDepth > bundleSize-1) {
            maxDepth = bundleSize-1;
        }

        // Mix to higher order
        // TODO: Less copying
        for (std::size_t d = 0; d < maxDepth; d++) {
            workingSet = *this;

            for (std::size_t i = 0; i < bundleSize-1; i++) {
                Emotion element = workingSet.at(i);

                // operation: avg(\/\/\/\/\/\/\/)
                for (std::size_t j = i+1; j < bundleSize; j++) {
                    Emotion bindingElement = workingSet.at(j);

                    // Average theta and r in terms of 2D polar coordinates
                    bindingElement.intensity = (bindingElement.intensity+element.intensity)/2;
                    bindingElement.position = (bindingElement.position+element.position)/2;
                    if (bindingElement.position >= 360) bindingElement.position -= 360;

                    resultSet.push_back(bindingElement);
                }
            }

            // Add to result pile
            result.push_back(resultSet);
            workingSet = resultSet;
            resultSet.clear();
        }

        return result;
    }
}

#endif
