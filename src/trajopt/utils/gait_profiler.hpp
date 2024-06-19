#pragma once

#include <numeric>
#include <trajopt/robo_spline/types.hpp>
#include <tuple>

namespace trajopt {

    inline std::tuple<std::vector<double>, std::vector<size_t>, std::vector<size_t>> createGait(size_t numSteps,
        double stanceTime,
        double swingTime,
        size_t knotsPerSwing,
        size_t knotsPerForceSwing,
        rspl::Phase initialPhase)
    {
        std::vector<size_t> stepKnotsPerSwing;
        std::vector<size_t> forceKnotsPerSwing;
        std::vector<double> phaseTimes;

        bool isStance = (initialPhase == rspl::Phase::Stance) ? true : false;
        size_t stepCounter = 0;
        while (stepCounter < numSteps) {
            if (isStance) {
                stepCounter++;
                forceKnotsPerSwing.push_back(knotsPerForceSwing);

                phaseTimes.push_back(stanceTime);
            }
            else {
                phaseTimes.push_back(swingTime);
                stepKnotsPerSwing.push_back(knotsPerSwing);
            }
            isStance = !isStance;
        }
        return std::make_tuple(phaseTimes, stepKnotsPerSwing, forceKnotsPerSwing);
    }

    inline void fixDurations(std::vector<std::vector<double>>& phaseTimes)
    {
        // find largest duration
        double max = 0.;
        for (auto& vec : phaseTimes) {
            double duration = std::accumulate(vec.begin(), vec.end(), 0.);
            if (duration > max)
                max = duration;
        }

        for (auto& vec : phaseTimes) {
            double duration = std::accumulate(vec.begin(), vec.end(), 0.);
            vec.back() += max - duration;
        }
    }
} // namespace trajopt
