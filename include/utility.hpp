// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

# pragma once
#include <samples/ocv_common.hpp>
#include <samples/slog.hpp>

#include "detectors.hpp"
#include "feature_extractor.hpp"
#include "classifier_model.hpp"

template<typename Component>
struct Load {
    Component& component;

    explicit Load(Component& component);

    void into(InferenceEngine::InferencePlugin & plg, bool enable_dynamic_batch = false) const;
};

class CallStat {
public:
    typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;

    CallStat();

    double getSmoothedDuration();
    double getTotalDuration();
    void calculateDuration();
    void setStartTime();

private:
    size_t _number_of_calls;
    double _total_duration;
    double _last_call_duration;
    double _smoothed_duration;
    std::chrono::time_point<std::chrono::high_resolution_clock> _last_call_start;
};

class Timer {
public:
    void start(const std::string& name);
    void finish(const std::string& name);
    CallStat& operator[](const std::string& name);

private:
    std::map<std::string, CallStat> _timers;
};
