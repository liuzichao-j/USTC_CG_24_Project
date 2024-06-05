#pragma once
#include <pxr/usd/usd/stage.h>

#include "USTC_CG.h"

USTC_CG_NAMESPACE_OPEN_SCOPE

// This is not best practice, but I am really in a hurry to get them all
// running. Later this will be improved with usd path resolver functionality.
// This stage serves for sharing data from the nodes to the renderer
struct GlobalUsdStage {
    static pxr::UsdStageRefPtr global_usd_stage;

    static constexpr int timeCodesPerSecond = 15;
    static inline float speed_of_light = 1000.0f;
    static inline int enable_limited_light_speed_transform = 1;
    static inline int iteration_num = 5;
    static inline float iteration_damping = 0.5;
    static inline int enable_god_view = 0;

    static inline std::vector<void*> relativity_console_bind_data;
};
USTC_CG_NAMESPACE_CLOSE_SCOPE