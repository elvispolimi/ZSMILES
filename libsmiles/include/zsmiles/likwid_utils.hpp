#pragma once

#ifdef LIKWID_PERFMON
  #include <likwid-marker.h>

  #if defined(GPU_NVIDIA)
    #define GPUMON_MARKER_INIT                NVMON_MARKER_INIT
    #define GPUMON_MARKER_THREADINIT          NVMON_MARKER_THREADINIT
    #define GPUMON_MARKER_SWITCH              NVMON_MARKER_SWITCH
    #define GPUMON_MARKER_REGISTER(regionTag) NVMON_MARKER_REGISTER(regionTag)
    #define GPUMON_MARKER_START(regionTag)    NVMON_MARKER_START(regionTag)
    #define GPUMON_MARKER_STOP(regionTag)     NVMON_MARKER_STOP(regionTag)
    #define GPUMON_MARKER_CLOSE               NVMON_MARKER_CLOSE
    #define GPUMON_MARKER_GET(regionTag, nevents, events, time, count) \
      NVMON_MARKER_GET(regionTag, nevents, events, time, count)
  #elif defined(GPU_AMD)
    #define GPUMON_MARKER_INIT                ROCMON_MARKER_CLOSE MON_MARKER_INIT
    #define GPUMON_MARKER_THREADINIT          ROCMON_MARKER_CLOSE MON_MARKER_THREADINIT
    #define GPUMON_MARKER_SWITCH              ROCMON_MARKER_CLOSE MON_MARKER_SWITCH
    #define GPUMON_MARKER_REGISTER(regionTag) ROCMON_MARKER_CLOSE MON_MARKER_REGISTER(regionTag)
    #define GPUMON_MARKER_START(regionTag)    ROCMON_MARKER_CLOSE MON_MARKER_START(regionTag)
    #define GPUMON_MARKER_STOP(regionTag)     ROCMON_MARKER_CLOSE MON_MARKER_STOP(regionTag)
    #define GPUMON_MARKER_CLOSE               ROCMON_MARKER_CLOSE MON_MARKER_CLOSE
    #define GPUMON_MARKER_GET(regionTag, nevents, events, time, count) \
      ROCMON_MARKER_GET(regionTag, nevents, events, time, count)
  #endif
#else
  #define LIKWID_MARKER_INIT
  #define LIKWID_MARKER_THREADINIT
  #define LIKWID_MARKER_SWITCH
  #define LIKWID_MARKER_REGISTER(regionTag)
  #define LIKWID_MARKER_START(regionTag)
  #define LIKWID_MARKER_STOP(regionTag)
  #define LIKWID_MARKER_CLOSE
  #define LIKWID_MARKER_GET(regionTag, nevents, events, time, count)

  #define GPUMON_MARKER_INIT
  #define GPUMON_MARKER_THREADINIT
  #define GPUMON_MARKER_SWITCH
  #define GPUMON_MARKER_REGISTER(regionTag)
  #define GPUMON_MARKER_START(regionTag)
  #define GPUMON_MARKER_STOP(regionTag)
  #define GPUMON_MARKER_CLOSE
  #define GPUMON_MARKER_GET(regionTag, nevents, events, time, count)
#endif
