#pragma once

#ifdef LIKWID_PERFMON
  #include <likwid-marker.h>
#else
  #define LIKWID_MARKER_INIT
  #define LIKWID_MARKER_THREADINIT
  #define LIKWID_MARKER_SWITCH
  #define LIKWID_MARKER_REGISTER(regionTag)
  #define LIKWID_MARKER_START(regionTag)
  #define LIKWID_MARKER_STOP(regionTag)
  #define LIKWID_MARKER_CLOSE
  #define LIKWID_MARKER_GET(regionTag, nevents, events, time, count)

  #define NVMON_MARKER_INIT
  #define NVMON_MARKER_THREADINIT
  #define NVMON_MARKER_SWITCH
  #define NVMON_MARKER_REGISTER(regionTag)
  #define NVMON_MARKER_START(regionTag)
  #define NVMON_MARKER_STOP(regionTag)
  #define NVMON_MARKER_CLOSE
  #define NVMON_MARKER_GET(regionTag, nevents, events, time, count)
#endif
