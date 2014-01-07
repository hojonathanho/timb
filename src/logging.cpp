#include <string>
#include <iostream>
#include "logging.hpp"
#include <cstdlib>
using namespace std;

namespace util {

LogLevel gLogLevel;
bool gLoggingInitialized = false;

int LoggingInit() {
  if (gLoggingInitialized) return 1;

  const char* VALID_THRESH_VALUES = "FATAL ERROR WARN INFO DEBUG TRACE";

  char* lvlc = getenv("TRAJOPT_LOG_THRESH");
  string lvlstr;
  if (lvlc == NULL) {
    lvlstr = "INFO";
  }
  else lvlstr = string(lvlc);
  if (lvlstr == "FATAL") gLogLevel = LevelFatal;
  else if (lvlstr == "ERROR") gLogLevel =  LevelError;
  else if (lvlstr == "WARN") gLogLevel = LevelWarn;
  else if (lvlstr == "INFO") gLogLevel = LevelInfo;
  else if (lvlstr == "DEBUG") gLogLevel = LevelDebug;
  else if (lvlstr == "TRACE") gLogLevel = LevelTrace;
  else {
    printf("Invalid value for environment variable TRAJOPT_LOG_THRESH: %s\n", lvlstr.c_str());
    printf("Valid values: %s\n", VALID_THRESH_VALUES);
    printf("Defaulting to INFO.\n");
    gLogLevel = LevelInfo;
  }

  gLoggingInitialized = true;
  return 1;
}

int this_is_a_hack_but_rhs_executes_on_library_load = LoggingInit();

}
