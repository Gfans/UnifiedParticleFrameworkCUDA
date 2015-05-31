#include "scalar.h"

#ifdef WIN32
#include <stdio.h>
#include <windows.h>
#else
#include <stdio.h>
#include <sys/resource.h>
#include <sys/times.h>
#endif


// Difference in successive myTime() calls tells you the elapsed time
// - return time in seconds

double ElapsedUserTime(){
  #ifdef WIN32

    // Parameters for GetThreadTimes()
    FILETIME creationTime, exitTime, kernelTime, userTime;

    // Actually get the thread timings
    GetProcessTimes( GetCurrentProcess(),&creationTime,
                                  &exitTime, &kernelTime, &userTime );


    double lo_time;
    double hi_time;
    double seconds;

    // Get the low order data.
    if(userTime.dwLowDateTime < 0){
        lo_time = 2 ^ 31 + (userTime.dwLowDateTime & 0x7FFFFFFF);
    } else {
        lo_time = userTime.dwLowDateTime;
    }

    // Get the high order data.
    if(userTime.dwHighDateTime < 0){
        hi_time = 2 ^ 31 + (userTime.dwHighDateTime & 0x7FFFFFFF);
    } else {
        hi_time = userTime.dwHighDateTime;
    }

    // Combine them and turn the result into hours.
    seconds = (lo_time + pow(2.0,32) * hi_time) / 10000000;

    return seconds;

    //////////////////////////////////////////////////////
    // WIN32: use QueryPerformance (very accurate)
    //////////////////////////////////////////////////////

//    LARGE_INTEGER freq , t ;

    // freq is the clock speed of the CPU
  //  QueryPerformanceFrequency ( & freq ) ;

	//cout << "freq = " << ((double) freq.QuadPart) << endl ;

    // t is the high resolution performance counter (see MSDN)
    //QueryPerformanceCounter ( & t ) ;

    //return ( (double) t.QuadPart /(double) (freq.QuadPart * 10000.0)) ;
    //return (  t ) ;

  #else

    //////////////////////////////////////////////////////
    // Unix or Linux: use resource usage
    //////////////////////////////////////////////////////

    struct rusage t;
    double procTime;

    /// (1) Get the rusage data structure at this moment (man getrusage)
    getrusage(0,&t);

    // (2) What is the elapsed time?
    //     - CPU time = User time + System time
 
    // (2a) Get the seconds
    procTime = t.ru_utime.tv_sec + t.ru_stime.tv_sec;
    
    // (2b) More precisely! Get the microseconds part!
    return ( procTime + (t.ru_utime.tv_usec + t.ru_stime.tv_usec) * 1e-6 ) ;

  #endif
}
