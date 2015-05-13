#include "System/Timer.h"

namespace sr
{
	namespace sys
	{

#ifdef SPH_WIN32
		int GetTimeOfDay(struct timeval *tv, int)
		{
			union
			{
				long long ns100;
				FILETIME ft;
			} now;

			GetSystemTimeAsFileTime(&now.ft);
			tv->tv_usec = (long)((now.ns100 / 10LL) % 1000000LL);
			tv->tv_sec = (long)((now.ns100 - 116444736000000000LL) / 10000000LL);
			return (0);
		}
#endif

		Timer::Timer()
		{}

		Timer::~Timer()
		{}

		double Timer::GetElapsedTime() const
		{
			double elapsed = (end_time_.tv_sec - start_time_.tv_sec) +
				(end_time_.tv_usec - start_time_.tv_usec) / 1000000.0;
			return elapsed;
		}

	}
}
