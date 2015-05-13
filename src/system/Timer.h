#ifndef GPU_UNIFIED_SYSTEM_TIMER_H_
#define GPU_UNIFIED_SYSTEM_TIMER_H_

#include "System/Config.h"

#ifdef SPH_WIN32
#  define NOMINMAX
#  include <windows.h>
#else
#  include <sys/time.h>
#endif

#include <WinSock.h>

namespace sr
{
	namespace sys
	{

#ifdef SPH_WIN32
		int GetTimeOfDay(struct timeval *tv, int);
#endif

		/**
		* A simple timer
		**/
		class Timer
		{
		public:
			Timer();
			~Timer();

			void Start();
			void Stop();

			double GetElapsedTime() const;
			void PrintElapsedTime() const;

		private:
			timeval start_time_;
			timeval end_time_;
		};

		inline void Timer::Start()
		{
			GetTimeOfDay(&start_time_, 0);
		}

		inline void Timer::Stop()
		{
			GetTimeOfDay(&end_time_, 0);
		}

	}
}

#endif	// GPU_UNIFIED_SYSTEM_TIMER_H_
