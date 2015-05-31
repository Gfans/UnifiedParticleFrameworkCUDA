#ifndef STOPWATCH_H
#define STOPWATCH_H

#include <iostream>
#include <cstdlib>
#include <string>
#include <sstream>

#include <sys/timeb.h>

class Stopwatch{
  friend std::ostream &operator<<(std::ostream&, const Stopwatch&);
 private:
  float saved_time;
  struct timeb start_time;
  struct timeb finish_time;
 public:
  Stopwatch() { reset();}


  void start(){
    saved_time = 0.0;
    ftime(&start_time);
  }
  void reset(){
    saved_time = 0.0;
    ftime(&start_time);
  }
  void stop(){
    ftime(&finish_time);
    saved_time += (finish_time.time - start_time.time) + (finish_time.millitm - start_time.millitm)/ 1000.0F; 
    start_time = finish_time;
  }
  float time() const{
    return lap();
  }
  float lap() const{
    struct timeb now;
    ftime(&now);
    return saved_time + (now.time - start_time.time) + (now.millitm - start_time.millitm)/ 1000.0F;
  }
  std::string toString() const{
    std::ostringstream s;
    s << time();
    return s.str();
  }
};


inline std::ostream &operator<<(std::ostream &output, const Stopwatch& sw){
  return output << sw.toString();
}

#endif
