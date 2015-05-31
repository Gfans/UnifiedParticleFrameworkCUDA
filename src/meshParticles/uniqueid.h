#ifndef UNIQUEID_H
#define UNIQUEID_H


#include <iostream>

class UniqueID {
  friend std::ostream& operator<<(std::ostream&,const UniqueID&);
private:
  static unsigned int next_id; 
  int uid;
public:
  UniqueID() : uid(next_id++){}
  
  bool operator==(const UniqueID& u) const { return uid == u.uid; }
  bool operator!=(const UniqueID& u) const { return uid != u.uid; }
};



inline std::ostream &operator<<(std::ostream& output, const UniqueID& u){
  return output << u.uid;
}


#endif
