#ifndef INDEX_H
#define INDEX_H

#include <vector>
#include <set>

typedef unsigned int       Index;
typedef std::vector<Index> IndexVector;
typedef std::set<Index>    IndexSet;

typedef int                SignedIndex;
typedef std::vector<Index> SignedIndexVector;
typedef std::set<Index>    SignedIndexSet;




class IndexPair{
  friend std::ostream &operator<<(std::ostream &output, const IndexPair&);
  friend std::istream &operator>>(std::istream &input,        IndexPair&);
 private:
  Index pair[2];
 public:
  IndexPair() { pair[0] = 0; pair[1] = 0;};
  IndexPair(Index a,Index b) { pair[0] = a; pair[1] = b;}
  
  const Index& A() const {return pair[0];}
        Index& A()       {return pair[0];}
  const Index& B() const {return pair[1];}
        Index& B()       {return pair[1];}

  const Index& operator[](unsigned int i) const{ return pair[i]; }
        Index& operator[](unsigned int i)      { return pair[i]; }
};


class IndexTriple{
  friend std::ostream &operator<<(std::ostream &output, const IndexTriple&);
  friend std::istream &operator>>(std::istream &input,        IndexTriple&);
 private:
  Index triple[3];
 public:
  IndexTriple() { triple[0] = 0; triple[1] = 0; triple[2] = 0;}
  IndexTriple(Index a,Index b,Index c) { triple[0] = a; triple[1] = b; triple[2] = c;}
  
  const Index& A() const {return triple[0];}
        Index& A()       {return triple[0];}
  const Index& B() const {return triple[1];}
        Index& B()       {return triple[1];}
  const Index& C() const {return triple[2];}
        Index& C()       {return triple[2];}

  const Index& operator[](unsigned int i) const{ return triple[i]; }
        Index& operator[](unsigned int i)      { return triple[i]; }
};


class IndexQuadruple{
  friend std::ostream &operator<<(std::ostream &output, const IndexQuadruple&);
  friend std::istream &operator>>(std::istream &input,        IndexQuadruple&);
 private:
  Index quadruple[4];
 public:
  IndexQuadruple() { quadruple[0] = 0; quadruple[1] = 0; quadruple[2] = 0; quadruple[3] = 0;}
  IndexQuadruple(Index a,Index b,Index c,Index d) { quadruple[0] = a; quadruple[1] = b; quadruple[2] = c; quadruple[3] = d;}
  
  const Index& A() const {return quadruple[0];}
        Index& A()       {return quadruple[0];}
  const Index& B() const {return quadruple[1];}
        Index& B()       {return quadruple[1];}
  const Index& C() const {return quadruple[2];}
        Index& C()       {return quadruple[2];}
  const Index& D() const {return quadruple[3];}
        Index& D()       {return quadruple[3];}

  const Index& operator[](unsigned int i) const{ return quadruple[i]; }
        Index& operator[](unsigned int i)      { return quadruple[i]; }
  
};


class SignedIndexQuadruple{
  friend std::ostream &operator<<(std::ostream &output, const SignedIndexQuadruple&);
  friend std::istream &operator>>(std::istream &input,        SignedIndexQuadruple&);
 private:
  SignedIndex quadruple[4];
 public:
  SignedIndexQuadruple() { quadruple[0] = 0; quadruple[1] = 0; quadruple[2] = 0; quadruple[3] = 0;}
  SignedIndexQuadruple(SignedIndex a,SignedIndex b,SignedIndex c,SignedIndex d) { quadruple[0] = a; quadruple[1] = b; quadruple[2] = c; quadruple[3] = d;}
  
  const SignedIndex& A() const {return quadruple[0];}
        SignedIndex& A()       {return quadruple[0];}
  const SignedIndex& B() const {return quadruple[1];}
        SignedIndex& B()       {return quadruple[1];}
  const SignedIndex& C() const {return quadruple[2];}
        SignedIndex& C()       {return quadruple[2];}
  const SignedIndex& D() const {return quadruple[3];}
        SignedIndex& D()       {return quadruple[3];}

  const SignedIndex& operator[](unsigned int i) const{ return quadruple[i]; }
        SignedIndex& operator[](unsigned int i)      { return quadruple[i]; }
  
};




inline std::ostream &operator<<(std::ostream &output, const IndexPair& t){
  output << t.A() << " " << t.B();
  return output;
}

inline std::istream &operator>>(std::istream &input,        IndexPair& t){
  input >> t.A() >> t.B();
  return input;
}



inline std::ostream &operator<<(std::ostream &output, const IndexTriple& t){
  output << t.A() << " " << t.B() << " " << t.C();
  return output;
}

inline std::istream &operator>>(std::istream &input,        IndexTriple& t){
  input >> t.A() >> t.B() >> t.C();
  return input;
}



inline std::ostream &operator<<(std::ostream &output, const IndexQuadruple& t){
  output << t.A() << " " << t.B() << " " << t.C() << " " << t.D();
  return output;
}

inline std::istream &operator>>(std::istream &input,        IndexQuadruple& t){
  input >> t.A() >> t.B() >> t.C() >> t.D();
  return input;
}




inline std::ostream &operator<<(std::ostream &output, const SignedIndexQuadruple& t){
  output << t.A() << " " << t.B() << " " << t.C() << " " << t.D();
  return output;
}

inline std::istream &operator>>(std::istream &input,        SignedIndexQuadruple& t){
  input >> t.A() >> t.B() >> t.C() >> t.D();
  return input;
}


#endif
