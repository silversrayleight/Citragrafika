#ifndef _DATAENTRY
#define _DATAENTRY

//standard libraries
#include <iostream>
#include <vector>

using namespace std;

class dataEntry
{
public:
	
	double* pattern;	
	double* target;		

public:

	
	dataEntry(double* p, double* t): pattern(p), target(t) {}
		
	~dataEntry()
	{				
		delete[] pattern;
		delete[] target;
	}

};

#endif
