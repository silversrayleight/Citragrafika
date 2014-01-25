#ifndef _DATAREADER
#define _DATAREADER


#include <vector>
#include <fstream>
#include <string>
#include <math.h>
#include <algorithm>

#include "dataEntry.h"

using namespace std;

enum { NONE, STATIC, GROWING, WINDOWING };

class dataSet
{
public:

	vector<dataEntry*> trainingSet;
	vector<dataEntry*> generalizationSet;
	vector<dataEntry*> validationSet;

	dataSet()
	{		
	}

	~dataSet()
	{
		trainingSet.clear();
		generalizationSet.clear();
		validationSet.clear();
	}

	void clear()
	{
		trainingSet.clear();
		generalizationSet.clear();
		validationSet.clear();
	}
};

class dataReader
{
	

private:

	vector<dataEntry*> data;
	int nInputs;
	int nTargets;
	dataSet dSet;
	int creationApproach;
	int numDataSets;
	int trainingDataEndIndex;

	double growingStepSize;			
	int growingLastDataIndex;		
	int windowingSetSize;			
	int windowingStepSize;			
	int windowingStartIndex;		
	
public:

	dataReader(): creationApproach(NONE), numDataSets(-1)
	{				
	}
	~dataReader()
	{
		for (int i=0; i < (int) data.size(); i++ ) delete data[i];		
		data.clear();		 
	}

	bool loadDataFile( const char* filename, int nI, int nT )
	{
		for (int i=0; i < (int) data.size(); i++ ) delete data[i];		
		data.clear();
		dSet.clear();
		nInputs = nI;
		nTargets = nT;

		fstream inputFile;
		inputFile.open(filename, ios::in);	

		if ( inputFile.is_open() )
		{
			string line = "";
			
			while ( !inputFile.eof() )
			{
				getline(inputFile, line);				
				
				if (line.length() > 2 ) processLine(line);
			}		
			
			random_shuffle(data.begin(), data.end());

			trainingDataEndIndex = (int) ( 0.6 * data.size() );
			int gSize = (int) ( ceil(0.2 * data.size()) );
			int vSize = (int) ( data.size() - trainingDataEndIndex - gSize );
								
			for ( int i = trainingDataEndIndex; i < trainingDataEndIndex + gSize; i++ ) dSet.generalizationSet.push_back( data[i] );
					
			for ( int i = trainingDataEndIndex + gSize; i < (int) data.size(); i++ ) dSet.validationSet.push_back( data[i] );
			
			cout << "Data File Read Complete >> Patterns Loaded: " << data.size() << endl;

			inputFile.close();
			
			return true;
		}
		else return false;
	}

	void setCreationApproach( int approach, double param1 = -1, double param2 = -1 )
	{
		if ( approach == STATIC )
		{
			creationApproach = STATIC;
			
			numDataSets = 1;
		}

		else if ( approach == GROWING )
		{			
			if ( param1 <= 100.0 && param1 > 0)
			{
				creationApproach = GROWING;
			
				growingStepSize = param1 / 100;
				growingLastDataIndex = 0;

				numDataSets = (int) ceil( 1 / growingStepSize );				
			}
		}

		else if ( approach == WINDOWING )
		{
			if ( param1 < data.size() && param2 <= param1)
			{
				creationApproach = WINDOWING;
				
				windowingSetSize = (int) param1;
				windowingStepSize = (int) param2;
				windowingStartIndex = 0;			

				numDataSets = (int) ceil( (double) ( trainingDataEndIndex - windowingSetSize ) / windowingStepSize ) + 1;
			}			
		}

	}

	int nDataSets()
	{
		return numDataSets;
	}
	
	dataSet* getDataSet()
	{		
		switch ( creationApproach )
		{
		
			case STATIC : createStaticDataSet(); break;
			case GROWING : createGrowingDataSet(); break;
			case WINDOWING : createWindowingDataSet(); break;
		}
		
		return &dSet;
	}

private:
	
	void createStaticDataSet()
	{
		for ( int i = 0; i < trainingDataEndIndex; i++ ) dSet.trainingSet.push_back( data[i] );		
	}

	void createGrowingDataSet()
	{
		growingLastDataIndex += (int) ceil( growingStepSize * trainingDataEndIndex );		
		if ( growingLastDataIndex > (int) trainingDataEndIndex ) growingLastDataIndex = trainingDataEndIndex;

		dSet.trainingSet.clear();
		
		for ( int i = 0; i < growingLastDataIndex; i++ ) dSet.trainingSet.push_back( data[i] );			
	}

	void createWindowingDataSet()
	{
		int endIndex = windowingStartIndex + windowingSetSize;
		if ( endIndex > trainingDataEndIndex ) endIndex = trainingDataEndIndex;		

		dSet.trainingSet.clear();
						
		for ( int i = windowingStartIndex; i < endIndex; i++ ) dSet.trainingSet.push_back( data[i] );
				
		windowingStartIndex += windowingStepSize;
	}
	
	void processLine( string &line )
	{
		double* pattern = new double[nInputs];
		double* target = new double[nTargets];
		
		char* cstr = new char[line.size()+1];
		char* t;
		strcpy(cstr, line.c_str());

		int i = 0;
		t=strtok (cstr,",");
		
		while ( t!=NULL && i < (nInputs + nTargets) )
		{	
			if ( i < nInputs ) pattern[i] = atof(t);
			else target[i - nInputs] = atof(t);

			t = strtok(NULL,",");
			i++;			
		}
		
		data.push_back( new dataEntry( pattern, target ) );		
	}
};

#endif
