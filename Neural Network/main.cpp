
#include <iostream>
#include <sstream>

#include "dataEntry.h"
#include "dataReader.h"
#include "neuralNetwork.h"


using namespace std;

void main()
{	
	
	dataReader d;

	
	d.loadDataFile("vowel-recognition.csv",16,1);
	d.setCreationApproach( STATIC );

	
	neuralNetwork nn(16, 8, 1);
	nn.enableLogging("trainingResults.csv");
	nn.setLearningParameters(0.01, 0.8);
	nn.setDesiredAccuracy(100);
	nn.setMaxEpochs(200);
	
	//dataset
	dataSet* dset;		

	for ( int i=0; i < d.nDataSets(); i++ )
	{
		dset = d.getDataSet();	
		nn.trainNetwork( dset->trainingSet, dset->generalizationSet, dset->validationSet );
	}	
	
	cout << "-- END OF PROGRAM --" << endl;
	char c; cin >> c;
}
