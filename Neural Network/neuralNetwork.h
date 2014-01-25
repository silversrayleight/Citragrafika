#ifndef NNETWORK
#define NNETWORK

//standard libraries
#include <math.h>
#include <ctime>
#include <vector>
#include <fstream>
#include <sstream>

//custom includes
#include "dataEntry.h"

using namespace std;

//Konstanta
#define LEARNING_RATE 0.001
#define MOMENTUM 0.9
#define MAX_EPOCHS 20000
#define DESIRED_ACCURACY 100  

/*******************************************************************************************************************
 *	NEURAL NETWORK CLASS
 *	----------------------------------------------------------------------------------------------------------------
 *	Classic Back-propagation Neural Network 
 *	menggunakan stochastic dan batch learning
 *******************************************************************************************************************/

class neuralNetwork
{

//private members

private:

	//learning parameters
	double learningRate;
	double momentum;	

	//jumlah neurons
	int nInput, nHidden, nOutput;
	
	//neurons
	double* inputNeurons;
	double* hiddenNeurons;
	double* outputNeurons;

	//weights
	double** wInputHidden;
	double** wHiddenOutput;

	//epoch counter
	long epoch;
	long maxEpochs;
	
	//akurasi
	double desiredAccuracy;

	//perubahan weights
	double** deltaInputHidden;
	double** deltaHiddenOutput;

	//error gradients
	double* hiddenErrorGradients;
	double* outputErrorGradients;

	//accuracy  per epoch
	double trainingSetAccuracy;
	double validationSetAccuracy;
	double generalizationSetAccuracy;
	double trainingSetMSE;
	double validationSetMSE;
	double generalizationSetMSE;

	//batch learning flag
	bool useBatch;

	//log file handle
	bool logResults;
	fstream logFile;
	int logResolution;
	int lastEpochLogged;

//public methods
//----------------------------------------------------------------------------------------------------------------
public:

	//constructor
	neuralNetwork(int in, int hidden, int out) : nInput(in), nHidden(hidden), nOutput(out), epoch(0), logResults(false), logResolution(10), lastEpochLogged(-10), trainingSetAccuracy(0), validationSetAccuracy(0), generalizationSetAccuracy(0), trainingSetMSE(0), validationSetMSE(0), generalizationSetMSE(0)
	{				
		// neuron lists
		//--------------------------------------------------------------------------------------------------------
		inputNeurons = new( double[in + 1] );
		for ( int i=0; i < in; i++ ) inputNeurons[i] = 0;

		// bias neuron
		inputNeurons[in] = -1;

		hiddenNeurons = new( double[hidden + 1] );
		for ( int i=0; i < hidden; i++ ) hiddenNeurons[i] = 0;

		// bias neuron
		hiddenNeurons[hidden] = -1;

		outputNeurons = new( double[out] );
		for ( int i=0; i < out; i++ ) outputNeurons[i] = 0;

		//membuat weight lists
		//--------------------------------------------------------------------------------------------------------
		wInputHidden = new( double*[in + 1] );
		for ( int i=0; i <= in; i++ ) 
		{
			wInputHidden[i] = new (double[hidden]);
			for ( int j=0; j < hidden; j++ ) wInputHidden[i][j] = 0;		
		}

		wHiddenOutput = new( double*[hidden + 1] );
		for ( int i=0; i <= hidden; i++ ) 
		{
			wHiddenOutput[i] = new (double[out]);			
			for ( int j=0; j < out; j++ ) wHiddenOutput[i][j] = 0;		
		}

		//membuat delta lists
		deltaInputHidden = new( double*[in + 1] );
		for ( int i=0; i <= in; i++ ) 
		{
			deltaInputHidden[i] = new (double[hidden]);
			for ( int j=0; j < hidden; j++ ) deltaInputHidden[i][j] = 0;		
		}

		deltaHiddenOutput = new( double*[hidden + 1] );
		for ( int i=0; i <= hidden; i++ ) 
		{
			deltaHiddenOutput[i] = new (double[out]);			
			for ( int j=0; j < out; j++ ) deltaHiddenOutput[i][j] = 0;		
		}

		//buat error gradient storage
		hiddenErrorGradients = new( double[hidden + 1] );
		for ( int i=0; i <= hidden; i++ ) hiddenErrorGradients[i] = 0;
		
		outputErrorGradients = new( double[out + 1] );
		for ( int i=0; i <= out; i++ ) outputErrorGradients[i] = 0;
		
		//inisialisasi weights
		initializeWeights();

		//default learning parameters

		learningRate = LEARNING_RATE; 
		momentum = MOMENTUM; 

		//penggunaan stochastic learning secara default
		useBatch = false;
		
		//stop conditions
		
		maxEpochs = MAX_EPOCHS;
		desiredAccuracy = DESIRED_ACCURACY;			
	}

	//destructor
	~neuralNetwork()
	{
		//hapus neurons
		delete[] inputNeurons;
		delete[] hiddenNeurons;
		delete[] outputNeurons;

		//haspus weight storage
		for (int i=0; i <= nInput; i++) delete[] wInputHidden[i];
		delete[] wInputHidden;

		for (int j=0; j <= nHidden; j++) delete[] wHiddenOutput[j];
		delete[] wHiddenOutput;

		//Hapus delta storage
		for (int i=0; i <= nInput; i++) delete[] deltaInputHidden[i];
		delete[] deltaInputHidden;

		for (int j=0; j <= nHidden; j++) delete[] deltaHiddenOutput[j];
		delete[] deltaHiddenOutput;

		//hapus error gradients
		delete[] hiddenErrorGradients;
		delete[] outputErrorGradients;

		//tutup log file
		if ( logFile.is_open() ) logFile.close();
	}

	//set learning parameters
	void setLearningParameters(double lr, double m)
	{
		learningRate = lr;		
		momentum = m;
	}

	//set max epoch
	void setMaxEpochs(int max)
	{
		maxEpochs = max;
	}

	//set akurasi 
	void setDesiredAccuracy(float d)
	{
		desiredAccuracy = d;
	}

	// batch learning
	void useBatchLearning()
	{
		useBatch = true;
	}

	// stochastic learning
	void useStochasticLearning()
	{
		useBatch = false;
	}

	//logging  training results
	void enableLogging(const char* filename, int resolution = 1)
	{
		//buat log file 
		if ( ! logFile.is_open() )
		{
			logFile.open(filename, ios::out);

			if ( logFile.is_open() )
			{
				//catat log file header
				logFile << "Epoch,Training Set Accuracy, Generalization Set Accuracy,Training Set MSE, Generalization Set MSE" << endl;
				
				// logging
				logResults = true;
				
				//resolution setting;
				logResolution = resolution;
				lastEpochLogged = -resolution;
			}
		}
	}

	//reset neural network
	void resetWeights()
	{
		//atur ulang weights
		initializeWeights();		
	}

	//proses data melalui network
	double* feedInput( double* inputs)
	{
		//masukan data ke dalam network
		feedForward(inputs);

		//return result
		return outputNeurons;
	}

	//train  network
	void trainNetwork( vector<dataEntry*> trainingSet, vector<dataEntry*> generalizationSet, vector<dataEntry*> validationSet )
	{
		cout<< endl << " Neural Network Training proses: " << endl
			<< "==========================================================================" << endl
			<< " LR: " << learningRate << ", Momentum: " << momentum << ", Max Epochs: " << maxEpochs << endl
			<< " " << nInput << " Neuron masukan, " << nHidden << " Hidden Neurons, " << nOutput << " Output Neurons" << endl
			<< "==========================================================================" << endl << endl;

		//reset epoch dan log counters
		epoch = 0;
		lastEpochLogged = -logResolution;
			
		//train network menggunakan training dataset untuk training dan generalisasi dataset untuk testing
		//--------------------------------------------------------------------------------------------------------
		while (	( trainingSetAccuracy < desiredAccuracy || generalizationSetAccuracy < desiredAccuracy ) && epoch < maxEpochs )				
		{			
			//simpan accuracy sebelumnya
			double previousTAccuracy = trainingSetAccuracy;
			double previousGAccuracy = generalizationSetAccuracy;

			//gunakan training set untuk proses train network
			runTrainingEpoch( trainingSet );

			// generalisasi set accuracy dan MSE
			generalizationSetAccuracy = getSetAccuracy( generalizationSet );
			generalizationSetMSE = getSetMSE( generalizationSet );

			//hasil Log Training 
			if (logResults && logFile.is_open() && ( epoch - lastEpochLogged == logResolution ) ) 
			{
				logFile << epoch << "," << trainingSetAccuracy << "," << generalizationSetAccuracy << "," << trainingSetMSE << "," << generalizationSetMSE << endl;
				lastEpochLogged = epoch;
			}
			
			//print out perubahan training /generalisasi akurasi
			if ( ceil(previousTAccuracy) != ceil(trainingSetAccuracy) || ceil(previousGAccuracy) != ceil(generalizationSetAccuracy) ) 
			{	
				cout << "Epoch :" << epoch;
				cout << " TSet Acc:" << trainingSetAccuracy << "%, MSE: " << trainingSetMSE ;
				cout << " GSet Acc:" << generalizationSetAccuracy << "%, MSE: " << generalizationSetMSE << endl;				
			}
			
			//setelah training set selesai increment epoch
			epoch++;

		}//end while

		//catch validation set accuracy dan MSE
		validationSetAccuracy = getSetAccuracy(validationSet);
		validationSetMSE = getSetMSE(validationSet);

		//log end
		logFile << epoch << "," << trainingSetAccuracy << "," << generalizationSetAccuracy << "," << trainingSetMSE << "," << generalizationSetMSE << endl << endl;
		logFile << "Training selesai!!! - >  Epochs yang dilakukan: " << epoch << " Validation Set Accuracy: " << validationSetAccuracy << " Validation Set MSE: " << validationSetMSE << endl;
				
		//hasil validation accuracy dan MSE
		cout << endl << "Training selesai!!! - > Epochs yang dilakukan: " << epoch << endl;
		cout << " Validation Set Accuracy: " << validationSetAccuracy << endl;
		cout << " Validation Set MSE: " << validationSetMSE << endl << endl;
	}
	
	
//private methods
private:

	//inisialisasi weights dan perubahan weight 
	void initializeWeights()
	{
		//init random number generator
		srand( (unsigned int) time(0) );
			
		//set weights diantara input dan hidden ke random value antara -05 dan 0.5
		//--------------------------------------------------------------------------------------------------------
		for(int i = 0; i <= nInput; i++)
		{		
			for(int j = 0; j < nHidden; j++) 
			{
				//set weights ke random values
				wInputHidden[i][j] = (double)rand() / (RAND_MAX + 1) - 0.5;

				//buat blank delta
				deltaInputHidden[i][j] = 0;
			}
		}
		
		//set weights antara input dan hidden ke random value antara -05 and 0.5
		//--------------------------------------------------------------------------------------------------------
		for(int i = 0; i <= nHidden; i++)
		{		
			for(int j = 0; j < nOutput; j++) 
			{
				//set weights ke random values
				wHiddenOutput[i][j] = (double)rand() / (RAND_MAX + 1) - 0.5;

				//buat blank delta
				deltaHiddenOutput[i][j] = 0;
			}
		}
	}

	//jalankan single training epoch
	void runTrainingEpoch( vector<dataEntry*> trainingSet )
	{
		// patterns yang salah
		double incorrectPatterns = 0;
		double mse = 0;
			
		//untuk setiap training pattern
		for ( int tp = 0; tp < (int) trainingSet.size(); tp++)
		{						
			//masukan inputs ke network dan backpropagate errors
			feedForward( trainingSet[tp]->pattern );
			backpropagate( trainingSet[tp]->target );	

			//pattern correct flag
			bool patternCorrect = true;

			//cek semua outputs dari neural network dengan nilai yang diinginkan
			for ( int k = 0; k < nOutput; k++ )
			{					
				//pattern salah bila diinginkan dan output differ
				if ( getRoundedOutputValue( outputNeurons[k] ) != trainingSet[tp]->target[k] ) patternCorrect = false;
				
				//hitung MSE
				mse += pow((outputNeurons[k] - trainingSet[tp]->target[k]), 2);
			}
			
			//apabila pattern salah dan hitung kesalahan
			if ( !patternCorrect ) incorrectPatterns++;	
			
		}//end for

		//jika menggunakan batch learning - lakukan update weights
		if ( useBatch ) updateWeights();
		
		//update training accuracy dan MSE
		trainingSetAccuracy = 100 - (incorrectPatterns/trainingSet.size() * 100);
		trainingSetMSE = mse / ( nOutput * trainingSet.size() );
	}

	//masukan input forward
	void feedForward( double *inputs )
	{
		//set input neuron ke input values
		for(int i = 0; i < nInput; i++) inputNeurons[i] = inputs[i];
		
		//hitung nilai Hidden Layer  - termasuk bias neuron
		//--------------------------------------------------------------------------------------------------------
		for(int j=0; j < nHidden; j++)
		{
			//clear value
			hiddenNeurons[j] = 0;				
			for( int i=0; i <= nInput; i++ ) hiddenNeurons[j] += inputNeurons[i] * wInputHidden[i][j];
			hiddenNeurons[j] = activationFunction( hiddenNeurons[j] );			
		}
		for(int k=0; k < nOutput; k++)
		{
			outputNeurons[k] = 0;				
			
			for( int j=0; j <= nHidden; j++ ) outputNeurons[k] += hiddenNeurons[j] * wHiddenOutput[j][k];
			outputNeurons[k] = activationFunction( outputNeurons[k] );
		}
	}
void backpropagate( double* desiredValues )
	{		
		for (int k = 0; k < nOutput; k++)
		{
			outputErrorGradients[k] = getOutputErrorGradient( desiredValues[k], outputNeurons[k] );
			
			for (int j = 0; j <= nHidden; j++) 
			{				
				if ( !useBatch ) deltaHiddenOutput[j][k] = learningRate * hiddenNeurons[j] * outputErrorGradients[k] + momentum * deltaHiddenOutput[j][k];
				else deltaHiddenOutput[j][k] += learningRate * hiddenNeurons[j] * outputErrorGradients[k];
			}
		}

		for (int j = 0; j < nHidden; j++)
		{
			hiddenErrorGradients[j] = getHiddenErrorGradient( j );

			for (int i = 0; i <= nInput; i++)
			{
				if ( !useBatch ) deltaInputHidden[i][j] = learningRate * inputNeurons[i] * hiddenErrorGradients[j] + momentum * deltaInputHidden[i][j];
				else deltaInputHidden[i][j] += learningRate * inputNeurons[i] * hiddenErrorGradients[j]; 

			}
		}
		
		if ( !useBatch ) updateWeights();
	}

	void updateWeights()
	{
		for (int i = 0; i <= nInput; i++)
		{
			for (int j = 0; j < nHidden; j++) 
			{
				wInputHidden[i][j] += deltaInputHidden[i][j];	
				
				if (useBatch) deltaInputHidden[i][j] = 0;				
			}
		}
		
		for (int j = 0; j <= nHidden; j++)
		{
			for (int k = 0; k < nOutput; k++) 
			{					
				wHiddenOutput[j][k] += deltaHiddenOutput[j][k];
				
				if (useBatch)deltaHiddenOutput[j][k] = 0;
			}
		}
	}

	inline double activationFunction( double x )
	{
		return 1/(1+exp(-x));
	}

	inline double getOutputErrorGradient(double desiredValue, double outputValue)
	{
		return outputValue * ( 1 - outputValue ) * ( desiredValue - outputValue );
	}

	double getHiddenErrorGradient( int j )
	{
		double weightedSum = 0;
		for( int k = 0; k < nOutput; k++ ) weightedSum += wHiddenOutput[j][k] * outputErrorGradients[k];

		return hiddenNeurons[j] * ( 1 - hiddenNeurons[j] ) * weightedSum;
	}

	int getRoundedOutputValue( double x )
	{
		if ( x < 0.1 ) return 0;
		else if ( x > 0.9 ) return 1;
		else return -1;
	}	
	double getSetAccuracy( vector<dataEntry*> set )
	{
		double incorrectResults = 0;
		
		for ( int tp = 0; tp < (int) set.size(); tp++)
		{						
			feedForward( set[tp]->pattern );
			
			bool correctResult = true;

			for ( int k = 0; k < nOutput; k++ )
			{					
				if ( getRoundedOutputValue(outputNeurons[k]) != set[tp]->target[k] ) correctResult = false;
			}
			
			if ( !correctResult ) incorrectResults++;	
			
		}//end for
		return 100 - (incorrectResults/set.size() * 100);
	}

	double getSetMSE ( vector<dataEntry*> set )
	{
		double mse = 0;
		
		for ( int tp = 0; tp < (int) set.size(); tp++)
		{						
			feedForward( set[tp]->pattern );
			
			for ( int k = 0; k < nOutput; k++ )
			{					
				mse += pow((outputNeurons[k] - set[tp]->target[k]), 2);
			}		
			
		}return mse/(nOutput * set.size());
	}
};

#endif
