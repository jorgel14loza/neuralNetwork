
import java.util.Random;
import java.io.File;
import java.io.IOException;
import java.lang.Math;


public class NeuralNetwork implements NeuralNetInterface {
	
	static int INPUTS =3 ; //two binary inputs for the XOR problem
	static int HIDDEN =4;  
	
	
	//Custom Sigmoid function limits
	private double argA;
	private double argB;

	
	// Parameters for neural network
	public int numInputs;
	public int numHiddenNeurons;
	
	
	//Learning rate and momentum terms
	public double learningRate;
	public double momentumTerm;
	
	//Input values for NN assumes a binary input for the XOR problem (for a different problem should change type of variable)
	// first index is bias=1
	public double [] inputVariables = new double [INPUTS];
	 //this.InputVariables[0]= bias;
	
	//Arrays for storing weights of inputs of the hidden layer neurons
	public double [][] inputWeights= new double [HIDDEN] [INPUTS] ;
	//  array for previous weights
	
	public static double [][] previousWeights= new double [HIDDEN][INPUTS];
	
	//array of hidden neurons without Sigmoid
	public double [] hiddenNeuronUnactivated = new double [HIDDEN];
	// Array for hidden neurons outputs
	public double[] hiddenNeuronOutputs= new double [HIDDEN];
	
	// Array for hidden neurons errors
	public double[] hiddenNeuronErrors= new double [HIDDEN];
	
	
	//Array  for the output neuron input weights
	public  double [] outputNeuronWeights= new double[HIDDEN];
	
	//Array for previous output neuron weights
	public double[] previousOutputNeuronWeights = new double[HIDDEN];
	
	// values of the ouput neuron Bias
	public double outputNeuronBiasWeight;
	public double previousOutputNeuronBiasWeight;
	
	
	//output for out neuron without sigmoid function
	public double outputNeuronUnactivated;
	// output for the out neuron
	public double outputNeuronValue;
	
	//output neuron error value
	public double outputNeuronError;
	
	//Neural network constructor
	
	public NeuralNetwork(int argNumInputs, int argNumHidden, double argLearningRate, double argMomentumTerm,  double argA1, double argB1) {
		
		argA= argA1;
		argB=argB1;
		
        
        numInputs=argNumInputs; 
        numHiddenNeurons=argNumHidden;
        momentumTerm=argMomentumTerm;
        learningRate=argLearningRate;
        zeroWeights();
	}
		
			
	//Method to evaluate a value with a Bipolar Sigmoid function       
	public double sigmoidBipolar(double x)
     {
    	 
    	 double sigmoidvalue;
    	 sigmoidvalue= 2/(1+Math.exp(-x))-1;
    	 return sigmoidvalue;
     }
     
	//Method to evaluate a value with a Bipolar Sigmoid function derivative
	
	public double sigmoidBipolarDerivative(double x) {
		
		
		double sigmoidDeriv;
		sigmoidDeriv= (1-Math.pow(sigmoidBipolar(x),2))/2;
		return sigmoidDeriv;
	}
	

	
	//Method to evaluate a value with a Binary sigmoid function 
	public double sigmoidBinary(double x)
    {
   	 
   	 double sigmoidvalue;
   	 sigmoidvalue= 1/(1+Math.exp(-x));
   	 return sigmoidvalue;
    }
     
	
	
	//Method to evaluate a value with th derivative of the Binary sigmoid function
     public double sigmoidBinaryDerivative(double x) {
    	 
    	 double sigmoidDeriv;
    	 sigmoidDeriv=sigmoidBinary(x)*(1 - sigmoidBinary(x));
    	 return sigmoidDeriv;
    	 
    	 
     } 
      
     
     //Method to evaluate a value with a custom sigmoid with baoundaries A and B
     public double customSigmoid(double x) {
    	  double customSigmoid;
    	  customSigmoid=(argB - argA) * sigmoidBipolar(x) + argA;
    	  return customSigmoid;
    	 
    	     }
     
     
	//	 Method to evaluate a value with a Custom Sigmoid derivative
     public double customSigmoidDerivative(double x) {
    	 
    	 double custSigmDerv;
    	 custSigmDerv=(1.0/(argB - argA)) * (customSigmoid(x) - argA) * (argB - customSigmoid(x));
    	 return custSigmDerv;
    	 
     }
     
     // method to clear the output and hidden neurons  Weights  to zero
     public void zeroWeights() {
    	 
    	 
    	 //hidden neurons
    	 for (int i=0; i<numHiddenNeurons; i++){
    		 
    		 
    		 for(int j=0; j<numInputs; j++) {
    			 
    			 inputWeights[i][j]=0;
    			 previousWeights[i][j]=0;
    			     			 
    		 }
    		 //output Neurons
    		 previousOutputNeuronWeights[i]=0;
    		 outputNeuronWeights[i]=0;
    	 }
    	 
    	 
     }
     
     	
     // This method to initialize neuron weights with  random values 
     
     public void initializeWeights(double min, double max) {
    	 
    	 double random;
    	
    	 
    	 for (int i=0; i<numHiddenNeurons;i++) {
    		 
    		 for(int j=0;j<numInputs;j++) {
    			 random= new Random().nextDouble();
    			 inputWeights[i][j] = min + (random*(max-min));
    			 
    			 
    		 }
    		 random= new Random().nextDouble();
    		 outputNeuronWeights[i]= min + (random*(max-min));
    		 outputNeuronBiasWeight= min + (random*(max-min));// initialize the Weight of the bias for the output neuron
    	 }
    	 
     }
     
     
     //Generates the  momentum term, learning rate for the backpropagation step
    public double deltaRuleFactors (double input, double error, double currentWeight, double previousWeight) {
      
    double momentum;
    double learning;
     
     momentum= momentumTerm*(currentWeight-previousWeight);
     learning= learningRate*error*input;///*(outputNeuronValue*(1-outputNeuronValue));/////////+++++++++++++++++++++++++++++++++++++++
     
     return (momentum+learning);
          
    }
     
   // This method updates the weights of the output  and hidden neurons  by the Backpropagation method
    public void weightUpdate () {
    	
    	//int hiddenNeuron, input;
    	double newOutputNeuronBiasWeight;
    	double [] newOutputNeuronWeights = new double [numHiddenNeurons];
    	double [][] newInputNeuronWeights = new double [numHiddenNeurons][numInputs];
    	newOutputNeuronBiasWeight=outputNeuronBiasWeight+deltaRuleFactors(1, outputNeuronError, outputNeuronBiasWeight, previousOutputNeuronBiasWeight );
    	// Weight update for the output neuron
    	
    	for (int i=0; i<numHiddenNeurons;i++) {
    		
    		newOutputNeuronWeights[i]= outputNeuronWeights[i]+ deltaRuleFactors(hiddenNeuronOutputs[i], outputNeuronError, outputNeuronWeights[i], previousOutputNeuronWeights[i]);
    		
    		
    	}
    	
    	//Weight update for the hidden neurons
     	
    	
    	for (int i=0; i<numHiddenNeurons;i++) {
    		
    		for (int j=0; j<numInputs; j++) {
    			
    			newInputNeuronWeights[i][j]=inputWeights[i][j]+ deltaRuleFactors (inputVariables[j], hiddenNeuronErrors[i],inputWeights[i][j], previousWeights[i][j]);
    		}
    		
    	}
    	
    	previousOutputNeuronBiasWeight = outputNeuronBiasWeight;
    	previousOutputNeuronWeights= outputNeuronWeights; 
    	previousWeights=inputWeights;
    	
    	outputNeuronBiasWeight = newOutputNeuronBiasWeight;
    	outputNeuronWeights= newOutputNeuronWeights;
    	inputWeights=newInputNeuronWeights;
    		
    }
    
    
    // This method  stores the output for the hidden and output neurons in an array
    public double outputFor (double [] inputs) {
    
    
    	inputVariables=inputs;
    
    	//calculate hidden neurons outputs bias is the first index of the input matrix
    	for (int i=0; i<numHiddenNeurons; i++) { 
    		
    		hiddenNeuronUnactivated[i]=0;
    		
    		for(int j=0; j<numInputs; j++) {
    		
    		hiddenNeuronUnactivated[i]+=inputWeights[i][j]*inputVariables[j];
    		}
    		
    		
    	hiddenNeuronOutputs[i]=sigmoidBinary(hiddenNeuronUnactivated[i]);/////////////////////////////////////// Should be modified for bipolar or binary approach
    	}
    	
    	
    	outputNeuronUnactivated = 0.0;
    	// calculation of the Output for the out neuron
    	for (int i=0; i<numHiddenNeurons; i++) {
    		
    		outputNeuronUnactivated +=hiddenNeuronOutputs[i]* outputNeuronWeights[i];
    		
    	    }
    	
    	//add bias
    	outputNeuronUnactivated +=(1*outputNeuronBiasWeight);
    	//apply activation function to  the out neuron
    	
    	outputNeuronValue= sigmoidBinary(outputNeuronUnactivated);/////////////////////////////////////////// Should be modified for bipolar or binary approach
    	
    	return outputNeuronValue;
    	
    
    }
    
    
    
    
    
    public void calculateErrors(double expectedValue) {
    	
    	outputNeuronError= (expectedValue-outputNeuronValue)*sigmoidBinaryDerivative(outputNeuronUnactivated);////////////////////////// Should be modified for bipolar or binary approach
    	
    	//error backpropagated
    	
    	for (int i=0;i<numHiddenNeurons; i++) {
    	
    	hiddenNeuronErrors[i]= outputNeuronError*outputNeuronWeights[i]*sigmoidBinaryDerivative(hiddenNeuronUnactivated[i]);//////////////////////// Should be modified for bipolar or binary approach
    	}
    
    }
    
    
    
    public void printWeights() {
    	
    	System.out.println("hidden weigths:");
    
    	
    	for(int i=0; i<numHiddenNeurons;i++) {
    		
    		for(int j=0; j<numInputs;j++) {
    			
    			System.out.format("%d %d %f\n", i, j, inputWeights[i][j]);
    			
    		}
    		
    	}
    	
    	System.out.println("Output neuron Weights:");
    	
    	for(int i=0; i<numHiddenNeurons; i++) {
    		
    		System.out.format("%d%5f\n", i, outputNeuronWeights[i]);
    	}
    		
    		
    }
    
    
    
    
    public double train(double[] inputs, double expectedVal) {
    	
    	outputFor(inputs);
    	
    	//calculate errors for the input x 
    	calculateErrors(expectedVal);
    	//Weight update
    	weightUpdate();
    	
    	//printWeights();
    	
    	return (expectedVal-outputNeuronValue);
    	
    	
    }
    
	
    
    public void save (File argFile) {
    	
    	
    }
    
    
    public void load (String argFileName) throws IOException {
    	
    }

    
    
    
    
    
}
	
	




