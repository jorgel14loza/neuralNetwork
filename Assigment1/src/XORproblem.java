 import java.io.*;
import java.util.Random;

public class XORproblem {

	
	
public static void main (String[] args) {
// input values and expected outputs  arrays for XOR problem
	
	double XORInput[][]={{1,0,0},{1,0,1},{1,1,0},{1,1,1}};
	double XOROutput[]= {0,1,1,0};
	
	
//Definition of  the neural network with the desired number inputs, outputs, neurons, learning rate, momentum  and random range for weights
	
	NeuralNetwork neuralNet = new NeuralNetwork(3, 4, .2, 0, 0, 1);
	
       //this method clear the weights to  zero value
		neuralNet.zeroWeights();
		// This method initialize the weights with random values from -.5 to .5
		neuralNet.initializeWeights(-.5, .5);

// definition of variables for  the Neural Net training algorithm
		
		int epoch=0;// variable to store the number of epochs
		double error=1;// variable to store the total error after each training iteration
		
		
// definition of variables for testing the training of the neural net
		double output; // variable to store the output of the NN with  for a given input
		int patNum; // variable  for storing a  random variable to input an array to the NN and test after the network is trained
		
		
		
//Training algorithm for a total error less than .05
	
		while(error>0.05) {	
		error=0;// clear error after  an iteration
		
		for (int j=0;j<4;j++) {	
			
			//  training of the neural net  with 4 for patterns  and calculation of total error after the training 
			error +=Math.pow(neuralNet.train(XORInput[j],XOROutput[j]),2);
		
		
		}
		error=error/2;//final step for the total error calculation
		epoch=1+epoch;//epoch counting
		System.out.format("%d,",epoch);
		System.out.format("%f\n",error);
	}
	
		
		
		neuralNet.printWeights();

	// printing the final number of epochs
	System.out.format("x %d\n ",epoch);
	
	
//test   algorithm  20 times with a pattern fed in a random order
	for (int i=0;i<20;i++) {	
			
		 patNum = new Random().nextInt(4);
		 
 output=neuralNet.outputFor(XORInput[patNum]);
 
System.out.format("%d ",patNum);
System.out.format("%f\n",output);
}
	
	
	
	
	
	
	/*for (int i=0; i<neuralNet.numHiddenNeurons; i++)
		{
			for (int j=0; j<neuralNet.numInputs; j++) {
			
			
			System.out.format("for the input %d at  the neuron %d,  Weight is : %f%n",j, i, neuralNet.inputWeights[i][j]);
			
			}
		} */
		
	
}

	

 }
	
	
