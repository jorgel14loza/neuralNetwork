
public interface NeuralNetInterface extends CommonInterface{

	final double bias= 1;
	//public abstract NeuralNet (int argNumInputs,int argNumHidden, double argLearningRate, double argMomentum, double argA, double argB);

	/**Returns a Bipolar sigmoid of the imput x
	 * @param x the input
	 * Returns f(x)=2/(1+e(-x)	*/
	
	
	/**public double sigmoid(double x);
	
	/** this method implements a general sigmoid with asymptotes bounded by (a,b)
	 *@param x the input 
	 *return f(x)=b-a/(1+e(-x))-1
	*/
	
	public double customSigmoid(double x);
	
	/** initialize the weight with random values
	 * for say 2 inputs the input vector is [0]&[1]. We add [2] for the bias
	 * like wise for hidden units. For say 2hidden units which are stored in an array.
	 * [0]&[1] are hidden  and [2] is the bias.
	 * We also initialize the last weight change arrays. this is to implement the alpha term. 
	 * */
	
	public void initializeWeights (double min, double max);
	
	/**Initialize the weights to zero.*/
	
	public void zeroWeights();
	
	//End of the public interface Neural NetInterface
}
