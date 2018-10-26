
import java.io.File;
import java.io.IOException;


public interface CommonInterface {
	
	/**
	 * @param x the input vector. An array of doubles
	 * @Return The value returned by the LUT or NN for this input vector
	 * */
	
	public double outputFor (double [] x);
	
	
	/**this method will tell the NN or the LUT the output calculate that should be mapped to the givven inpt vector i.E the desired correct output value for an input
	 * @param X the input vector
	 * @param argVAlue the new value to learn
	 * @return the error in the output for that input vector
	 * */
	public double train (double [] x, double expectedVal);
	
	/**method to write either a LUT or weights of  a neural network to a file
	 * @param arfile of type of file 
	 * */
	
	public void save(File argFile);
	 
	/**
	 * Loads the LUT or neural net weights form file. The load must of course have knowledge of how the data was written out by the save method.
	 * you should raise an error in case that an attempt is being made to load data into an LUT or neural net whose structure does nor match thedata in the file. (e.g. wrong number of hidden neurons)
	 */
	
	
	public void load (String argFileName) throws IOException;
	
}


