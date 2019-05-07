// Authors
// Stavropoulos Petros (AM : 3150230)
// Savvidis Konstantinos (AM : 3150229)
// Mpanakos Vasileios (AM : 3140125)

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;

// Class that represents a Naive Bayes Classifier
// The classifier is first initialized, making a bernoulli values table of all the mails and features + the last column
// in which is the class category (Spam/Ham), and a 0/1 value for existence of the feature or not -- and 0 for ham,1 for spam
// for the last column
// The classifier is then trained, at which stage it fills an array of propabilities. The array has 2 rows and such columns as
// the features list size, as for each values it holds in the first row the value for P(X=1|C=0) and P(X=1|C=1) for the second row
// The trained classifier then evaluates each Mail by finding the probabilities of P(C=1|X) and P(C=0|X) using the probability table
// and P(C=1), P(C=0) accordingly by calculating the formula for Naive Bayes
// If P(C=1|X)>P(C=0|X) then the mail is spam, else it is ham
public class NaiveBayesClassifier {
	
	// The list of all the features
	private ArrayList<Feature> features;
	
	// The table that holds the bernoulli values for each mail and feature
	private int[][] bernoulliValuesTable;
	
	// The table that holds the trained probabilities for each feature
	private double[][] probabilityTable;
	
	// The number of spam mails
	private int countSpam;
	
	// Parametrized Constructor -- only feature list and size
	public NaiveBayesClassifier(ArrayList<Mail> trainMails,ArrayList<Feature> features) {
		this.features = features;
		bernoulliValuesTable = Main.findBernoulliTable(trainMails, features);
		probabilityTable = new double[2][features.size()+1];
	}
	
	// Method that train the Naive Bayes classifier using the bernoulliValuesTable in order to fill the probability table for each
	// feature, so that the classifier can evaluate emails using these probabilities
	public void trainNaiveBayes() {
		
		// For each feature Xi we must find P(X=1|C=1) and P(X=1|C=0)
		
		// Create a variable for the bernoulliValues table rows
		int r = bernoulliValuesTable.length;
		// Create a variable for the bernoulliValues table columns
		int c = bernoulliValuesTable[0].length;
		
		// For j from 0 to the length of the columns in the bernoulli values table
		for(int j=0;j<c;j++) {
			// Initialize a counter for X=1 and C=1 and set it to 0
			int countX1C1 = 0;
			// Initialize a counter for X=0 and C=1 and set it to 0
			int countX0C1 = 0;
			// Initialize a counter for X=1 and C=0 and set it to 0
			int countX1C0 = 0;
			// Initialize a counter for X=0 and C=0 and set it to 0
			int countX0C0 = 0;
			// For i from 0 to the length of the rows in the bernoulli values table
			for(int i=0;i<r;i++) {
				// If the value in the last column of row i is 1 then C=1
				if(bernoulliValuesTable[i][c-1]==1) {
					// If the value in indexes i,j is 1 then X=1, so increment counter for X=1,C=1
					if(bernoulliValuesTable[i][j]==1) countX1C1++;
					// Else X=0, so increment counter for X=0,C=1
					else countX0C1++;
				}else {
					// Else the value in the last column of row i is 0, so C=0
					// If the value in indexes i,j is 1 then X=1, so increment counter for X=1,C=0
					if(bernoulliValuesTable[i][j]==1) countX1C0++;
					// Else X=0, so increment counter for X=0,C=0
					else countX0C0++;
				}
			}
			// Calculate the probabilities P(X=1|C=0) and P(X=1|C=1) using Laplace Estimator +1 / +2
			// P(X=1|C=0):
			probabilityTable[0][j] = (countX1C0+1) / ((countX1C0+countX0C0+2) * 1.0);
			// P(X=1|C=1):
			probabilityTable[1][j] = (countX1C1+1) / ((countX1C1+countX0C1+2) * 1.0);
		}
		
		// Find the spam mail count (C=1)
		// Set the counter to 0
		countSpam = 0;
		// For i from 0 to the row size of the bernoulli values table
		for(int i=0;i<r;i++) {
			// If the last column is 1 then C=1, so increment the counter
			if(bernoulliValuesTable[i][c-1]==1) countSpam++;
		}
		
	}
	
	// Method that evaluates a Mail Object using the probabilities table and returns whether the classifier classifies it as:
	// spam - true
	// ham  - false
	public boolean evaluateNaiveBayes(Mail m) {
		
		// Get the text from the mail -- to uppercase and crop the first 9 characters ("Subject: ") as it is not important
		String text = m.getText().toUpperCase().substring(9);
		// Get the mail's words using a delimiter for all whitespace characters and add them to a hashset
		HashSet<String> words = new HashSet<>( Arrays.asList(text.split("\\s+")));
		
		// Initialize a feature vector to indicate whether 
		int[] featureVectorBernoulli = new int[features.size()];
		
		// See which features are 1 and which are 0
		// For j from 0 to the features list size
		for(int j=0;j<features.size();j++) {
			// If the set of the mail words contains the feature name in the features list index j
			// then the feature vector value for the mail is 1, else it is 0
			if(words.contains(features.get(j).name)){
				featureVectorBernoulli[j]=1;
			}else {
				featureVectorBernoulli[j]=0;
			}
		}
		
		// Calculate P(C=1)
		double PC = countSpam / (bernoulliValuesTable.length*1.0);
		
		// Initialize the probability P(C=1|X) to P(C=1)
		double probSpam = PC;
		
		// For every feature in the feature list
		for(int j=0;j<features.size();j++) {
			// If the feature Xi=1
			if(featureVectorBernoulli[j]==1) {
				// Multiply probSpam with P(Xi=1|C=1)
				probSpam *= probabilityTable[1][j];
			}else {
				// Multiply probSpam with P(Xi=0|C=1) = 1- P(Xi=1|C=1)
				probSpam *= 1-probabilityTable[1][j];
			}
		}
		
		// Initialize the probability P(C=0|X) to P(C=0)
		double probNotSpam = 1-PC;
		
		// For every feature in the feature list
		for(int j=0;j<features.size();j++) {
			// If the feature Xi=1
			if(featureVectorBernoulli[j]==1) {
				// Multiply probNotSpam with P(Xi=1|C=0)
				probNotSpam *= probabilityTable[0][j];
			}else {
				// Multiply probNotSpam with P(Xi=0|C=0) = 1- P(Xi=1|C=0)
				probNotSpam *= 1-probabilityTable[0][j];
			}
		}
		
		// Return whether the probability of the mail being spam is bigger 
		// than the probability of the mail not being spam
		return probSpam>probNotSpam;
	}
}
