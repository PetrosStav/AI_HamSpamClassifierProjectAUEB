// Authors
// Stavropoulos Petros (AM : 3150230)
// Savvidis Konstantinos (AM : 3150229)
// Mpanakos Vasileios (AM : 3140125)

import java.util.ArrayList;
import java.util.Collections;

// Class that represents an AdaBoost classifier, which creates and trains M weak Decision Stumps with the given features, trainMails and
// a weight array using control features and giving them weighted votes and then classifies incoming Mails using the voting weight of 
// each of the M weak classifiers
public class AdaboostClassifier {

	// The list of all the features
	private ArrayList<Feature> features;
	
	// The list of the train Mails
	private ArrayList<Mail> trainMails;
	
	// The weights of the train Mails
	ArrayList<Double> w;
	
	// The number of repetitions/weak classifiers
	int M;
	
	// The array of the classifiers
	ArrayList<DecisionStump> h;
	
	// The weight of each classifier
	ArrayList<Double> z;
	
	// Parameterized Constructor
	public AdaboostClassifier(ArrayList<Mail> trainMails, ArrayList<Feature> features, int M) {
		this.features = features;
		this.trainMails = trainMails;
		this.M = M;
		h = new ArrayList<>();
		z = new ArrayList<>();
		w = new ArrayList<>();
		// The start weight is 1/N
		double start_weight = 1.0 / trainMails.size();
		// Set the start weight to all weights
		for(int i=0;i<trainMails.size();i++) {
			w.add(start_weight);
		}
	}
	
	// Method that trains AdaBoost, creating each weak classifier with it's weight
	public void trainAdaboost() {
		
		// Find the bernoulli values table outside the loop for optimization
		int [][] bernoulli = Main.findBernoulliTable(trainMails, features);
		// Set r to the bernoulli table rows
		int r = bernoulli.length;
		// Set c to the bernoulli table columns
		int c = bernoulli[0].length;
		
		// For m from 0 to the number of repetitions/weak classifiers
		for(int m=0;m<M;m++) {
			
//			System.out.println("Training " + (m+1));
			
			// P(C=1)
			double PC = 0.0;
			
			// For i from 0 to the row size of the bernoulli values table
			for(int i=0;i<r;i++) {
				// If the last column is 1 then C=1, so increment PC by the weight at index i,
				// as the weights are added to 1, so they represent the probability of P(C=1)
				// if added and the bernoulliValuesTable is a mask of which will be accounted
				if(bernoulli[i][c-1]==1) PC += w.get(i);
			}
			
			// Find the best control feature from the features using P(C=1), the bernoulli table and the weights
			Feature controlFeature = findBestFeature(PC, features, bernoulli, w);
			
			// Create a new DecisionStump
			h.add(new DecisionStump(trainMails, controlFeature, w));
			// Train the Decision Stump using the trainMails and the weights list w
			h.get(m).trainDecisionStump();
			// Initialize the error to 0
			double error = 0;
			// For i from 0 to the size of the trainMails
			for(int i=0;i<trainMails.size();i++) {
				// Evaluate the mail that has index i in trainMails by h with index m
				// and if the answer differs from the real answer then increment the
				// error by the weight of the train Mail
				if(h.get(m).evaluateDecisionStump(trainMails.get(i))!=trainMails.get(i).isSpam()) error = error + w.get(i);
			}
			// For i from 0 to the size of the trainMails
			for(int i=0;i<trainMails.size();i++) {
				// Evaluate the mail that has index i in trainMails by h with index m
				// and if the answer is the same as the real answer then modify the weight
				// of that train Mail using the error
				if(h.get(m).evaluateDecisionStump(trainMails.get(i))==trainMails.get(i).isSpam()) w.set(i, w.get(i) * (error/(1.0-error)));
			}
			// Normalize the weight list
			normalize(w);
			// Set the weight for the classifier m
			z.add(Math.log((1-error)/error)/Math.log(2));
		}
	}
	
	// Method that evaluates the class of a Mail (Spam/Ham) using the trained AdaBoost classifier
	public boolean evaluateAdaboost(Mail m) {
		
		// Initialize a boolean variable for each classifier answer
		boolean ans;
		
		// Initialize a double variable for the classifier weights
		double zWeightsVar = 0.0;
		
		// For i from 0 to the classifier list size
		for(int i = 0 ; i < h.size() ; i++) {
			// Get the answer from the classifier i and put it into the variable
			ans = h.get(i).evaluateDecisionStump(m);
			// If ans is true then increase the variable zWeightsVar by the z weight of the classifier i
			if(ans) zWeightsVar += z.get(i);
			// Else decrease the variable zWeightsVar by the z weight of the classifier i
			else zWeightsVar -= z.get(i);
		}
		
		// If zWeightsVar is positive (>0) then classify the mail as Spam
		// else classify the mail as Ham
		return zWeightsVar>0;
	}
	
	// Method that normalizes the weights so that they add to 1
	private void normalize(ArrayList<Double> weights){
		// Initialize the sum of the weights
		double sum = 0.0;
		// For every weight add it to the sum
		for(double d : weights) sum += d;
		// FOr every weight divide it by the sum
		for(int i=0;i<weights.size();i++) weights.set(i, weights.get(i)/sum);
	}
	
	// Method that finds the best feature regarding IG using the feature List, the bernoulli values, the weights and the class probability
	// Returns the best feature from the feature list
	private static Feature findBestFeature(double PC,ArrayList<Feature> features,int[][] bernoulliValues,ArrayList<Double> weights) {
		
		// Create a variable for the size of the features
		int pfeatSize = features.size();
		
		// H(C):
		double HC = Main.entropy(PC);
		
		// Create an array for the probability of existence of every possible feature P(X=1)
		double[] featureProbabilities = new double[pfeatSize];
		
		// Create an array for the probability of C depending on the existence of every possible feature P(C | X=1)
		double[] featureX1SpamProbabilities = new double[pfeatSize];
		
		// Create an array for the probability of C depending on the non-existence of every possible feature P(C | X=0)
		double[] featureX0SpamProbabilities = new double[pfeatSize];
		
		// Create an array for the entropy of H(C | X=1) for every possible feature
		double[] entropyOfFeaturesX1 = new double[pfeatSize];
		
		// Create an array for the entropy of H(C | X=0) for every possible feature
		double[] entropyOfFeaturesX0 = new double[pfeatSize];
		
		// Create an array for the Information Gain (IG) for every possible feature
		double[] informationGainOfFeatures = new double[pfeatSize];
		
		// Create an ArrayList of Feature objects
		ArrayList<Feature> possiblefeaturesObj = new ArrayList<>();
		
		// For j from 0 to the length of the size of the possible features
		for(int j=0;j<pfeatSize;j++) {
			
			// For i from 0 to the length of the rows of the bernoulliValues table
			for(int i=0;i<bernoulliValues.length;i++) {
				// If the value is 1
				if(bernoulliValues[i][j]==1) {
					// Increase the probability for X=1 by the weight at index i
					featureProbabilities[j] += weights.get(i);
					// If the value in the table for the last column is 1 (C=1) then increase the probability for X=1,C=1 by the weight at index i
					if(bernoulliValues[i][bernoulliValues[i].length-1] == 1) featureX1SpamProbabilities[j] += weights.get(i);

				}else {
					// Else X=0
					// If the value in the table for the last column is 1 (C=1) then increase the probability for X=0,C=1 by the weight at index i
					if(bernoulliValues[i][bernoulliValues[i].length-1] == 1) featureX0SpamProbabilities[j] += weights.get(i);
				}
			}
			
			// If P(X=1) == 0 then set P(C=1^X=1) to 0, so that we don't divide by 0
			if(featureProbabilities[j]==0) featureX1SpamProbabilities[j] = 0;
			// Else do the formula P(C=1^X=1)/P(X=1) for P(C=1 | X=1)
			else featureX1SpamProbabilities[j] /= featureProbabilities[j];
			// If P(X=1) == 1 then P(X=0)=0, so set P(C=1^X=0) to 0, so that we don't divide by 0
			if(featureProbabilities[j]==1) featureX0SpamProbabilities[j] = 0;
			// Else do the formula P(C=1^X=0) / P(X=0) for P(C=1 | X=0)
			else featureX0SpamProbabilities[j] /= 1.0 - featureProbabilities[j];
			
			// Calculate H(C | X=1) for the possible feature
			entropyOfFeaturesX1[j] = Main.entropy(featureX1SpamProbabilities[j]);
			// Calculate H(C | X=0) for the possible feature
			entropyOfFeaturesX0[j] = Main.entropy(featureX0SpamProbabilities[j]);
			// Calculate the IG for the possible feature using the probabilities and the entropies above
			informationGainOfFeatures[j] = HC - (featureProbabilities[j]*entropyOfFeaturesX1[j] + (1.0-featureProbabilities[j])*entropyOfFeaturesX0[j]);
			
			// Create a new feature and add it to the possiblefeaturesObj arraylist using feature's name and IG from the corresponding list
			possiblefeaturesObj.add(new Feature(features.get(j).name,informationGainOfFeatures[j]));
		}
		
		// Get the best feature regarding IG
		Feature feat = Collections.max(possiblefeaturesObj);
		
		// Return the best feature
		return feat;
	}
	
}
