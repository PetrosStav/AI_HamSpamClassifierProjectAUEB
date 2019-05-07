// Authors
// Stavropoulos Petros (AM : 3150230)
// Savvidis Konstantinos (AM : 3150229)
// Mpanakos Vasileios (AM : 3140125)

import java.util.ArrayList;

// Class that represents a Decision Stump, meaning a decision tree that has depth of 1
// it has a control feature in it's root and then has two decisions, hasFeature and NotHasFeature
// which are both boolean and represent the class (Spam/Ham) regarding a mail having the feature 
// or not
public class DecisionStump {
	
	// The control feature for the root
	Feature controlFeature;
	// The left child / decision -- the mail has the control feature
	boolean hasFeature;
	// The right child / decision -- the mail doesn't have the control feature
	boolean NotHasFeature;
	// The list of train Mails
	ArrayList<Mail> trainMails;
	// THe list of weights for the train Mails
	ArrayList<Double> weights;
	
	// Parameterized Constructor
	public DecisionStump(ArrayList<Mail> trainMails, Feature controlFeature, ArrayList<Double> weights) {
		this.trainMails = trainMails;
		this.controlFeature = controlFeature;
		this.weights = weights;
	}
	
	// Method that trains the Decision Stump
	public void trainDecisionStump() {
		
		// Initialize P(X=1) to 0
		double PX1 = 0.0;
		// Initialize P(C=1^X=1) to 0
		double PCL = 0.0;
		
		// For i from 0 to the size of the train Mails
		for(int i=0;i<trainMails.size();i++) {
			// If the mail in the index i contains the control feature
			if(trainMails.get(i).getText().toUpperCase().substring(9).contains(controlFeature.name)){
				// Increase P(X=1) by the weight at index i
				PX1 += weights.get(i);
				// If the mail in index i is spam then increase  P(C=1^X=1) by the weight at index i
				if(trainMails.get(i).isSpam()) PCL += weights.get(i);
			}
		}
		
		// Divide PCL by P(X=1) so that it becomes from P(C=1^X=1) to P(C=1|X=1)
		PCL = PCL / PX1;
		
		// Set P(C=0|X=1) to 1 - P(C=1|X=1)
		double PNCL = 1-PCL;
		
		// Set P(X=0) to 1 - P(X=1) 
		double PX0 = 1 - PX1;
		
		// Initialize P(C=1^X=0) to 0
		double PCR = 0.0;
		
		// For i from 0 to the size of the train Mails
		for(int i=0;i<trainMails.size();i++) {
			// If the mail at index i doesn't contain the controlFeature and it is spam then increase P(C=1^X=0) by the weight at index i
			if(!trainMails.get(i).getText().toUpperCase().substring(9).contains(controlFeature.name) && trainMails.get(i).isSpam()) PCR += weights.get(i);
		}
		
		// Divide PCR by P(X=0) so that it becomes from P(C=1^X=0) to P(C=1|X=0)
		PCR = PCR / PX0;
		
		// Set P(C=0|X=0) to 1 - P(C=1|X=0)
		double PNCR = 1-PCR;
		
		// Set left child / hasFeature to P(C=1|X=1) > P(C=0|X=1)
		hasFeature = PCL>PNCL;
		// Set right child / NotHasFeature to P(C=1|X=0) > P(C=0|X=0)
		NotHasFeature = PCR>PNCR;
		
	}
	
	// Method that gets a Mail and evaluates it using the decision stump, regarding whether the mail has
	// or not the control feature of the decision stump
	public boolean evaluateDecisionStump(Mail m) {
		// If the mail's text contains the control feature
		if(m.getText().toUpperCase().substring(9).contains(controlFeature.name))
			// Return the left child / decision -- hasFeature
			return hasFeature;
		else
			// Return the right child / decision -- NotHasFeature
			return NotHasFeature;
	}
}
