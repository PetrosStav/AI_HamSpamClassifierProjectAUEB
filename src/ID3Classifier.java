// Authors
// Stavropoulos Petros (AM : 3150230)
// Savvidis Konstantinos (AM : 3150229)
// Mpanakos Vasileios (AM : 3140125)

import java.util.ArrayList;

// Class that represents an ID3 Classifier
// The classifier is trained using the trainMails and the features by the method ID3 which returns the ID3 Tree
// The ID3 Tree is the trained classifier, which is traversed at the evaluation, using the existence of the
// control features in the email's text for each Node until we get to a decision Node which will answer:
// true  - spam
// false - ham
public class ID3Classifier {

	// The list of all the trainMails
	private ArrayList<Mail> trainMails;
	
	// The list of all the features
	private ArrayList<Feature> features;
	
	// The root of the ID3 tree
	private ID3Node root;
	
	// The difference from 1.0 and 0.0 that P(C) will stop
	private double diff = 0.05;
	
	// Parametrized Constructor
	public ID3Classifier(ArrayList<Mail> trainMails,ArrayList<Feature> features) {
		this.trainMails = trainMails;
		this.features = features;
		root = null;
	}
	
	// Method that trains ID3 using the trainMails and the features it has
	public void trainID3() {
		// Set the root Node to the whole ID3 tree that the recursive function ID3 returns using the the trainMails,
		// the feature List, false for the prespecified category (it wont matter if it is true/false unless the trainMails
		// are empty, which won't be in the start or we can't train the classifier), null for the control feature
		// and diff for the difference from 1.0 and 0.0 that P(C) will stop
		root = ID3Node.ID3(trainMails, features, false, null,diff);
	}
	
	// Method that evaluates a mail as spam (true) or ham(false)
	public boolean evaluateID3(Mail m) {
		
		// Create a temp reference to the root of the tree
		ID3Node temp = root;
		
		// While the reference temp isn't at a decision Node
		while(!temp.isDecisionNode()) {
			
			// Get the control feature of temp Node
			Feature control = temp.getCtrlFeature();
			
			// Check if the feature exists in the mail
			boolean featureExists = m.getText().toUpperCase().substring(9).contains(control.name);
			
			// If it exists goto the left Node
			if(featureExists) temp=temp.getLeftNode();
			// Else goto the right Node
			else temp=temp.getRightNode();
			
		}
		
		// We have reached a decision Node, so return the Node's class with the highest probability
		// true  -- spam
		// false -- ham
		return temp.getClassProb();
		
	}
	
	// Print method for the ID3Classifier
	public void print() {
		root.print();
	}
	
	// Print to file method for the ID3Classifier
	public void printToFile(String filename) {
		root.printToFile(filename);
	}

	// Getter for the trainMails
	public ArrayList<Mail> getTrainMails() {
		return trainMails;
	}

	// Setter for the trainMails
	public void setTrainMails(ArrayList<Mail> trainMails) {
		this.trainMails = trainMails;
	}

	// Getter for features
	public ArrayList<Feature> getFeatures() {
		return features;
	}

	// Setter for features
	public void setFeatures(ArrayList<Feature> features) {
		this.features = features;
	}

	// Getter for root Node
	public ID3Node getRoot() {
		return root;
	}

	// Setter for root Node
	public void setRoot(ID3Node root) {
		this.root = root;
	}

	// Getter for the difference value that P(C=1) and P(C=0) must have for a decision
	public double getDiff() {
		return diff;
	}

	// Setter for the difference value that P(C=1) and P(C=0) must have for a decision
	public void setDiff(double diff) {
		this.diff = diff;
	}
	

}