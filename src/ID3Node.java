// Authors
// Stavropoulos Petros (AM : 3150230)
// Savvidis Konstantinos (AM : 3150229)
// Mpanakos Vasileios (AM : 3140125)

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;

// Class that represents a Node in a ID3 Tree that has it's features, control feature, trainMails, class probability type (Spam/Ham)
// as well as it's left and right child which is an ID3Node
// A Node can be a decision Node, meaning that it doesn't have any children or control feature and it is a stop case for the evaluation
// of the ID3 Classifier
public class ID3Node {

	// The list of all the features
	private ArrayList<Feature> features;
	
	// The list of the train Mails
	private ArrayList<Mail> trainMails;
	
	// The prespecified class (spam - true /ham - false) -- the highest class probability
	private boolean classProb;
	
	// Decision node boolean
	private boolean decisionNode;
	
	// Left child node -- null if decisionNode
	private ID3Node leftNode;
	
	// Right child node -- null if decisionNode
	private ID3Node rightNode;
	
	// Feature which the node controls -- null if decisionNode
	private Feature ctrlFeature;
	
	// Parametrized Constructor
	public ID3Node(ArrayList<Mail> trainMails, ArrayList<Feature> features, boolean classProb,Feature ctrlFeature) {
		this.trainMails = trainMails;
		this.features = features;
		this.classProb = classProb;
		this.ctrlFeature = ctrlFeature;
		decisionNode = false;
		leftNode = null;
		rightNode = null;
	}
	
	// Parametrized Constructor
	public ID3Node(ArrayList<Mail> trainMails, ArrayList<Feature> features, boolean classProb, Feature ctrlFeature, boolean decisionNode) {
		this.trainMails = trainMails;
		this.features = features;
		this.classProb = classProb;
		this.decisionNode = decisionNode;
		this.ctrlFeature = ctrlFeature;
		leftNode = null;
		rightNode = null;
	}

	// Recursive method that creates the ID3 Tree
	// This method is called at the root of the tree and it is recursively called for the node's left and right child
	// Return the ID3Node that has the tree from that node and downwards
	public static ID3Node ID3(ArrayList<Mail> trainMails, ArrayList<Feature> features, boolean classProb, Feature ctrlFeature, double diff) {
		
		// If train Mails is empty 
		if(trainMails.isEmpty()) {
			// Create a decision Node based on the parameters
			ID3Node n = new ID3Node(trainMails, features, classProb, null, true);
			// Return the decision Node
			return n;
		}
		
		// Find the bernoulliValues Table
		int[][] bernoulliValuesTable = Main.findBernoulliTable(trainMails, features);
		
		// Calculate P(C=1) and P(C=0)
		
		int r = bernoulliValuesTable.length;
		int c = bernoulliValuesTable[0].length;
		
		// Find the spam mail count (C=1)
		// Set the counter to 0
		int countSpam = 0;
		// For i from 0 to the row size of the bernoulli values table
		for(int i=0;i<r;i++) {
			// If the last column is 1 then C=1, so increment the counter
			if(bernoulliValuesTable[i][c-1]==1) countSpam++;
		}
		
		// P(C=1)
		double PC = countSpam / (bernoulliValuesTable.length*1.0);
		
		// P(C=0)
		double PNC = 1-PC;
		
		// If the probability of C is very close to 1
		if(1-PC <= diff) { // or 1.0e-100
			// Create a decision Node for C=1
			ID3Node n = new ID3Node(trainMails,features,true,null,true);
			// Return the node
			return n;
			
		// If the probability of C is very close to 0
		}else if(1-PNC <= diff) { // or 1.0e-100
			// Create a decision Node for C=0
			ID3Node n = new ID3Node(trainMails,features,false,null,true);
			// Return the node
			return n;
			
		}
		
		// If features is Empty return P(C=1)>P(C=0) decision Node
		if(features.isEmpty()) {
			// Create a decision Node based on the parameters
			ID3Node n = new ID3Node(trainMails, features, PC>PNC, null, true);
			// Return the decision Node
			return n;
		}
		
		// Find best feature using IG
		Feature bestFeature = findBestFeature(PC, features, bernoulliValuesTable);
		
		// Create a Node that uses best feature for control feature
		ID3Node node = new ID3Node(trainMails, features, PC>PNC, bestFeature);
		
		// Change left child (Xi=1)
		
		// Create a list that is a copy of the trainMails list
		ArrayList<Mail> trainMailsX1 = new ArrayList<>(trainMails);
		// Remove all the items that have the feature Xi=1
		trainMailsX1.removeIf(s -> !s.getText().toUpperCase().substring(9).contains(bestFeature.name));
		
		// Create a list that is a copy of the features list
		ArrayList<Feature> featuresNextL = new ArrayList<>(features);
		// Remove the best feature
		featuresNextL.removeIf(s -> s.name.equals(bestFeature.name));
		
		// Assign to the left child the Node that will be return from the call of the ID3 function
		// using trainMailsSpam, featuresNext
		node.leftNode = ID3(trainMailsX1, featuresNextL, PC>PNC, null,diff);
		
		// Change right child (Xi=0)
		
		// Create a list that is a copy of the trainMails list
		ArrayList<Mail> trainMailsX0 = new ArrayList<>(trainMails);
		// Remove all the items that have the feature Xi=1
		trainMailsX0.removeIf(s -> s.getText().toUpperCase().substring(9).contains(bestFeature.name));
		
		// Create a list that is a copy of the features list
		ArrayList<Feature> featuresNextR = new ArrayList<>(features);
		// Remove the best feature
		featuresNextR.removeIf(s -> s.name.equals(bestFeature.name));
		
		// Assign to the left child the Node that will be return from the call of the ID3 function
		// using trainMailsSpam, featuresNext
		node.rightNode = ID3(trainMailsX0, featuresNextR, PC>PNC, null,diff);
		
		// Return the Node that uses best feature for control feature
		return node;
	}
	
	private static Feature findBestFeature(double PC,ArrayList<Feature> features,int[][] bernoulliValues) {
		
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
		
		// For j from 0 to the length of the size of the possible features
		for(int j=0;j<pfeatSize;j++) {
			// Initialize a counter for all the mails for a specific feature in the table that are X=1 and set value to 0
			int count = 0;
			// Initialize a counter for all the mails for a specific feature that have X=0 and C=0 and set value to 0
			int countX0C0 = 0;
			// Initialize a counter for all the mails for a specific feature that have X=0 and C=1 and set value to 0
			int countX0C1 = 0;
			// Initialize a counter for all the mails for a specific feature that have X=1 and C=0 and set value to 0
			int countX1C0 = 0;
			// Initialize a counter for all the mails for a specific feature that have X=1 and C=1 and set value to 0
			int countX1C1 = 0;
			// For i from 0 to the length of the rows of the bernoulliValues table
			for(int i=0;i<bernoulliValues.length;i++) {
				// If the value is 1
				if(bernoulliValues[i][j]==1) {
					// Increment the counter for X=1
					count++;
					// If the value in the table for the last column is 1 (C=1) then increment the counter for X=1,C=1
					if(bernoulliValues[i][bernoulliValues[i].length-1] == 1) countX1C1++;
					// Else increment the counter for X=1,C=0
					else countX1C0++;
				}else {
					// Else X=0
					// If the value in the table for the last column is 1 (C=1) then increment the counter for X=0,C=1
					if(bernoulliValues[i][bernoulliValues[i].length-1] == 1) countX0C1++;
					// Else increment the counter for X=0,C=0
					else countX0C0++;
				}
			}
			// Calculate the probability P(X=1) for the possible feature
			featureProbabilities[j] = count/(bernoulliValues.length*1.0);
			
			// If the two counters are adding to 0 then make the probability 0 (otherwise it will be a division by zero)
			if(countX1C0 + countX1C1 == 0) featureX1SpamProbabilities[j] = 0;
			// Else calculate P(C=1 | X=1) for the possible feature
			else featureX1SpamProbabilities[j] = countX1C1 / ((countX1C0+countX1C1)*1.0);
			
			// If the two counters are adding to 0 then make the probability 0 (otherwise it will be a division by zero)
			if(countX0C0 + countX0C1 == 0) featureX0SpamProbabilities[j] = 0;
			// Else calculate P(C=1 | X=0) for the possible feature
			else featureX0SpamProbabilities[j] = countX0C1 / ((countX0C0+countX0C1)*1.0);
			
			// Calculate H(C | X=1) for the possible feature
			entropyOfFeaturesX1[j] = Main.entropy(featureX1SpamProbabilities[j]);
			// Calculate H(C | X=0) for the possible feature
			entropyOfFeaturesX0[j] = Main.entropy(featureX0SpamProbabilities[j]);
			// Calculate the IG for the possible feature using the probabilities and the entropies above
			informationGainOfFeatures[j] = HC - (featureProbabilities[j]*entropyOfFeaturesX1[j] + (1-featureProbabilities[j])*entropyOfFeaturesX0[j]);
		}
		
		// Create an ArrayList of Feature objects
		ArrayList<Feature> possiblefeaturesObj = new ArrayList<>();

		// Convert the features to objects in order to sort them
		// For j from 0 to the size of the possible features
		for(int j=0;j<pfeatSize;j++) {
			// Add to the arraylist of feature objects a new feature object using feature's name and IG from the corresponding list  
			possiblefeaturesObj.add(new Feature(features.get(j).name,informationGainOfFeatures[j]));
		}
		
		// Get the best feature
		Feature feat = Collections.max(possiblefeaturesObj);
		
		// Return the best feature
		return feat;
	}

	// Getter for features
	public ArrayList<Feature> getFeatures() {
		return features;
	}

	// Setter for features
	public void setFeatures(ArrayList<Feature> features) {
		this.features = features;
	}

	// Getter for trainMails
	public ArrayList<Mail> getTrainMails() {
		return trainMails;
	}

	// Setter for trainMails
	public void setTrainMails(ArrayList<Mail> trainMails) {
		this.trainMails = trainMails;
	}

	// Getter for class probability type (Spam/Ham)
	public boolean getClassProb() {
		return classProb;
	}

	// Setter for class probability type (Spam/Ham)
	public void setClassProb(boolean prespecSpam) {
		this.classProb = prespecSpam;
	}

	// Getter for decisionNode boolean
	public boolean isDecisionNode() {
		return decisionNode;
	}

	// Setter for decisionNode boolean
	public void setDecisionNode(boolean decisionNode) {
		this.decisionNode = decisionNode;
	}

	// Getter for leftNode
	public ID3Node getLeftNode() {
		return leftNode;
	}

	// Setter for leftNode
	public void setLeftNode(ID3Node leftNode) {
		this.leftNode = leftNode;
	}

	// Getter for rightNode
	public ID3Node getRightNode() {
		return rightNode;
	}

	// Setter for rightNode
	public void setRightNode(ID3Node rightNode) {
		this.rightNode = rightNode;
	}

	// Getter for control feature
	public Feature getCtrlFeature() {
		return ctrlFeature;
	}

	// Setter for control feature
	public void setCtrlFeature(Feature ctrlFeature) {
		this.ctrlFeature = ctrlFeature;
	}
	
	// Print method for the tree -- public
	public void print() {
		// Call the recursive print method
        print("", true);
    }

	// Print method for the tree -- private - recursive
    private void print(String prefix, boolean isTail) {
    	// Create an arraylist for the node's children
    	ArrayList<ID3Node> children = new ArrayList<>();
    	// Add the left child
    	children.add(leftNode);
    	// Add the right child
    	children.add(rightNode);
    	// Print the control feature or classProb (if it is null) along with the prefix and the tail
        System.out.println(prefix + (isTail ? "\\-- " : "|-- ") + (ctrlFeature!=null?ctrlFeature.name:classProb));
        // For every child
        for (int i = 0; i < children.size() - 1; i++) {
        	// If it isn't null print the child using a specific prefix
            if(children.get(i)!=null) children.get(i).print(prefix + (isTail ? "    " : "|   "), false);
        }
        // If the node has children (size > 0)
        if (children.size() > 0) {
        	// If the last child isn't null (right child)
        	// then call the print function to it using true for tail (print different symbols)
        	if(children.get(children.size()-1)!=null) children.get(children.size() - 1).print(prefix + (isTail ?"    " : "|   "), true);
        }
    }
    
    // Print method for the tree to a file -- public
    public void printToFile(String filename) {
    	// Create a string builder
    	StringBuilder sb = new StringBuilder();
    	// Call the recursive printToFile to fill the string builder
    	printToFile("",true,sb);
    	// Initialize a Buffered Writer to null
    	BufferedWriter bw=null;
    	try {
    		// Open the File using the bufferedwriter
			bw = new BufferedWriter(new FileWriter(new File(filename)));
			// Write the contents of the string builder to the file
			bw.write(sb.toString());
		} catch (IOException e) {
			e.printStackTrace();
		}finally {
			// If the buffered writer isn't null
			if(bw!=null) {
				try {
					// Close the buffered writer
					bw.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
    }
    
    // Print mehtod for the tree to a file -- private
    private void printToFile(String prefix, boolean isTail, StringBuilder sb) {
    	// Create an arraylist for the node's children
    	ArrayList<ID3Node> children = new ArrayList<>();
    	// Add the left child
    	children.add(leftNode);
    	// Add the right child
    	children.add(rightNode);
    	// Append to sb the control feature or classProb (if it is null) along with the prefix and the tail
        sb.append(prefix + (isTail ? "\\-- " : "|-- ") + (ctrlFeature!=null?ctrlFeature.name:classProb));
        // Append a new line to sb
        sb.append("\n");
        // For every child
        for (int i = 0; i < children.size() - 1; i++) {
        	// If it isn't null print the child using a specific prefix
            if(children.get(i)!=null) children.get(i).printToFile(prefix + (isTail ? "    " : "|   "), false,sb);
        }
        // If the node has children (size > 0)
        if (children.size() > 0) {
        	// If the last child isn't null (right child)
        	// then call the printToFile function to it using true for tail (print different symbols)
            if(children.get(children.size()-1)!=null) children.get(children.size() - 1).printToFile(prefix + (isTail ?"    " : "|   "), true,sb);
        }
    }
	
}
