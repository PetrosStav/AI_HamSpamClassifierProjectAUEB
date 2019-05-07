// Authors
// Stavropoulos Petros (AM : 3150230)
// Savvidis Konstantinos (AM : 3150229)
// Mpanakos Vasileios (AM : 3140125)

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Random;

public class Main {
	
	public static void main(String[] args) {
		
		// Print message to console
		System.out.println("Reading train/validate/test mails...");
		
		// Initialize each mail list
		ArrayList<Mail> trainMails = new ArrayList<>();
		ArrayList<Mail> validateMails = new ArrayList<>();
		ArrayList<Mail> testMails = new ArrayList<>();
		
		// Initialize and set seed
		int seed = 1;
		
		// Get the mails from the folder, set the mail lists and set the parameters for
		// ham_spam ratio, seed number and the use of the seed
		getMails("mails",trainMails,validateMails,testMails,0.7,seed,true);
		
		System.out.println();
		
		// Initialize a feature list to null
		ArrayList<Feature> features = null;
		
		// Set the boolean that controls whether the features are loaded from file
		// or calculated using information gain and the train mails
		boolean featuresFromFile = true;
		
		// Check the featuresFromFile boolean
		// true  -> load features from file
		// false -> calculate using train mails and information gain
		if(featuresFromFile) {
			// Print message to console
			System.out.println("Reading features from file...");
			// Load the features and IG from the features_{seed}.txt file
			features = findFeatures("features_"+ seed +".txt");
			// Print message to console
			System.out.println("Features loaded.");
		}else {
			// Print message to console
			System.out.println("Searching for best features...");
			// Initialize the number of best features from all the possible features that will be kept
			int featNum = 1000;
			// Find the best features from the trainMails using Information Gain and keep the number indicated by featNum
			features = findFeatures(trainMails, featNum,seed);
			// Print message to console
			System.out.println("Features created and loaded.");
		}
		
		// Copy the features list to another with the name oldfeatures
		ArrayList<Feature> oldfeatures = new ArrayList<>(features);
		
		// Print message to console
		System.out.println();
		
		//FOR BAYES
		
		// Boolean whether to use validate mail list to calclate the best features number parameter for the Naive Bayes Classifier
		boolean calcBestFeatNB = false;
		
		// Best for seed 1 : 78 features -- default value
		int bestFeatNumNB = 78;
		
		// If true then find the best features number for Bayes
		if(calcBestFeatNB) {
		
			// Begin with 5 features
			bestFeatNumNB = 5;
			// Initialize the best success rate (accuracy) found
			double bestSuccRate = 0.0;
			
			// Print message to console
			System.out.print("Finding best feature number for Naive Bayes using validation data");
			
			// Find how many features give the best successrate at validation (test from 5 to 100)
			
			// For i from 5 to 100
			for(int i=5;i<100;i++) {
			
				// Select how many of the best features will be used
				features = new ArrayList<>( oldfeatures.subList(0, i) );
				
				// Initialize the classifier with the trainMails and the feature list
				NaiveBayesClassifier classifierNB = new NaiveBayesClassifier(trainMails,features);
				
				// Train the Naive Bayes classifier
				classifierNB.trainNaiveBayes();
				
				// Initialize the variable for all the right answers found to 0
				int countRightNBVal = 0;
				// For each mail in the validate mails
				for(Mail m : validateMails) {
					// Get the answer from the classifier
					boolean ans = classifierNB.evaluateNaiveBayes(m);
					// If the asnwer is right then increment the count of right answers
					if(ans==m.isSpam()) countRightNBVal++;
				}
				
				// Calculate the success rate for the validate set
				double succRateNBVal = countRightNBVal/(validateMails.size()*1.0);
				
				// If the success rate is better than the best succes rate
				if(succRateNBVal > bestSuccRate) {
					// Set the best success rate to this success rate
					bestSuccRate = succRateNBVal;
					// Set the best features number to i
					bestFeatNumNB = i;
				}
				// Print message to console for progress
				if(i%20==0) System.out.print(".");
				
			}
			
			System.out.println();
			
			// Print message to console about the best features number found
			System.out.println("Best feature number: " + bestFeatNumNB);
		
		}
		
		// Select how many of the best features will be used using the best features number
		features = new ArrayList<>( oldfeatures.subList(0, bestFeatNumNB) );
		
		// Create an arff file for the train set
		createArffFile(trainMails, features, "Bayes.arff");
		// Create an arff file for the test set
		createArffFile(testMails, features, "Bayes_test.arff");
		
		// Print message to console
		System.out.println("Creating Naive Bayes Classifier...");
		
		// Initialize a naive bayes classifier using the trainMails and the features list
		NaiveBayesClassifier classifierNB = new NaiveBayesClassifier(trainMails,features);
		
		// Print message to console
		System.out.println("Training Naive Bayes...");
		// Train the classifier
		classifierNB.trainNaiveBayes();
		
//		System.out.println("Evaluating validate mails...");
//		
//		int countRightNBVal = 0;
//		for(Mail m : validateMails) {
//			boolean ans = classifierNB.evaluateNaiveBayes(m);
//			if(ans==m.isSpam()) countRightNBVal++;
//		}
//		
//		double succRateNBVal = countRightNBVal/(validateMails.size()*1.0);
//		
//		System.out.println("SuccessRate: " +succRateNBVal);
		
		System.out.println();
		
		// Print message to console
		System.out.println("Evaluating train mails...");
		
		// Initialize variables to 0 for:
		
		// Right answers for the train set, C=1
		int countRightNBTrain = 0;
		// False positives for the train set, C=1
		int falsePositiveNBTrain = 0;
		// True positives for the train set, C=1
		int truePositiveNBTrain = 0;
		// False negatives for the train set, C=1
		int falseNegativeNBTrain = 0;
		// True positives for the train set, C=0
		int truePositiveNBTrainC0 = 0;
		// False positives for the train set, C=0
		int falsePositiveNBTrainC0 = 0;
		// False negatives for the train set, C=0
		int falseNegativeNBTrainC0 = 0;
		
		// For each mail in the trainMails
		for(Mail m : trainMails) {
			// Get the answer from the classifier
			boolean ans = classifierNB.evaluateNaiveBayes(m);
			// If ans=true(spam) and the mail isn't spam then increment false positives for C=1
			if(ans && !m.isSpam()) falsePositiveNBTrain++;
			// If ans=true(spam) and the mail is spam then increment true positives for C=1
			if(ans && m.isSpam()) truePositiveNBTrain++;
			// If ans=false(ham) and the mail is spam then increment false negatives for C=1
			if(!ans && m.isSpam()) falseNegativeNBTrain++;
			// If ans=false(ham) and the mail isn't spam then increment true positives for C=0
			if(!ans && !m.isSpam()) truePositiveNBTrainC0++;
			// If ans=false(ham) and the mail is spam then increment false positives for C=0
			if(!ans && m.isSpam()) falsePositiveNBTrainC0++;
			// If ans=true(spam) and the mail isn't spam then increment false negatives for C=0
			if(ans && !m.isSpam()) falseNegativeNBTrainC0++;
			// If answer is right then increment right answers
			if(ans==m.isSpam()) countRightNBTrain++;
		}
		
		// Calculate success rate for train set
		double SuccRateNBTrain = countRightNBTrain/(trainMails.size()*1.0);
		// Calculate precision rate for train set, C=1
		double precisionRateNBTrain = truePositiveNBTrain / ((truePositiveNBTrain+falsePositiveNBTrain)*1.0);
		// Calculate recall rate for train set, C=1
		double recallRateNBTrain = truePositiveNBTrain / ((truePositiveNBTrain+falseNegativeNBTrain)*1.0);
		// Calculate precision rate for train set, C=0
		double precisionRateNBTrainC0 = truePositiveNBTrainC0 / ((truePositiveNBTrainC0+falsePositiveNBTrainC0)*1.0);
		// Calculate recall rate for train set, C=0
		double recallRateNBTrainC0 = truePositiveNBTrainC0 / ((truePositiveNBTrainC0+falseNegativeNBTrainC0)*1.0);
		
		// Print to console all rates
		
		System.out.println("SuccessRateTrain: " + SuccRateNBTrain);
		System.out.println("PrecisionRateTrain: " + precisionRateNBTrain);
		System.out.println("RecallRateTrain: " + recallRateNBTrain);
		System.out.println("PrecisionRateTrainC0: " + precisionRateNBTrainC0);
		System.out.println("RecallRateTrainC0: " + recallRateNBTrainC0);
		
		System.out.println();
		
		// Print message to console
		System.out.println("Evaluating test mails...");
		
		// Initialize variables to 0 for:
		
		// Right answers for the test set, C=1
		int countRightNB = 0;
		// False positives for the test set, C=1
		int falsePositiveNB = 0;
		// True positives for the test set, C=1
		int truePositiveNB = 0;
		// False negatives for the test set, C=1
		int falseNegativeNB = 0;
		// True positives for the test set, C=0
		int truePositiveNBC0 = 0;
		// False positives for the test set, C=0
		int falsePositiveNBC0 = 0;
		// False negatives for the test set, C=0
		int falseNegativeNBC0 = 0;
		
		// For each mail in the test set
		for(Mail m : testMails) {
			// Get the answer from the classifier
			boolean ans = classifierNB.evaluateNaiveBayes(m);
			// If ans=true(spam) and the mail isn't spam then increment false positives for C=1
			if(ans && !m.isSpam()) falsePositiveNB++;
			// If ans=true(spam) and the mail is spam then increment true positives for C=1
			if(ans && m.isSpam()) truePositiveNB++;
			// If ans=false(ham) and the mail is spam then increment false negatives for C=1
			if(!ans && m.isSpam()) falseNegativeNB++;
			// If ans=false(ham) and the mail isn't spam then increment true positives for C=0
			if(!ans && !m.isSpam()) truePositiveNBC0++;
			// If ans=false(ham) and the mail is spam then increment false positives for C=0
			if(!ans && m.isSpam()) falsePositiveNBC0++;
			// If ans=true(spam) and the mail isn't spam then increment false negatives for C=0
			if(ans && !m.isSpam()) falseNegativeNBC0++;
			// If answer is right then increment right answers
			if(ans==m.isSpam()) countRightNB++;
		}
		
		// Calculate success rate for test set
		double SuccRateNB = countRightNB/(testMails.size()*1.0);
		// Calculate precision rate for test set, C=1
		double precisionRateNB = truePositiveNB / ((truePositiveNB+falsePositiveNB)*1.0);
		// Calculate recall rate for test set, C=1
		double recallRateNB = truePositiveNB / ((truePositiveNB+falseNegativeNB)*1.0);
		// Calculate precision rate for test set, C=0
		double precisionRateNBC0 = truePositiveNBC0 / ((truePositiveNBC0+falsePositiveNBC0)*1.0);
		// Calculate recall rate for test set, C=0
		double recallRateNBC0 = truePositiveNBC0 / ((truePositiveNBC0+falseNegativeNBC0)*1.0);
		
		// Print to console all rates
		
		System.out.println("SuccessRate: " + SuccRateNB);
		System.out.println("PrecisionRate: " + precisionRateNB);
		System.out.println("RecallRate: " + recallRateNB);
		System.out.println("PrecisionRateC0: " + precisionRateNBC0);
		System.out.println("RecallRateC0: " + recallRateNBC0);
		
		System.out.println();
		
		// END FOR BAYES
		
		// FOR ID3
		
		// Boolean whether to use validate mail list to calclate the best features number parameter for the ID3 Classifier
		boolean calcBestFeatID3 = false;
		// Best for seed 1 from validate is 40 -- default value
		int bestFeatNumID3 = 40;
		// The array of the diferent diff values
		double[] diffs = {0.0, 0.05, 0.1, 0.15};
		// The index of the best diff value from the diffs array
		int bestDiff = 0;
		
		// If true then find the best features number for ID3
		if(calcBestFeatID3) {
			
			// Begin with 5 features
			bestFeatNumID3 = 5;
			// Initialize the best success rate (accuracy) found
			double bestSuccRate = 0.0;
			
			// Print message to console
			System.out.print("Finding best feature number for ID3 using validation data");
			
			// Find how many features give the best successrate at validation (test from 5 to 100)
			for(int i=5;i<100;i++) {
				for(int j=0;j<4;j++) {
					
					// Select how many of the best features will be used
					features = new ArrayList<>(oldfeatures.subList(0, i));
					
					// Initialize the classifier with the trainMails and the feature list
					ID3Classifier classifierID3 = new ID3Classifier(trainMails,features);
					
					// Set diff -- how much from 1.0 or 0.0 must P(C=1) differ to make a decision Node from the diffs array
					classifierID3.setDiff(diffs[j]);
					
					// Train the ID3 Classifier
					classifierID3.trainID3();
					
					// Initialize the variable for all the right answers found to 0
					int countRightID3Val = 0;
					// For each mail in the validate mails
					for(Mail m : validateMails) {
						// Get the answer from the classifier
						boolean ans = classifierID3.evaluateID3(m);
						// If the asnwer is right then increment the count of right answers
						if(ans==m.isSpam()) countRightID3Val++;
					}
					// Calculate the success rate for the validate set
					double succRateID3Val = countRightID3Val/(validateMails.size()*1.0);
					// If the success rate is better than the best succes rate
					if(succRateID3Val > bestSuccRate) {
						// Set the best success rate to this success rate
						bestSuccRate = succRateID3Val;
						// Set the best features number to i
						bestFeatNumID3 = i;
						// Set the best diff index to j
						bestDiff = j;
					}
					
				}
				// Print message to console for progress
				if(i%20==0) System.out.print(".");
			}
			
			System.out.println();
			
			// Print message to console about the best features number found
			System.out.println("Best feature number: " + bestFeatNumID3);
			// Print message to console about the best diff index found
			System.out.println("Best diff: " + bestDiff);
			
		}
		
		// Select how many of the best features will be used using the best features number
		features = new ArrayList<>(oldfeatures.subList(0, bestFeatNumID3));
		
		// Create an arff file for the train set
		createArffFile(trainMails, features, "ID3.arff");
		// Create an arff file for the test set
		createArffFile(testMails, features, "ID3_test.arff");
		
		// Print message to console
		System.out.println("Creating ID3 Classifier...");
		
		// Initialize an ID3 classifier using the trainMails and the features list
		ID3Classifier classifierID3 = new ID3Classifier(trainMails,features);
		
		// Set diff -- how much from 1.0 or 0.0 must P(C=1) differ to make a decision Node from the diffs array using the best index
		classifierID3.setDiff(diffs[bestDiff]);
		
		// Print message to console
		System.out.println("Training ID3...");
		
		// Train the classifier
		classifierID3.trainID3();
		
//		System.out.println("Evaluating validate mails...");
//		
//		int countRightID3Val = 0;
//		for(Mail m : validateMails) {
//			boolean ans = classifierID3.evaluateID3(m);
//			if(ans==m.isSpam()) countRightID3Val++;
//		}
//		
//		double succRateID3Val = countRightID3Val/(validateMails.size()*1.0);
//		
//		System.out.println("SuccessRate: " +succRateID3Val);
		
		System.out.println();
		
		// Print message to console
		System.out.println("Evaluating train mails...");
		
		// Initialize variables to 0 for:
		
		// Right answers for the train set, C=1
		int countRightID3Train = 0;
		// False positives for the train set, C=1
		int falsePositiveID3Train = 0;
		// True positives for the train set, C=1
		int truePositiveID3Train = 0;
		// False negatives for the train set, C=1
		int falseNegativeID3Train = 0;
		// True positives for the train set, C=0
		int truePositiveID3TrainC0 = 0;
		// False positives for the train set, C=0
		int falsePositiveID3TrainC0 = 0;
		// False negatives for the train set, C=0
		int falseNegativeID3TrainC0 = 0;
		
		// For each mail in the trainMails
		for(Mail m : trainMails) {
			// Get the answer from the classifier
			boolean ans = classifierID3.evaluateID3(m);
			// If ans=true(spam) and the mail isn't spam then increment false positives for C=1
			if(ans && !m.isSpam()) falsePositiveID3Train++;
			// If ans=true(spam) and the mail is spam then increment true positives for C=1
			if(ans && m.isSpam()) truePositiveID3Train++;
			// If ans=false(ham) and the mail is spam then increment false negatives for C=1
			if(!ans && m.isSpam()) falseNegativeID3Train++;
			// If ans=false(ham) and the mail isn't spam then increment true positives for C=0
			if(!ans && !m.isSpam()) truePositiveID3TrainC0++;
			// If ans=false(ham) and the mail is spam then increment false positives for C=0
			if(!ans && m.isSpam()) falsePositiveID3TrainC0++;
			// If ans=true(spam) and the mail isn't spam then increment false negatives for C=0
			if(ans && !m.isSpam()) falseNegativeID3TrainC0++;
			// If answer is right then increment right answers
			if(ans==m.isSpam()) countRightID3Train++;
		}
		// Calculate success rate for train set
		double succRateID3Train = countRightID3Train/(trainMails.size()*1.0);
		// Calculate precision rate for train set, C=1
		double precisionRateID3Train = truePositiveID3Train / ((truePositiveID3Train+falsePositiveID3Train)*1.0);
		// Calculate recall rate for train set, C=1
		double recallRateID3Train = truePositiveID3Train / ((truePositiveID3Train+falseNegativeID3Train)*1.0);
		// Calculate precision rate for train set, C=0
		double precisionRateID3TrainC0 = truePositiveID3TrainC0 / ((truePositiveID3TrainC0+falsePositiveID3TrainC0)*1.0);
		// Calculate recall rate for train set, C=0
		double recallRateID3TrainC0 = truePositiveID3TrainC0 / ((truePositiveID3TrainC0+falseNegativeID3TrainC0)*1.0);
		
		// Print to console all rates
		
		System.out.println("SuccessRateTrain: " +succRateID3Train);
		System.out.println("PrecisionRateTrain: " + precisionRateID3Train);
		System.out.println("RecallRateTrain: " + recallRateID3Train);
		
		System.out.println("PrecisionRateTrainC0: " + precisionRateID3TrainC0);
		System.out.println("RecallRateTrainC0: " + recallRateID3TrainC0);
		
		System.out.println();
		
		// Print message to console
		System.out.println("Evaluating test mails...");
		
		// Initialize variables to 0 for:
		
		// Right answers for the test set, C=1
		int countRightID3 = 0;
		// False positives for the test set, C=1
		int falsePositiveID3 = 0;
		// True positives for the test set, C=1
		int truePositiveID3 = 0;
		// False negatives for the test set, C=1
		int falseNegativeID3 = 0;
		// True positives for the test set, C=0
		int truePositiveID3C0 = 0;
		// False positives for the test set, C=0
		int falsePositiveID3C0 = 0;
		// False negatives for the test set, C=0
		int falseNegativeID3C0 = 0;
		// For each mail in the test set
		for(Mail m : testMails) {
			// Get the answer from the classifier
			boolean ans = classifierID3.evaluateID3(m);
			// If ans=true(spam) and the mail isn't spam then increment false positives for C=1
			if(ans && !m.isSpam()) falsePositiveID3++;
			// If ans=true(spam) and the mail is spam then increment true positives for C=1
			if(ans && m.isSpam()) truePositiveID3++;
			// If ans=false(ham) and the mail is spam then increment false negatives for C=1
			if(!ans && m.isSpam()) falseNegativeID3++;
			// If ans=false(ham) and the mail isn't spam then increment true positives for C=0
			if(!ans && !m.isSpam()) truePositiveID3C0++;
			// If ans=false(ham) and the mail is spam then increment false positives for C=0
			if(!ans && m.isSpam()) falsePositiveID3C0++;
			// If ans=true(spam) and the mail isn't spam then increment false negatives for C=0
			if(ans && !m.isSpam()) falseNegativeID3C0++;
			// If answer is right then increment right answers
			if(ans==m.isSpam()) countRightID3++;
		}
		// Calculate success rate for test set
		double succRateID3 = countRightID3/(testMails.size()*1.0);
		// Calculate precision rate for test set, C=1
		double precisionRateID3 = truePositiveID3 / ((truePositiveID3+falsePositiveID3)*1.0);
		// Calculate recall rate for test set, C=1
		double recallRateID3 = truePositiveID3 / ((truePositiveID3+falseNegativeID3)*1.0);
		// Calculate precision rate for test set, C=0
		double precisionRateID3C0 = truePositiveID3C0 / ((truePositiveID3C0+falsePositiveID3C0)*1.0);
		// Calculate recall rate for test set, C=0
		double recallRateID3C0 = truePositiveID3C0 / ((truePositiveID3C0+falseNegativeID3C0)*1.0);
		
		// Print to console all rates
		
		System.out.println("SuccessRate: " +succRateID3);
		System.out.println("PrecisionRate: " + precisionRateID3);
		System.out.println("RecallRate: " + recallRateID3);
		
		System.out.println("PrecisionRateC0: " + precisionRateID3C0);
		System.out.println("RecallRateC0: " + recallRateID3C0);
		
		System.out.println();
		
		// Print the tree to the console or to a file
		
//		classifier.print();
//		classifierID3.printToFile("ID3tree.txt");
		
		// END FOR ID3
		
		// FOR ADABOOST
		
		// Boolean whether to use validate mail list to calculate the best features number parameter for the AdaBoost Classifier
		boolean calcBestFeatAda = false;
		
		// Best for seed 1 : 44 features -- default value
		int bestFeatNumAda = 44;
		// Number of repetitions (weak classifiers) array
		int[] numOfHypotheses = {20,30,50};
		// Index for the array above for the best hypotheses number
		// Best for seed 1 : index 2 -> 50 hypotheses
		int bestHypNum = 2;
		
		// If true then find the best features number for AdaBoost
		if(calcBestFeatAda) {
			// Begin with 5 features
			bestFeatNumAda = 5;
			// Initialize the best success rate (accuracy) found
			double bestSuccRate = 0.0;
			// Begin with index 1 -> 30 hypotheses as the best
			bestHypNum = 1;
			
			// Print message to console
			System.out.print("Finding best feature number for AdaBoost using validation data");
			
			// Find how many features give the best successrate at validation (test from 5 to 100)
			for(int i=5;i<100;i++) {
				for(int j=0;j<3;j++) {
					// Select how many of the best features will be used
					features = new ArrayList<>(oldfeatures.subList(0, i));
					// Initialize the classifier with the trainMails,the feature list and the number of hypotheses using j as an index for the array
					// of hypotheses
					AdaboostClassifier classifierAda = new AdaboostClassifier(trainMails, features, numOfHypotheses[j]);
					
					// Train the AdaBoost classifier
					classifierAda.trainAdaboost();
					// Initialize the variable for all the right answers found to 0
					int countRightAdaVal = 0;
					// For each mail in the validate mails
					for(Mail m : validateMails) {
						// Get the answer from the classifier
						boolean ans = classifierAda.evaluateAdaboost(m);
						// If the asnwer is right then increment the count of right answers
						if(ans==m.isSpam()) countRightAdaVal++;
					}
					// Calculate the success rate for the validate set
					double succRateAdaVal = countRightAdaVal/(testMails.size()*1.0);
					// If the success rate is better than the best succes rate
					if(succRateAdaVal > bestSuccRate) {
						// Set the best success rate to this success rate
						bestSuccRate = succRateAdaVal;
						// Set the best features number to i
						bestFeatNumAda = i;
						// Set the best hypotheses number index to j
						bestHypNum = j;
					}
					
				}
				// Print message to console for progress
				if(i%20==0) System.out.print(".");
				
			}
			
			System.out.println();
			
			// Print message to console about the best features number found
			System.out.println("Best feature number: " + bestFeatNumAda);
			// Print message to console about the best hypotheses number index found
			System.out.println("Best hypotheses number: " + bestHypNum);
			
		}
		// Select how many of the best features will be used using the best features number
		features = new ArrayList<>(oldfeatures.subList(0, bestFeatNumAda));
		
		// Create an arff file for the train set
		createArffFile(trainMails, features, "AdaBoost.arff");
		// Create an arff file for the test set
		createArffFile(testMails, features, "AdaBoost_test.arff");
		// Print message to console
		System.out.println("Creating AdaBoost Classifier");
		
		// Initialize an AdaBoost classifier using the trainMails, the features list and the number of repetitions (number of hypotheses)
		AdaboostClassifier classifierAda = new AdaboostClassifier(trainMails, features, numOfHypotheses[bestHypNum]);
		
		// Print message to console
		System.out.println("Training AdaBoost...");
		// Train the classifier
		classifierAda.trainAdaboost();
		
//		System.out.println("Evaluating validate mails...");
//		
//		int countRightAdaVal = 0;
//		for(Mail m : validateMails) {
//			boolean ans = classifierAda.evaluateAdaboost(m);
//			if(ans==m.isSpam()) countRightAdaVal++;
//		}
//		
//		double succRateAdaVal = countRightAdaVal/(testMails.size()*1.0);
//		
//		System.out.println("SuccessRate: " +succRateAdaVal);
		
		System.out.println();
		// Print message to console
		System.out.println("Evaluating train mails...");
		
		// Initialize variables to 0 for:
		
		// Right answers for the train set, C=1
		int countRightAdaTrain = 0;
		// False positives for the train set, C=1
		int falsePositiveAdaTrain = 0;
		// True positives for the train set, C=1
		int truePositiveAdaTrain = 0;
		// False negatives for the train set, C=1
		int falseNegativeAdaTrain = 0;
		// True positives for the train set, C=0
		int truePositiveAdaTrainC0 = 0;
		// False positives for the train set, C=0
		int falsePositiveAdaTrainC0 = 0;
		// False negatives for the train set, C=0
		int falseNegativeAdaTrainC0 = 0;
		
		// For each mail in the trainMails
		for(Mail m : trainMails) {
			// Get the answer from the classifier
			boolean ans = classifierAda.evaluateAdaboost(m);
			// If ans=true(spam) and the mail isn't spam then increment false positives for C=1
			if(ans && !m.isSpam()) falsePositiveAdaTrain++;
			// If ans=true(spam) and the mail is spam then increment true positives for C=1
			if(ans && m.isSpam()) truePositiveAdaTrain++;
			// If ans=false(ham) and the mail is spam then increment false negatives for C=1
			if(!ans && m.isSpam()) falseNegativeAdaTrain++;
			// If ans=false(ham) and the mail isn't spam then increment true positives for C=0
			if(!ans && !m.isSpam()) truePositiveAdaTrainC0++;
			// If ans=false(ham) and the mail is spam then increment false positives for C=0
			if(!ans && m.isSpam()) falsePositiveAdaTrainC0++;
			// If ans=true(spam) and the mail isn't spam then increment false negatives for C=0
			if(ans && !m.isSpam()) falseNegativeAdaTrainC0++;
			// If answer is right then increment right answers
			if(ans==m.isSpam()) countRightAdaTrain++;
		}
		
		// Calculate success rate for train set
		double succRateAdaTrain = countRightAdaTrain/(trainMails.size()*1.0);
		// Calculate precision rate for train set, C=1
		double precisionRateAdaTrain = truePositiveAdaTrain / ((truePositiveAdaTrain+falsePositiveAdaTrain)*1.0);
		// Calculate recall rate for train set, C=1
		double recallRateAdaTrain = truePositiveAdaTrain / ((truePositiveAdaTrain+falseNegativeAdaTrain)*1.0);
		// Calculate precision rate for train set, C=0
		double precisionRateAdaTrainC0 = truePositiveAdaTrainC0 / ((truePositiveAdaTrainC0+falsePositiveAdaTrainC0)*1.0);
		// Calculate recall rate for train set, C=0
		double recallRateAdaTrainC0 = truePositiveAdaTrainC0 / ((truePositiveAdaTrainC0+falseNegativeAdaTrainC0)*1.0);
		
		// Print to console all rates
		
		System.out.println("SuccessRateTrain: " +succRateAdaTrain);
		System.out.println("PrecisionRateTrain: " + precisionRateAdaTrain);
		System.out.println("RecallRateTrain: " + recallRateAdaTrain);
		System.out.println("PrecisionRateTrainC0: " + precisionRateAdaTrainC0);
		System.out.println("RecallRateTrainC0: " + recallRateAdaTrainC0);
		
		System.out.println();
		
		// Print message to console
		System.out.println("Evaluating test mails...");
		
		// Initialize variables to 0 for:
		
		// Right answers for the test set, C=1
		int countRightAda = 0;
		// False positives for the test set, C=1
		int falsePositiveAda = 0;
		// True positives for the test set, C=1
		int truePositiveAda = 0;
		// False negatives for the test set, C=1
		int falseNegativeAda = 0;
		// True positives for the test set, C=0
		int truePositiveAdaC0 = 0;
		// False positives for the test set, C=0
		int falsePositiveAdaC0 = 0;
		// False negatives for the test set, C=0
		int falseNegativeAdaC0 = 0;
		
		// For each mail in the test set
		for(Mail m : testMails) {
			// Get the answer from the classifier
			boolean ans = classifierAda.evaluateAdaboost(m);
			// If ans=true(spam) and the mail isn't spam then increment false positives for C=1
			if(ans && !m.isSpam()) falsePositiveAda++;
			// If ans=true(spam) and the mail is spam then increment true positives for C=1
			if(ans && m.isSpam()) truePositiveAda++;
			// If ans=false(ham) and the mail is spam then increment false negatives for C=1
			if(!ans && m.isSpam()) falseNegativeAda++;
			// If ans=false(ham) and the mail isn't spam then increment true positives for C=0
			if(!ans && !m.isSpam()) truePositiveAdaC0++;
			// If ans=false(ham) and the mail is spam then increment false positives for C=0
			if(!ans && m.isSpam()) falsePositiveAdaC0++;
			// If ans=true(spam) and the mail isn't spam then increment false negatives for C=0
			if(ans && !m.isSpam()) falseNegativeAdaC0++;
			// If answer is right then increment right answers
			if(ans==m.isSpam()) countRightAda++;
		}
		
		// Calculate success rate for test set
		double succRateAda = countRightAda/(testMails.size()*1.0);
		// Calculate precision rate for test set, C=1
		double precisionRateAda = truePositiveAda / ((truePositiveAda+falsePositiveAda)*1.0);
		// Calculate recall rate for test set, C=1
		double recallRateAda = truePositiveAda / ((truePositiveAda+falseNegativeAda)*1.0);
		// Calculate precision rate for test set, C=0
		double precisionRateAdaC0 = truePositiveAdaC0 / ((truePositiveAdaC0+falsePositiveAdaC0)*1.0);
		// Calculate recall rate for test set, C=0
		double recallRateAdaC0 = truePositiveAdaC0 / ((truePositiveAdaC0+falseNegativeAdaC0)*1.0);
		
		// Print to console all rates
		
		System.out.println("SuccessRate: " +succRateAda);
		System.out.println("PrecisionRate: " + precisionRateAda);
		System.out.println("RecallRate: " + recallRateAda);
		System.out.println("PrecisionRateC0: " + precisionRateAdaC0);
		System.out.println("RecallRateC0: " + recallRateAdaC0);
		
		System.out.println();
		
		// END FOR ADABOOST
		
	}
	
	// Method that finds the features from a file, using a filename as a parameter
	// Returns an arraylist with all the features found
	public static ArrayList<Feature> findFeatures(String filename){
		// Initialize the file
		File featuresFile = new File(filename);
		
		// Initialize the feature ArrayList
		ArrayList<Feature> features = new ArrayList<>();
		
		// Initialize a string for each line
		String line = null;
		// Initialize a buffered reader to null
		BufferedReader br = null;
		try {
			// Create the buffered reader using the features file
			br = new BufferedReader(new FileReader(featuresFile));
			
			// While there are more lines in the file
			while((line = br.readLine())!= null) {
				// If the line isn't empty ""
				if(!line.trim().equals("")) {
					// Split the line using '~' as a delimiter
					String[] feature_parts = line.split("~");
					// Create a new feature using the name and IG and add it to the features list
					features.add(new Feature(feature_parts[0],Double.parseDouble(feature_parts[1])));
				}
			}
			
			
			
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			// If buffered reader isn't null
			if(br!=null) {
				try {
					// Close it
					br.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
		
		// Return the feature list
		return features;
		
	}
	
	// Method that finds the n-best features using Information Gain from the trainMails, writes a file named "features.txt" with them
	// and returns them as an ArrayList
	public static ArrayList<Feature> findFeatures(ArrayList<Mail> trainMails, int n, int seed){
		
		// Initialize a set of all the possible features from the words of the mails
		HashSet<String> possibleFeatures = new HashSet<>();
		
		// Create a list for all unique words of each mail - ham and spam - seperately
		ArrayList<HashSet<String>> mailWordsListHam = new ArrayList<>();
		ArrayList<HashSet<String>> mailWordsListSpam = new ArrayList<>();
		
		// For each mail in the trainMails
		for(Mail m : trainMails) {
			// Get the text from the mail -- to uppercase and crop the first 9 characters ("Subject: ") as it is not important
			String text = m.getText().toUpperCase().substring(9);
			// Get the mail's words using a delimiter for all whitespace characters and add them to a list
			ArrayList<String> words = new ArrayList<>( Arrays.asList(text.split("\\s+"))) ;
			// Add the mail's words to the list as a hashSet according to ham / spam
			if(!m.isSpam()) {
				mailWordsListHam.add(new HashSet<String>(words));
			}else {
				mailWordsListSpam.add(new HashSet<String>(words));
			}
			// Add the mail's word in the hashSet of all possible features
			possibleFeatures.addAll(words);
		}
		// Remove string "" -- not useful
		possibleFeatures.remove("");
		// Remove string with null character -- not useful
		possibleFeatures.remove(""+((char)0));
		
		// Convert the possible features to a list
		ArrayList<String> possibleFeaturesList = new ArrayList<>( possibleFeatures);
		
		int hamSize = mailWordsListHam.size();
		int spamSize = mailWordsListSpam.size();
		int pfeatSize = possibleFeaturesList.size();
		
		// Create a 2D array for the bernoulli values for each mail and each possible feature
		int[][] bernoulliValues = new int[hamSize+spamSize][pfeatSize+1];
		
		// For i from 0 to the size of the ham mails
		for(int i=0;i<hamSize;i++) {
			// For j from 0 to the size of the possible features
			for(int j=0;j<pfeatSize;j++) {
				// If the ham mail in index i contains the word in the possibleFeaturesList in index j
				// then the bernoulli value for that feature and that mail is 1, else it is 0
				if(mailWordsListHam.get(i).contains(possibleFeaturesList.get(j))){
					bernoulliValues[i][j]=1;
				}else {
					bernoulliValues[i][j]=0;
				}
			}
			// Set last value (column) to 0 for ham
			bernoulliValues[i][pfeatSize] = 0;
		}
		
		// For i from the size of the ham mails to the size of the ham mails + spam mails (the size of all the train mails)
		for(int i=hamSize;i<hamSize+spamSize;i++) {
			// For j from 0 to the size of the possible features
			for(int j=0;j<pfeatSize;j++) {
				// If the ham mail in index i-hamSize contains the word in the possibleFeaturesList in index j
				// then the bernoulli value for that feature and that mail is 1, else it is 0
				if(mailWordsListSpam.get(i-hamSize).contains(possibleFeaturesList.get(j))){
					bernoulliValues[i][j]=1;
				}else {
					bernoulliValues[i][j]=0;
				}
			}
			// Set last value (column) to 1 for spam
			bernoulliValues[i][pfeatSize] = 1;
		}
		
		// Calculate probability and entropy for C=1
		// P(C=1):
		double PC = (hamSize / (trainMails.size() * 1.0));
		// H(C):
		double HC = entropy(PC);
		
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
			entropyOfFeaturesX1[j] = entropy(featureX1SpamProbabilities[j]);
			// Calculate H(C | X=0) for the possible feature
			entropyOfFeaturesX0[j] = entropy(featureX0SpamProbabilities[j]);
			// Calculate the IG for the possible feature using the probabilities and the entropies above
			informationGainOfFeatures[j] = HC - (featureProbabilities[j]*entropyOfFeaturesX1[j] + (1-featureProbabilities[j])*entropyOfFeaturesX0[j]);
		}
		
		// Create an ArrayList of Feature objects
		ArrayList<Feature> possiblefeaturesObj = new ArrayList<>();

		// Convert the features to objects in order to sort them
		// For j from 0 to the size of the possible features
		for(int j=0;j<pfeatSize;j++) {
			// Add to the arraylist of feature objects a new feature object using feature's name and IG from the corresponding list  
			possiblefeaturesObj.add(new Feature(possibleFeaturesList.get(j),informationGainOfFeatures[j]));
		}
		
		// Sort array list of Features in descending order
		Collections.sort(possiblefeaturesObj, Collections.reverseOrder());
		
		// Initialize the feature List that will be returned
		ArrayList<Feature> features = new ArrayList<>();
		
		// Get the first n features and add them to the feature list
		for(int i=0;i<n;i++) {
			features.add(possiblefeaturesObj.get(i));
		}
		
		// Write the features to a file
		
		// Initialize a buffered writer to null
		BufferedWriter bw = null;
		try {
			// Create the buffered writer using the file "features.txt" that will be created/updated
			bw = new BufferedWriter(new FileWriter(new File("features_"+ seed +".txt")));
			// For each feature in the features list
			for(Feature f : features) {
				// Write to the file it's name and IG using a tidle "~" as a separator
				bw.write(f.name + "~" + f.IG + "\n");
			}
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
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
		
		// Return the features list
		return features;
	}
	
	// Create arffFile for comparison with Weka
	public static void createArffFile(ArrayList<Mail> mails,ArrayList<Feature> features,String fileName){
		// Initialize a file writer to null
		FileWriter fw = null;
		try {
		// Create a file using filename
		fw = new FileWriter(new File(fileName));
		// Write the relation tag
		fw.write("@relation mail\n\n");
		// For every feature
		for(Feature f : features) {
			// If the feature is the percentage symbol and because it is a comment in arff files
			if(f.name.trim().equals("%")) {
				// Instead write _PERCENT_
				fw.write("@attribute " + "_PERCENT_" +" {0, 1}\n");
			}else {
				// Else write the atrribute tag with the feature's name and possible values (bernoulli)
				fw.write("@attribute " + f.name +" {0, 1}\n");
			}
		}
		// Write the last attribute tag for the class and the possible values (0-ham, 1-spam)
		fw.write("@attribute spam {0, 1}\n\n");
		// Write the data tag
		fw.write("@data\n");
		// Find the bernoulli table for the trainMails and the feature list
		int[][] bernoulliTable = findBernoulliTable(mails, features);
		// For i from 0 to the rows of the bernoulli Table
		for(int i=0;i<bernoulliTable.length;i++) {
			// For j from 0 to the columns of the bernoulli Table
			for(int j=0;j<features.size();j++) {
				// Write the data for each feature (value 0 or 1 for existence in the specific mail) and a comma
				fw.write(bernoulliTable[i][j] + ",");
			}
			// Write the class of the specific mail
			fw.write(bernoulliTable[i][bernoulliTable[0].length-1] + "\n");
		}
		}catch(IOException e) {
			e.printStackTrace();
		}finally {
			// If the file writer is not null
			if(fw!=null) {
				try {
					// CLose the file writer
					fw.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
		
	}
	
	// Method that calculates the entropy for a bernoulli variable (0 or 1) using probability p
	public static double entropy(double p) {
		// If p is 0 or 1 then we are certain of the outcome, so the entropy is 0
		if(p==0 || p==1) return 0;
		// Else calculate using the formula
		return - (p * (Math.log(p)/Math.log(2))) - ((1-p) * (Math.log((1-p))/Math.log(2))) ;
	}
	
	// Method that creates a bernoulli values table from the trainMails and the feature list,which is a two dimensional
	// array that has values {0,1} whether a feature for a specific mail exists or not, as well as in the last column
	// the category of the mail (0 for ham, 1 for spam)
	// Returns the bernoulli table
	public static int[][] findBernoulliTable(ArrayList<Mail> trainMails, ArrayList<Feature> features){
		
		ArrayList<HashSet<String>> mailWords = new ArrayList<>();
		
		// For each mail in the trainMails
		for(Mail m : trainMails) {
			// Get the text from the mail -- to uppercase and crop the first 9 characters ("Subject: ") as it is not important
			String text = m.getText().toUpperCase().substring(9);
			// Get the mail's words using a delimiter for all whitespace characters and add them to a list
			ArrayList<String> words = new ArrayList<>( Arrays.asList(text.split("\\s+"))) ;
			// Add the mail's words to the list as a hashSet according to ham / spam
			mailWords.add(new HashSet<String>(words));
		}
		
		int featSize = features.size();
		int mailSize = trainMails.size();
		
		// Create a 2D array for the bernoulli values for each mail and each feature +1 for the spam/ham value
		int[][] bernoulliValuesTable = new int[trainMails.size()][featSize+1];
		
		// For i from 0 to the size of the ham mails
		for(int i=0;i<mailSize;i++) {
			// For j from 0 to the size of the features
			for(int j=0;j<featSize;j++) {
				// If the ham mail in index i contains the word in the features in index j
				// then the bernoulli value for that feature and that mail is 1, else it is 0
				if(mailWords.get(i).contains(features.get(j).name)){
					bernoulliValuesTable[i][j]=1;
				}else {
					bernoulliValuesTable[i][j]=0;
				}
			}
			// Set last value (column) to 0 for ham or 1 for spam
			if(trainMails.get(i).isSpam()) {
				bernoulliValuesTable[i][featSize] = 1;
			}else {
				bernoulliValuesTable[i][featSize] = 0;
			}
		}
		
		// Return the bernoulli Values Table
		return bernoulliValuesTable;
	}
	
	// Method that searches a folder provided for the subfolders ham, spam from which it searches for all the files (which are mails) and creates
	// a list of ham and a list of spam mails. After that regarding the ham_spam ratio given in the parameters, it distributes the mails into the
	// three lists (train, validate, test - with ratio 80%-10%-10%), choosing either from the spam list or the ham list.
	// All the random functions, which are used are given either a random seed (if useSeed is false) or a specific provided
	public static void getMails(String folderN,ArrayList<Mail> trainMails,ArrayList<Mail> validateMails,ArrayList<Mail> testMails,double ham_spam,int seed,boolean useSeed) {
	
		// Get the subfolder ham
		File hamFolder = new File(folderN + "//ham//");
		// Get the subfolder spam
		File spamFolder = new File(folderN + "//spam//");
		// Get all the files from the ham folder
		File[] hamFiles = hamFolder.listFiles();
		// Get all the files from the spam folder
		File[] spamFiles = spamFolder.listFiles();
		
		// Initialize the two lists - ham list , spam list
		ArrayList<Mail> hamMails = new ArrayList<>();
		ArrayList<Mail> spamMails = new ArrayList<>();
		
		// For each file in the hamFiles
		for(File file : hamFiles) {
			// If it is a file
			if(file.isFile()) {
				// Create a string builder for the mail's text
				StringBuilder text = new StringBuilder();
				// Initialize a string for each line
				String line = null;
				// Initialize a buffered Reader
				BufferedReader br = null;
				try {
					// Create the buffered reader using the file
					br = new BufferedReader(new FileReader(file));
					// While there are more lines
					while((line = br.readLine())!= null) {
						// Append the line to the text
						text.append(line);
						// Append a new line to the text
						text.append('\n');
					}
				} catch (IOException e) {
					e.printStackTrace();
				} finally {
					// If the buffered reader isn't null
					if(br!=null) {
						try {
							// Close the buffered reader
							br.close();
						} catch (IOException e) {
							e.printStackTrace();
						}
					}
				}
				// Add a new Mail to the ham list using the text and the category of the file (false)
				hamMails.add(new Mail(text.toString(),false));
			}
		}
		
		// For each file in the spamFiles
		for(File file : spamFiles) {
			// If it is a file
			if(file.isFile()) {
				// Create a string builder for the mail's text
				StringBuilder text = new StringBuilder();
				// Initialize a string for each line
				String line = null;
				// Initialize a buffered Reader
				BufferedReader br = null;
				try {
					// Create the buffered reader using the file
					br = new BufferedReader(new FileReader(file));
					// While there are more lines
					while((line = br.readLine())!= null) {
						// Append the line to the text
						text.append(line);
						// Append a new line to the text
						text.append('\n');
					}
				} catch (IOException e) {
					e.printStackTrace();
				} finally {
					// If the buffered reader isn't null
					if(br!=null) {
						try {
							// Close the buffered reader
							br.close();
						} catch (IOException e) {
							e.printStackTrace();
						}
					}
				}
				// Add a new Mail to the spam list using the text and the category of the file (true)
				spamMails.add(new Mail(text.toString(),true));
			}
		}
		
		// Print message to console about the mails count
		System.out.println("All mails are: " + (hamMails.size()+spamMails.size()));
		
		// Initialize a random object
		Random rand = null;
		// If useSeed is true
		if(useSeed) {
			// Use the seed variable for the random generator
			rand = new Random(seed);
		}else {
			// Don't use a seed for the random generator
			rand = new Random();
		}
		
		// Distribute the mails to train, validate and test by 80%,10%,10%
		
		// Calculate the trainMails size
		int trainCount = (int) Math.ceil((hamMails.size()+spamMails.size())*0.8);
		// Print the size to the console
		System.out.println("Train mails: " + trainCount);
		// Calculate the validateMails size
		int validCount = (int) Math.floor((hamMails.size()+spamMails.size())*0.1);
		// Print the size to the console
		System.out.println("Validate mails: " + validCount);
		// Calculate the testMails size
		int testCount = (int) Math.floor((hamMails.size()+spamMails.size())*0.1);
		// Print the size to the console
		System.out.println("Test mails: " + testCount);
		
		// Fill the testMails list
		for(int i=0;i<testCount;i++) {
			// Get a random value from 0 to 1
			double choice = rand.nextDouble();
			// If the spamMails list is empty OR the number is smaller than the ham_spam ratio (probability) and the hamMails list isn't empty
			if(spamMails.isEmpty() || (choice < ham_spam && !hamMails.isEmpty())) {
				// Take a random Mail from hamMails list and add it to the testMails list
				testMails.add(hamMails.remove(rand.nextInt(hamMails.size())));
			}else if(!spamMails.isEmpty()) {
				// Take a random Mail from spamMails list and add it to the testMails list
				testMails.add(spamMails.remove(rand.nextInt(spamMails.size())));
			}else {
				// The algorithm must never go here, if it gets here then there is an error,
				// so print the error
				System.out.println("both empty");
			}
		}
		
		// Fill the validateMails list
		for(int i=0;i<validCount;i++) {
			// Get a random value from 0 to 1
			double choice = rand.nextDouble();
			// If the spamMails list is empty OR the number is smaller than the ham_spam ratio (probability) and the hamMails list isn't empty
			if(spamMails.isEmpty() || (choice < ham_spam && !hamMails.isEmpty())) {
				// Take a random Mail from hamMails list and add it to the validateMails list
				validateMails.add(hamMails.remove(rand.nextInt(hamMails.size())));
			}else if(!spamMails.isEmpty()) {
				// Take a random Mail from spamMails list and add it to the validateMails list
				validateMails.add(spamMails.remove(rand.nextInt(spamMails.size())));
			}else {
				// The algorithm must never go here, if it gets here then there is an error,
				// so print the error
				System.out.println("Error: Both empty");
			}
		}
		
		// Fill the trainMails list
		for(int i=0;i<trainCount;i++) {
			// Get a random value from 0 to 1
			double choice = rand.nextDouble();
			// If the spamMails list is empty OR the number is smaller than the ham_spam ratio (probability) and the hamMails list isn't empty
			if(spamMails.isEmpty() || (choice < ham_spam && !hamMails.isEmpty())) {
				// Take a random Mail from hamMails list and add it to the trainMails list
				trainMails.add(hamMails.remove(rand.nextInt(hamMails.size())));
			}else if(!spamMails.isEmpty()) {
				// Take a random Mail from spamMails list and add it to the trainMails list
				trainMails.add(spamMails.remove(rand.nextInt(spamMails.size())));
			}else {
				// The algorithm must never go here, if it gets here then there is an error,
				// so print the error
				System.out.println("both empty");
			}
		}
		
	}

}
