// Authors
// Stavropoulos Petros (AM : 3150230)
// Savvidis Konstantinos (AM : 3150229)
// Mpanakos Vasileios (AM : 3140125)

// Class that represents a feature Xi in order to be used by the classifiers
// A feature in the Mails is a word in their text
public class Feature implements Comparable<Feature> {
	
	// The name of the feature (the string value)
	String name;
	// The Information Gain of the feature ( for where it is used )
	double IG;
	
	// Parameterized Constructor
	public Feature(String name,double IG) {
		this.name = name;
		this.IG = IG;
	}
	
	// Override compareTo and return the comparison of their IG
	@Override
	public int compareTo(Feature arg0) {
		return Double.compare(this.IG, arg0.IG);
	}
}