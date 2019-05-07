// Authors
// Stavropoulos Petros (AM : 3150230)
// Savvidis Konstantinos (AM : 3150229)
// Mpanakos Vasileios (AM : 3140125)

// A class that represents a Mail object
public class Mail {
	
	// The mail's text
	private String text;
	// The mail's category -- true if spam, false if ham
	private boolean spam;
	
	// Parameterized Constructor
	public Mail(String text,boolean spam) {
		this.text = text;
		this.spam = spam;
	}

	// Getter for text
	public String getText() {
		return text;
	}

	// Setter for text
	public void setText(String text) {
		this.text = text;
	}

	// Getter for spam boolean
	public boolean isSpam() {
		return spam;
	}

	// Setter for spam boolean
	public void setSpam(boolean spam) {
		this.spam = spam;
	}
	
}
