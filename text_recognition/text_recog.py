import cv2
import numpy as np
import pytesseract  # This is for OCR(Optical Character Recignition)
import cs50

# It is the configration of tesseract that specifies ythe location of tesseract in our pc.
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


img = cv2.imread("anh.png")


# This means we are not specifying pixels directly but we are specifying percentages this means width made 0.5 times the previous and same with the height.
img = cv2.resize(img, None, fx=1, fy=1)


# Converting to grayscale because images having different light intensiies at different locations can't be detected clearly.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# Now we will use adaptive threshold to specify a particular value below this value background and above this value text by using adaptive threshold we can easily.
# remove the background that is basically not homogenous i.e not all parts of image have same light intensity
adaptive_threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 85, 11)

# page segmentation mode values are:
# 0 = Orientation and Script detection only(OSD).
# 1 = Automatic page segmentation with OSD.
# 2 = Automatic Page segmentation, but no OSD or OCR.
# 3 = Fully automatic page segmentation, but no OSD.(default)
# 4 = Assume single column of text of variable sizes.
# 5 = Assume a single uniform block of vertically aligned text.
# 6 = Assume single uniform block of text.
# 7 = Treat the image as single text line.
# 8 = Treat the image as single word.
# 9 = Treat the image as single word in a circle.
# 10 = Treat the image as single character.


# Now we choose page segmentation mode
config = "--psm 3"

# Actually by doing this we can really detect the clear images very well but not others.
text = pytesseract.image_to_string(img, config=config)
print(text)




cv2.imshow("Book", adaptive_threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()