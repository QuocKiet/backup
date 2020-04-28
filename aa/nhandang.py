import cv2
import numpy as np
import math
import PossibleChar
import random
import PossiblePlate
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
import joblib

import os


MIN_PIXEL_WIDTH = 2
MIN_PIXEL_HEIGHT = 8

MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 1.0



MIN_PIXEL_AREA = 80

# constants for comparing two chars
MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.3
MAX_DIAG_SIZE_MULTIPLE_AWAY = 5.0

MAX_CHANGE_IN_AREA = 0.5

MAX_CHANGE_IN_WIDTH = 0.8
MAX_CHANGE_IN_HEIGHT = 0.2

MAX_ANGLE_BETWEEN_CHARS = 12.0

        # other constants
MIN_NUMBER_OF_MATCHING_CHARS = 3

RESIZED_CHAR_IMAGE_WIDTH = 20
RESIZED_CHAR_IMAGE_HEIGHT = 30

PLATE_WIDTH_PADDING_FACTOR = 1.1
PLATE_HEIGHT_PADDING_FACTOR = 1.5

MIN_CONTOUR_AREA = 100
SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)


GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
ADAPTIVE_THRESH_BLOCK_SIZE = 21
ADAPTIVE_THRESH_WEIGHT = 9


RESIZED_CHAR_IMAGE_WIDTH = 20
RESIZED_CHAR_IMAGE_HEIGHT = 30

# KNN

clfChar = joblib.load('datanewChar1.pkl')

clfDigit = joblib.load('datanewDigit1.pkl')

def extractValue(imgOriginal):
    height, width, numChannels = imgOriginal.shape

    imgHSV = np.zeros((height, width, 3), np.uint8)

    imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)

    imgHue, imgSaturation, imgValue = cv2.split(imgHSV)

    return imgValue
# end function

###################################################################################################
def maximizeContrast(imgGrayscale):

    height, width = imgGrayscale.shape

    imgTopHat = np.zeros((height, width, 1), np.uint8)
    imgBlackHat = np.zeros((height, width, 1), np.uint8)

    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement)

    imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

    return imgGrayscalePlusTopHatMinusBlackHat
# end function

def preprocess1(imgOriginal):
	im_gray = cv2.cvtColor(imgOriginal,cv2.COLOR_BGR2GRAY)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
	morph_image = cv2.morphologyEx(im_gray,cv2.MORPH_BLACKHAT,kernel)
	morph_image1 = cv2.morphologyEx(im_gray,cv2.MORPH_TOPHAT,kernel)
	im2 = cv2.add(im_gray, morph_image1)
	im3 = cv2.subtract(im2, morph_image)
# cv2.imshow("20",im3)
	im3 = cv2.GaussianBlur(im3, (5, 5), 0)
	imgThresh = cv2.adaptiveThreshold(im3, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 9)
	return im_gray, imgThresh

def preprocess(imgOriginal):
    # imgGrayscale = extractValue(imgOriginal)

    imgGrayscale = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)
    imgMaxContrastGrayscale = maximizeContrast(imgGrayscale)

    height, width = imgGrayscale.shape

    # imgBlurred = np.zeros((height, width, 1), np.uint8)

    imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)

    imgThresh = cv2.adaptiveThreshold(imgBlurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)

    return imgGrayscale, imgThresh

###################################################################################################
# tìm tất cả vị trí char của bản số
def findPossibleCharsInScene(imgThresh):
    listOfPossibleChars = []                # this will be the return value

    intCountOfPossibleChars = 0

    imgThreshCopy = imgThresh.copy()

    imgContours, contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)   # find all contours

    height, width = imgThresh.shape
    imgContours = np.zeros((height, width, 3), np.uint8)

    for i in range(0, len(contours)):                       # for each contour

        # if Main.showSteps == True: # show steps ###################################################
        cv2.drawContours(imgContours, contours, i, SCALAR_RED)
        # end if # show steps #####################################################################
       	# gán tọa độ của char vào list possiblechar
        possibleChar = PossibleChar.PossibleChar(contours[i])
        if checkIfPossibleChar(possibleChar):                   # if contour is a possible char, note this does not compare to other chars (yet) . . .
            intCountOfPossibleChars = intCountOfPossibleChars + 1           # increment count of possible chars
            listOfPossibleChars.append(possibleChar)                        # and add to list of possible chars
        # end if
    # end for
    # 
    # print(intCountOfPossibleChars)
    # cv2.imshow("imgContours",imgContours)
    return listOfPossibleChars
# end function


def findPossibleCharsInPlate(imgGrayscale, imgThresh):
    listOfPossibleChars = []                        # this will be the return value
    contours = []
    imgThreshCopy = imgThresh.copy()

            # find all contours in plate
    imgContours, contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:                        # for each contour
        possibleChar = PossibleChar.PossibleChar(contour)

        if checkIfPossibleChar(possibleChar):              # if contour is a possible char, note this does not compare to other chars (yet) . . .
            listOfPossibleChars.append(possibleChar)       # add to list of possible chars
        # end if
    # end if

    return listOfPossibleChars
# end function


###################################################################################################
def extractPlate(imgOriginal, listOfMatchingChars):
    possiblePlate = PossiblePlate.PossiblePlate()           # this will be the return value

    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)        # sort chars from left to right based on x position

            # calculate the center point of the plate
    fltPlateCenterX = (listOfMatchingChars[0].intCenterX + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterX) / 2.0
    fltPlateCenterY = (listOfMatchingChars[0].intCenterY + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY) / 2.0

    ptPlateCenter = fltPlateCenterX, fltPlateCenterY

            # calculate plate width and height
    intPlateWidth = int((listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectX + listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectWidth - listOfMatchingChars[0].intBoundingRectX) * PLATE_WIDTH_PADDING_FACTOR)

    intTotalOfCharHeights = 0

    for matchingChar in listOfMatchingChars:
        intTotalOfCharHeights = intTotalOfCharHeights + matchingChar.intBoundingRectHeight
    # end for

    fltAverageCharHeight = intTotalOfCharHeights / len(listOfMatchingChars)

    intPlateHeight = int(fltAverageCharHeight * PLATE_HEIGHT_PADDING_FACTOR)

            # calculate correction angle of plate region
    fltOpposite = listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY - listOfMatchingChars[0].intCenterY
    fltHypotenuse = distanceBetweenChars(listOfMatchingChars[0], listOfMatchingChars[len(listOfMatchingChars) - 1])
    fltCorrectionAngleInRad = math.asin(fltOpposite / fltHypotenuse)
    fltCorrectionAngleInDeg = fltCorrectionAngleInRad * (180.0 / math.pi)

            # pack plate region center point, width and height, and correction angle into rotated rect member variable of plate
    possiblePlate.rrLocationOfPlateInScene = ( tuple(ptPlateCenter), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg )

            # final steps are to perform the actual rotation

            # get the rotation matrix for our calculated correction angle
    rotationMatrix = cv2.getRotationMatrix2D(tuple(ptPlateCenter), fltCorrectionAngleInDeg, 1.0)

    height, width, numChannels = imgOriginal.shape      # unpack original image width and height

    imgRotated = cv2.warpAffine(imgOriginal, rotationMatrix, (width, height))       # rotate the entire image

    imgCropped = cv2.getRectSubPix(imgRotated, (intPlateWidth, intPlateHeight), tuple(ptPlateCenter))

    possiblePlate.imgPlate = imgCropped         # copy the cropped plate image into the applicable member variable of the possible plate

    return possiblePlate
# end function

# use Pythagorean theorem to calculate distance between two chars
def distanceBetweenChars(firstChar, secondChar):
    intX = abs(firstChar.intCenterX - secondChar.intCenterX)
    intY = abs(firstChar.intCenterY - secondChar.intCenterY)

    return math.sqrt((intX ** 2) + (intY ** 2))
# end function

###################################################################################################
# use basic trigonometry (SOH CAH TOA) to calculate angle between chars
def angleBetweenChars(firstChar, secondChar):
    fltAdj = float(abs(firstChar.intCenterX - secondChar.intCenterX))
    fltOpp = float(abs(firstChar.intCenterY - secondChar.intCenterY))

    if fltAdj != 0.0:                           # check to make sure we do not divide by zero if the center X positions are equal, float division by zero will cause a crash in Python
        fltAngleInRad = math.atan(fltOpp / fltAdj)      # if adjacent is not zero, calculate angle
    else:
        fltAngleInRad = 1.5708                          # if adjacent is zero, use this as the angle, this is to be consistent with the C++ version of this program
    # end if

    fltAngleInDeg = fltAngleInRad * (180.0 / math.pi)       # calculate angle in degrees

    return fltAngleInDeg
# end function



def findListOfListsOfMatchingChars(listOfPossibleChars):
            # with this function, we start off with all the possible chars in one big list
            # the purpose of this function is to re-arrange the one big list of chars into a list of lists of matching chars,
            # note that chars that are not found to be in a group of matches do not need to be considered further
    listOfListsOfMatchingChars = []                  # this will be the return value

    for possibleChar in listOfPossibleChars:                        # for each possible char in the one big list of chars
        listOfMatchingChars = findListOfMatchingChars(possibleChar, listOfPossibleChars)        # find all chars in the big list that match the current char

        listOfMatchingChars.append(possibleChar)                # also add the current char to current possible list of matching chars

        if len(listOfMatchingChars) < MIN_NUMBER_OF_MATCHING_CHARS:     # if current possible list of matching chars is not long enough to constitute a possible plate
            continue                            # jump back to the top of the for loop and try again with next char, note that it's not necessary
                                                # to save the list in any way since it did not have enough chars to be a possible plate
        # end if

                                                # if we get here, the current list passed test as a "group" or "cluster" of matching chars
        listOfListsOfMatchingChars.append(listOfMatchingChars)      # so add to our list of lists of matching chars

        listOfPossibleCharsWithCurrentMatchesRemoved = []

                                                # remove the current list of matching chars from the big list so we don't use those same chars twice,
                                                # make sure to make a new big list for this since we don't want to change the original big list
        listOfPossibleCharsWithCurrentMatchesRemoved = list(set(listOfPossibleChars) - set(listOfMatchingChars))

        recursiveListOfListsOfMatchingChars = findListOfListsOfMatchingChars(listOfPossibleCharsWithCurrentMatchesRemoved)      # recursive call

        for recursiveListOfMatchingChars in recursiveListOfListsOfMatchingChars:        # for each list of matching chars found by recursive call
            listOfListsOfMatchingChars.append(recursiveListOfMatchingChars)             # add to our original list of lists of matching chars
        # end for

        break       # exit for

    # end for

    return listOfListsOfMatchingChars
# end function

###################################################################################################
def findListOfMatchingChars(possibleChar, listOfChars):
            # the purpose of this function is, given a possible char and a big list of possible chars,
            # find all chars in the big list that are a match for the single possible char, and return those matching chars as a list
    listOfMatchingChars = []                # this will be the return value

    for possibleMatchingChar in listOfChars:                # for each char in big list
        if possibleMatchingChar == possibleChar:    # if the char we attempting to find matches for is the exact same char as the char in the big list we are currently checking
                                                    # then we should not include it in the list of matches b/c that would end up double including the current char
            continue                                # so do not add to list of matches and jump back to top of for loop
        # end if
                    # compute stuff to see if chars are a match
        fltDistanceBetweenChars = distanceBetweenChars(possibleChar, possibleMatchingChar)

        fltAngleBetweenChars = angleBetweenChars(possibleChar, possibleMatchingChar)

        fltChangeInArea = float(abs(possibleMatchingChar.intBoundingRectArea - possibleChar.intBoundingRectArea)) / float(possibleChar.intBoundingRectArea)

        fltChangeInWidth = float(abs(possibleMatchingChar.intBoundingRectWidth - possibleChar.intBoundingRectWidth)) / float(possibleChar.intBoundingRectWidth)
        fltChangeInHeight = float(abs(possibleMatchingChar.intBoundingRectHeight - possibleChar.intBoundingRectHeight)) / float(possibleChar.intBoundingRectHeight)

                # check if chars match
        if (fltDistanceBetweenChars < (possibleChar.fltDiagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY) and
            fltAngleBetweenChars < MAX_ANGLE_BETWEEN_CHARS and
            fltChangeInArea < MAX_CHANGE_IN_AREA and
            fltChangeInWidth < MAX_CHANGE_IN_WIDTH and
            fltChangeInHeight < MAX_CHANGE_IN_HEIGHT):

            listOfMatchingChars.append(possibleMatchingChar)        # if the chars are a match, add the current char to list of matching chars
        # end if
    # end for

    return listOfMatchingChars                  # return result
# end function

def removeInnerOverlappingChars(listOfMatchingChars):
    listOfMatchingCharsWithInnerCharRemoved = list(listOfMatchingChars)                # this will be the return value

    for currentChar in listOfMatchingChars:
        for otherChar in listOfMatchingChars:
            if currentChar != otherChar:        # if current char and other char are not the same char . . .
                                                                            # if current char and other char have center points at almost the same location . . .
                if distanceBetweenChars(currentChar, otherChar) < (currentChar.fltDiagonalSize * MIN_DIAG_SIZE_MULTIPLE_AWAY):
                                # if we get in here we have found overlapping chars
                                # next we identify which char is smaller, then if that char was not already removed on a previous pass, remove it
                    if currentChar.intBoundingRectArea < otherChar.intBoundingRectArea:         # if current char is smaller than other char
                        if currentChar in listOfMatchingCharsWithInnerCharRemoved:              # if current char was not already removed on a previous pass . . .
                            listOfMatchingCharsWithInnerCharRemoved.remove(currentChar)         # then remove current char
                        # end if
                    else:                                                                       # else if other char is smaller than current char
                        if otherChar in listOfMatchingCharsWithInnerCharRemoved:                # if other char was not already removed on a previous pass . . .
                            listOfMatchingCharsWithInnerCharRemoved.remove(otherChar)           # then remove other char
                        # end if
                    # end if
                # end if
            # end if
        # end for
    # end for

    return listOfMatchingCharsWithInnerCharRemoved


def checkIfPossibleChar(possibleChar):
            # this function is a 'first pass' that does a rough check on a contour to see if it could be a char,
            # note that we are not (yet) comparing the char to other chars to look for a group
    if (possibleChar.intBoundingRectArea > MIN_PIXEL_AREA and
        possibleChar.intBoundingRectWidth > MIN_PIXEL_WIDTH and possibleChar.intBoundingRectHeight > MIN_PIXEL_HEIGHT and
        MIN_ASPECT_RATIO < possibleChar.fltAspectRatio and possibleChar.fltAspectRatio < MAX_ASPECT_RATIO):
        return True
    else:
        return False


# tham số truyền vào là tất cả danh sách plate có thể tìm đc
def detectCharsInPlates(listOfPossiblePlates):
    intPlateCounter = 0
    imgContours = None
    contours = []
    rsContours =[]

    if len(listOfPossiblePlates) == 0:          # if list of possible plates is empty
        return listOfPossiblePlates             # return
    # end if

            # at this point we can be sure the list of possible plates has at least one plate
    j = 0
    for possiblePlate in listOfPossiblePlates:          # for each possible plate, this is a big for loop that takes up most of the function

        possiblePlate.imgGrayscale, possiblePlate.imgThresh = preprocess(possiblePlate.imgPlate)     # preprocess to get grayscale and threshold images
        # cv2.imshow("5a", possiblePlate.imgPlate)
        # cv2.imshow("5b", possiblePlate.imgGrayscale)
        # cv2.imshow("5c", possiblePlate.imgThresh)
        # cv2.waitKey(0)

                # increase size of plate image for easier viewing and char detection
        possiblePlate.imgThresh = cv2.resize(possiblePlate.imgThresh, (0, 0), fx = 1.6, fy = 1.6)

                # threshold again to eliminate any gray areas
        thresholdValue, possiblePlate.imgThresh = cv2.threshold(possiblePlate.imgThresh, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


                # find all possible chars in the plate,
                # this function first finds all contours, then only includes contours that could be chars (without comparison to other chars yet)
        listOfPossibleCharsInPlate = findPossibleCharsInPlate(possiblePlate.imgGrayscale, possiblePlate.imgThresh)
        listOfListsOfMatchingCharsInPlate = findListOfListsOfMatchingChars(listOfPossibleCharsInPlate)
       	height, width, numChannels = possiblePlate.imgPlate.shape
        imgContours = np.zeros((height, width, 3), np.uint8)
        # print(len(listOfListsOfMatchingCharsInPlate))
        print('########')
        for i in range(0, len(listOfListsOfMatchingCharsInPlate)):
        	                             # within each list of matching chars
        	
        	listOfListsOfMatchingCharsInPlate[i].sort(key = lambda matchingChar: matchingChar.intCenterX)        # sort chars from left to right
        	listOfListsOfMatchingCharsInPlate[i] = removeInnerOverlappingChars(listOfListsOfMatchingCharsInPlate[i])              # and remove inner overlapping chars
        # end for
        for listOfMatchingChars in listOfListsOfMatchingCharsInPlate:
        	intRandomBlue = random.randint(0, 255)
        	intRandomGreen = random.randint(0, 255)
        	intRandomRed = random.randint(0, 255)
        	for matchingChar in listOfMatchingChars:
        		contours.append(matchingChar.contour)
        		# break
        	# contours.append(listOfMatchingChars[1].contour)
        		# cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))
       		# cv2.imshow(str(j),imgContours)
        	# contours = []
       		# j += 1

        intLenOfLongestListOfChars = 0
        intIndexOfLongestListOfChars = 0

                # loop through all the vectors of matching chars, get the index of the one with the most chars
        # print(len(listOfListsOfMatchingCharsInPlate))
        for i in range(0, len(listOfListsOfMatchingCharsInPlate)):
            if len(listOfListsOfMatchingCharsInPlate[i]) > intLenOfLongestListOfChars:
                intLenOfLongestListOfChars = len(listOfListsOfMatchingCharsInPlate[i])
                intIndexOfLongestListOfChars = i
            # end if
        # end for

                # suppose that the longest list of matching chars within the plate is the actual list of chars
        print("index" + str(intIndexOfLongestListOfChars) +"\n" )
        print(len(listOfListsOfMatchingCharsInPlate))
        
        if (len(listOfListsOfMatchingCharsInPlate) == 0):
        	possiblePlate.strChars = ""
        	continue
        longestListOfMatchingCharsInPlate = listOfListsOfMatchingCharsInPlate[intIndexOfLongestListOfChars]
        imgContours = np.zeros((height, width, 3), np.uint8)
        del contours[:]
        for matchingChar in longestListOfMatchingCharsInPlate:
        	contours.append(matchingChar.contour)
        cv2.drawContours(imgContours, contours, -1,SCALAR_RED)

        # cv2.imshow("test", imgContours)
        j+=1
        print("so contours" + str(len(contours)))
        if(len(contours) >=6 and len(contours) <= 8):
        	# rsContours = contours
        	# imgtest = np.zeros((1000, 1000, 3), np.uint8)
        	# cv2.drawContours(imgtest, rsContours, -1,SCALAR_RED)
        	# cv2.imshow('hahah',imgtest)
        	# break
        	possiblePlateRs = PossiblePlate.PossiblePlate()
        	
        	possiblePlateRs.imgPlate = possiblePlate.imgPlate
        	possiblePlateRs.contour = contours
        	possiblePlateRs.imgThresh = possiblePlate.imgThresh

        	# cv2.imshow('PossiblePlate', possiblePlateRs.imgThresh)
        	# print(len(possiblePlateRs.contour))
        	# print(possiblePlateRs.contour)
        	return possiblePlateRs
        
def detectPlatesInScene(imgOriginalScene):
	listOfPossiblePlates = []
	height, width, numChannels = imgOriginalScene.shape
	imgGrayscaleScene = np.zeros((height, width, 1), np.uint8)
	imgThreshScene = np.zeros((height, width, 1), np.uint8)
	imgContours = np.zeros((height, width, 3), np.uint8)
	# tiền xử lý
	imgGrayscaleScene, imgThreshScene = preprocess(imgOriginalScene)
	
	# tìm tất cả các char trong hình
	listOfPossibleCharsInScene = findPossibleCharsInScene(imgThreshScene)	
	
	imgContours = np.zeros((height, width, 3), np.uint8)
	# add contours
	contours = []
	for possibleChar in listOfPossibleCharsInScene:
		contours.append(possibleChar.contour)
	# phân nhóm các char
	listOfListsOfMatchingCharsInScene = findListOfListsOfMatchingChars(listOfPossibleCharsInScene)
	# show step
	imgContours = np.zeros((height, width, 3), np.uint8)
	for listOfMatchingChars in listOfListsOfMatchingCharsInScene:
		intRandomBlue = random.randint(0, 255)
		intRandomGreen = random.randint(0, 255)
		intRandomRed = random.randint(0, 255)
		contours = []
		for matchingChar in listOfMatchingChars:
			contours.append(matchingChar.contour)
		cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))
	# cv2.imshow("3", imgContours)
	# end show step


	# trong mõi group của char
	for listOfMatchingChars in listOfListsOfMatchingCharsInScene:
		# tìm vị trí của bản số tương ứng với group char
		possiblePlate = extractPlate(imgOriginalScene, listOfMatchingChars)
		# nếu vị trí của bản số có tồn tại thì add vào danh sách possiblePlate
		if possiblePlate.imgPlate is not None:
			listOfPossiblePlates.append(possiblePlate)
	print("\n" + str(len(listOfPossiblePlates)) + " vị trí bản số được tìm thấy")
	#show step 
	# for i in range(0, len(listOfPossiblePlates)):
	# 	p2fRectPoints = cv2.boxPoints(listOfPossiblePlates[i].rrLocationOfPlateInScene)
	# 	cv2.line(imgContours, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)
	# 	cv2.line(imgContours, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
	# 	cv2.line(imgContours, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)		
	# 	cv2.line(imgContours, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)
	# 	cv2.imshow("4a", imgContours)
	# 	cv2.imshow("4b", listOfPossiblePlates[i].imgPlate)
	# 	cv2.waitKey(0)


	return listOfPossiblePlates
# chạy thử

def recognize1(char,clf):
    print('vo ham recognize1')
    char = cv2.GaussianBlur(char,(5,5),0)
    char = cv2.resize(char,(60,60))
    char = np.reshape(char, (1,-1))
    temp = clf.predict(char)[-1]
    return temp

def lpr(src):

    img = cv2.imread(src)
    imgGrayscale, imgThresh = preprocess(img)
    listOfPossibleChars =   findPossibleCharsInScene(imgThresh)
    listChar = detectPlatesInScene(img)
    temp = detectCharsInPlates(listChar)
    
    arrayChars = []
    aaaa =0
    for c in temp.contour:
        height, width, numChannels = temp.imgPlate.shape
        imgTemp = cv2.resize(temp.imgPlate, (0, 0), fx = 1.6, fy = 1.6)
        x,y,w,h = cv2.boundingRect(c)
        arrayChars.append(temp.imgThresh[y - 4: y + h + 4 , x - 4 :x + w + 4 ])
        cv2.imwrite('tempImage/'+str(aaaa)+".png",temp.imgThresh[y - 4: y + h + 4 , x - 4 :x + w + 4 ])

    flag = 0
    plate = ''
    if len(arrayChars) != 0:
        for char in arrayChars:
            if(flag != 2):
                tam = recognize1(char,clfDigit)
                plate = plate + tam
            else:
                tam = recognize1(char,clfChar)
                plate = plate + tam
            flag += 1
        print("plate ")
        print(plate)
        cv2.imwrite("Result.jpg", temp.imgThresh)
        return plate
    else:
        return "khong tim thay"

