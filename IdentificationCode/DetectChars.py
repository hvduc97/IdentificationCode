# phát hiện kí tự
import os

import cv2
import numpy as np
import math
import random

import Main
import Preprocess
import PossibleChar

# các biến cấp độ mô-đun ##########################################################################

kNearest = cv2.ml.KNearest_create()  #Tạo một mô hình KNN

        # các hằng số cho check IfPossibleChar, điều này chỉ kiểm tra một kí tự có thể (không so sánh với kí tự khác)
MIN_PIXEL_WIDTH = 2
MIN_PIXEL_HEIGHT = 8

MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 1.0

MIN_PIXEL_AREA = 80

        # hằng số để so sánh hai ký tự
MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.3
MAX_DIAG_SIZE_MULTIPLE_AWAY = 5.0

MAX_CHANGE_IN_AREA = 0.5

MAX_CHANGE_IN_WIDTH = 0.8
MAX_CHANGE_IN_HEIGHT = 0.2

MAX_ANGLE_BETWEEN_CHARS = 12.0

        # các hằng số khác
MIN_NUMBER_OF_MATCHING_CHARS = 3

RESIZED_CHAR_IMAGE_WIDTH = 20
RESIZED_CHAR_IMAGE_HEIGHT = 30

MIN_CONTOUR_AREA = 100

###################################################################################################
def loadKNNDataAndTrainKNN():
    allContoursWithData = []                # khai báo một dánh sách trống chứa tất cả các Contour
    validContoursWithData = []              # danh sách trống để chứa các Contour được xác nhận từ danh sách tất cả các Contour

    try:
        npaClassifications = np.loadtxt("classifications.txt", np.float32)                  # đọc file phân loại đào tạo
    except:                                                                                 # nếu file không đọc được.
        print("error, unable to open classifications.txt, exiting program\n")  # in ra thông báo lỗi
        os.system("pause")
        return False                                                                        # và trả về False
    # end try

    try:
        npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)                 # đọc file hình ảnh huấn luyện 
    except:                                                                                 
        print("error, unable to open flattened_images.txt, exiting program\n") 
        os.system("pause")
        return False                                                                       
    # end try

    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))       # reshape mảng numpy thành mảng 1d để huấn luyện dữ liệu

    # kNearest.setDefaultK(1)                                                             # đặt k mặc định là 1

    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)           # huấn luyện mô hình KNN

    return True                             # nếu huấn luyện thành công, trả về True
# end function

###################################################################################################
def detectCharsInPlates(listOfPossiblePlates):
    intPlateCounter = 0
    imgContours = None
    contours = []

    if len(listOfPossiblePlates) == 0:          # nếu danh sách các vùng chứa mã thẻ có thể trống
        return listOfPossiblePlates             # trả về danh sách các vùng chứa mã thẻ có thể
    # end if

    for possiblePlate in listOfPossiblePlates:       # cho mỗi tấm có thể có trong danh sách tấm có thể, đây là một vòng lặp lớn chiếm hầu hết các chức năng
        possiblePlate.imgGrayscale, possiblePlate.imgThresh = Preprocess.preprocess(possiblePlate.imgPlate)     # tiền xử lý để có ảnh thang độ xám ảnh ngưỡng.

        if Main.showSteps == True: # show các bước 
            cv2.imshow("5a", possiblePlate.imgPlate)
            cv2.imshow("5b", possiblePlate.imgGrayscale)
            cv2.imshow("5c", possiblePlate.imgThresh)
        # end if        

                # tăng kích thước của hình ảnh Plate để xem và phát hiện char dễ dàng hơn
        possiblePlate.imgThresh = cv2.resize(possiblePlate.imgThresh, (0, 0), fx = 1.6, fy = 1.6)

                # ngưỡng một lần nữa để loại bỏ bất kỳ khu vực nào màu xám
        thresholdValue, possiblePlate.imgThresh = cv2.threshold(possiblePlate.imgThresh, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        if Main.showSteps == True: # show các bước 
            cv2.imshow("5d", possiblePlate.imgThresh)
        # end if  
                # tìm tất cả các ký tự có thể trong tấm,
        listOfPossibleCharsInPlate = findPossibleCharsInPlate(possiblePlate.imgGrayscale, possiblePlate.imgThresh) #hàm tìm kí tự có trong Plate
        # hàm này trước tiên tìm tất cả các contour, sau đó chỉ giữ lại các contour có thể là ký tự (chưa so sánh với các ký tự khác)
        if Main.showSteps == True: # show các bước
            height, width, numChannels = possiblePlate.imgPlate.shape
            imgContours = np.zeros((height, width, 3), np.uint8)
            del contours[:]                                         # xóa danh sách contours

            for possibleChar in listOfPossibleCharsInPlate:
                contours.append(possibleChar.contour)
            # end for

            cv2.drawContours(imgContours, contours, -1, Main.SCALAR_WHITE) # vẽ contours

            cv2.imshow("6", imgContours)
        # end if 

                # đưa ra một danh sách tất cả các ký tự có thể, tìm các nhóm ký tự khớp trong tấm đó
        listOfListsOfMatchingCharsInPlate = findListOfListsOfMatchingChars(listOfPossibleCharsInPlate)

        if Main.showSteps == True: # show các bước
            imgContours = np.zeros((height, width, 3), np.uint8)
            del contours[:]

            for listOfMatchingChars in listOfListsOfMatchingCharsInPlate:
                intRandomBlue = random.randint(0, 255)
                intRandomGreen = random.randint(0, 255)
                intRandomRed = random.randint(0, 255)

                for matchingChar in listOfMatchingChars:
                    contours.append(matchingChar.contour)
                # end for
                cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))
            # end for
            cv2.imshow("7", imgContours)
        # end if # show các bước

        if (len(listOfListsOfMatchingCharsInPlate) == 0):			# nếu không có nhóm ký tự trùng khớp nào được tìm thấy trong tấm

            if Main.showSteps == True: # show các bước
                print("chars found in plate number " + str(
                    intPlateCounter) + " = (none), click on any image and press a key to continue . . .")
                intPlateCounter = intPlateCounter + 1
                cv2.destroyWindow("8")
                cv2.destroyWindow("9")
                cv2.destroyWindow("10")
                cv2.waitKey(0)
            # end if 

            possiblePlate.strChars = ""
            continue						# quay trở lại đầu vòng lặp
        # end if

        for i in range(0, len(listOfListsOfMatchingCharsInPlate)):                # trong mỗi danh sách các ký tự phù hợp
            listOfListsOfMatchingCharsInPlate[i].sort(key = lambda matchingChar: matchingChar.intCenterX)     # sắp xếp ký tự từ trái sang phải
            # listOfListsOfMatchingCharsInPlate[i] = removeInnerOverlappingChars(listOfListsOfMatchingCharsInPlate[i]) # and remove inner overlapping chars
        # end for

        if Main.showSteps == True: # show các bước
            imgContours = np.zeros((height, width, 3), np.uint8)

            for listOfMatchingChars in listOfListsOfMatchingCharsInPlate:
                intRandomBlue = random.randint(0, 255)
                intRandomGreen = random.randint(0, 255)
                intRandomRed = random.randint(0, 255)

                del contours[:]

                for matchingChar in listOfMatchingChars:
                    contours.append(matchingChar.contour)
                # end for

                cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))
            # end for
            cv2.imshow("8", imgContours)
        # end if 

               # trong mỗi tấm có thể, giả sử danh sách dài nhất của ký tự khớp, tiềm năng là danh sách ký tự thực tế
        intLenOfLongestListOfChars = 0
        intIndexOfLongestListOfChars = 0

                # lặp qua tất cả các vectơ của ký tự khớp, lấy chỉ số của ký tự có nhiều ký tự nhất
        for i in range(0, len(listOfListsOfMatchingCharsInPlate)):
            if len(listOfListsOfMatchingCharsInPlate[i]) > intLenOfLongestListOfChars:
                intLenOfLongestListOfChars = len(listOfListsOfMatchingCharsInPlate[i])
                intIndexOfLongestListOfChars = i
            # end if
        # end for

                # giả sử rằng danh sách dài nhất của các ký tự khớp trong bảng là danh sách các ký tự thực tế
        longestListOfMatchingCharsInPlate = listOfListsOfMatchingCharsInPlate[intIndexOfLongestListOfChars]

        if Main.showSteps == True: # show các bước
            imgContours = np.zeros((height, width, 3), np.uint8)
            del contours[:]

            for matchingChar in longestListOfMatchingCharsInPlate:
                contours.append(matchingChar.contour)
            # end for

            cv2.drawContours(imgContours, contours, -1, Main.SCALAR_WHITE)

            cv2.imshow("9", imgContours)
        # end if 

        possiblePlate.strChars = recognizeCharsInPlate(possiblePlate.imgThresh, longestListOfMatchingCharsInPlate)

        if Main.showSteps == True: # show các bước
            print("chars found in plate number " + str(
                intPlateCounter) + " = " + possiblePlate.strChars + ", click on any image and press a key to continue . . .")
            intPlateCounter = intPlateCounter + 1
            cv2.waitKey(0)
        # end if

    # kết thúc vòng lặp lớn

    if Main.showSteps == True:
        print("\nchar detection complete, click on any image and press a key to continue . . .\n")
        cv2.waitKey(0)
    # end if

    return listOfPossiblePlates
# end function

###################################################################################################
def findPossibleCharsInPlate(imgGrayscale, imgThresh):
    listOfPossibleChars = []                        # đây sẽ là giá trị trả về
    contours = []
    imgThreshCopy = imgThresh.copy()

            # tìm tất cả các Contour trong tấm
    contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:                        # cho mỗi tấm contour
        possibleChar = PossibleChar.PossibleChar(contour)

        if checkIfPossibleChar(possibleChar):        # nếu đường viền là một char có thể, lưu ý rằng điều này không so sánh với các ký tự khác (chưa). . .
            listOfPossibleChars.append(possibleChar)       # thêm vào danh sách các ký tự có thể
        # end if
    # end if

    return listOfPossibleChars
# end function

###################################################################################################
def checkIfPossibleChar(possibleChar):
            # hàm này thực hiện kiểm tra sơ bộ trên một đường viền để xem liệu nó có thể là một char không
    if (possibleChar.intBoundingRectArea > MIN_PIXEL_AREA and
        possibleChar.intBoundingRectWidth > MIN_PIXEL_WIDTH and possibleChar.intBoundingRectHeight > MIN_PIXEL_HEIGHT and
        MIN_ASPECT_RATIO < possibleChar.fltAspectRatio and possibleChar.fltAspectRatio < MAX_ASPECT_RATIO):
        return True
    else:
        return False
    # end if
# end function

###################################################################################################
def findListOfListsOfMatchingChars(listOfPossibleChars):
            # Với hàm này, bắt đầu với tất cả các ký tự có thể có trong một danh sách lớn
            # Mục đích của hàm này là sắp xếp lại một danh sách các ký tự lớn thành một danh sách các danh sách các ký tự khớp
            # note that chars that are not found to be in a group of matches do not need to be considered further
    listOfListsOfMatchingChars = []                  # đây sẽ là giá trị trả về

    for possibleChar in listOfPossibleChars:             # cho mỗi char có thể trong một danh sách lớn các ký tự
        listOfMatchingChars = findListOfMatchingChars(possibleChar, listOfPossibleChars)    # tìm tất cả các ký tự trong danh sách lớn phù hợp với char hiện tại

        listOfMatchingChars.append(possibleChar)                # và thêm char hiện tại vào danh sách các ký tự phù hợp hiện tại có thể
        if len(listOfMatchingChars) < MIN_NUMBER_OF_MATCHING_CHARS:     # nếu danh sách các ký tự trùng khớp hiện tại không đủ dài để tạo thành một tấm có thể
            continue                            # nhảy trở lại đầu vòng lặp for và thử lại với char tiếp theo
                                                
        # end if

                                                # tới đây, danh sách hiện tại đã vượt qua kiểm tra dưới dạng "nhóm" hoặc "cụm" ký tự khớp
        listOfListsOfMatchingChars.append(listOfMatchingChars)      # vì vậy hãy thêm danh sách trên vào danh sách các ký tự phù hợp

        listOfPossibleCharsWithCurrentMatchesRemoved = []

                                                # xóa danh sách các ký tự phù hợp hiện tại khỏi danh sách lớn để không sử dụng lại các ký tự đó hai lần
                                                
        listOfPossibleCharsWithCurrentMatchesRemoved = list(set(listOfPossibleChars) - set(listOfMatchingChars))

        recursiveListOfListsOfMatchingChars = findListOfListsOfMatchingChars(listOfPossibleCharsWithCurrentMatchesRemoved)      # gọi đệ quy
        for recursiveListOfMatchingChars in recursiveListOfListsOfMatchingChars:        # cho mỗi danh sách các ký tự trùng khớp được tìm thấy bằng lệnh gọi đệ quy
            listOfListsOfMatchingChars.append(recursiveListOfMatchingChars)             # thêm vào danh sách ban đầu  danh sách các ký tự phù hợp
        # end for

        break       # exit for

    # end for

    return listOfListsOfMatchingChars
# end function

###################################################################################################
def findListOfMatchingChars(possibleChar, listOfChars):
            # Mục đích của hàm này là, đưa ra một char có thể và một danh sách lớn các ký tự có thể
            # tìm tất cả các ký tự trong danh sách lớn phù hợp với char duy nhất có thể và trả về các ký tự khớp đó dưới dạng danh sách
    listOfMatchingChars = []                #đây sẽ là giá trị trả về

    for possibleMatchingChar in listOfChars:                # cho mỗi char trong danh sách lớn
        if possibleMatchingChar == possibleChar:    # nếu char tìm được có kết quả trùng khớp chính xác với char trong danh sách lớn chúng tôi hiện đang kiểm tra
                                                    
            continue                                # quay lại vòng lặp
        # end if
                    # công cụ tính toán để xem nếu ký tự là một char phù hợp
        #fltDistanceBetweenChars = distanceBetweenChars(possibleChar, possibleMatchingChar)

        fltAngleBetweenChars = angleBetweenChars(possibleChar, possibleMatchingChar)

        fltChangeInArea = float(abs(possibleMatchingChar.intBoundingRectArea - possibleChar.intBoundingRectArea)) / float(possibleChar.intBoundingRectArea)

        fltChangeInWidth = float(abs(possibleMatchingChar.intBoundingRectWidth - possibleChar.intBoundingRectWidth)) / float(possibleChar.intBoundingRectWidth)
        fltChangeInHeight = float(abs(possibleMatchingChar.intBoundingRectHeight - possibleChar.intBoundingRectHeight)) / float(possibleChar.intBoundingRectHeight)

                # kiểm tra xem ký tự khớp hay không
        if (#fltDista< nceBetweenChars (possibleChar.fltDiagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY) and
            fltAngleBetweenChars < MAX_ANGLE_BETWEEN_CHARS and
            fltChangeInArea < MAX_CHANGE_IN_AREA and
            fltChangeInWidth < MAX_CHANGE_IN_WIDTH and
            fltChangeInHeight < MAX_CHANGE_IN_HEIGHT):

            listOfMatchingChars.append(possibleMatchingChar)        # nếu các ký tự trùng khớp, thêm char hiện tại vào danh sách các ký tự khớp
        # end if
    # end for

    return listOfMatchingChars                  # kết quả trả về
# end function

###################################################################################################
# dùng Pytago tính khoảng cách giữa 2 kí tự
def distanceBetweenChars(firstChar, secondChar):
    intX = abs(firstChar.intCenterX - secondChar.intCenterX)
    intY = abs(firstChar.intCenterY - secondChar.intCenterY)

    return math.sqrt((intX ** 2) + (intY ** 2))
# end function

###################################################################################################
# sử dụng công thức lượng giác tính góc giữa các kí tự
def angleBetweenChars(firstChar, secondChar):
    fltAdj = float(abs(firstChar.intCenterX - secondChar.intCenterX))
    fltOpp = float(abs(firstChar.intCenterY - secondChar.intCenterY))

    if fltAdj != 0.0:                           # kiểm tra để đảm bảo rằng chúng ta không chia cho 0 nếu các vị trí trung tâm X bằng nhau, phép chia float cho số 0 sẽ gây ra sự cố trong Python
        fltAngleInRad = math.atan(fltOpp / fltAdj)      # nếu liền kề không bằng 0, tính góc
    else:
        fltAngleInRad = 1.5708                          # Nếu liền kề bằng 0, sử dụng giá trị này làm góc
    # end if

    fltAngleInDeg = fltAngleInRad * (180.0 / math.pi)       # tính góc theo độ

    return fltAngleInDeg
# end function


# Nhận dạng kí tự với data train
def recognizeCharsInPlate(imgThresh, listOfMatchingChars):
    strChars = ""               # Đây sẽ là giá trị trả về, ký tự trong tấm chứa mã thẻ

    height, width = imgThresh.shape

    imgThreshColor = np.zeros((height, width, 3), np.uint8)

    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)        # sắp xếp ký tự từ trái sang phải

    cv2.cvtColor(imgThresh, cv2.COLOR_GRAY2BGR, imgThreshColor)                     # tạo phiên bản màu của hình ảnh ngưỡng để chúng ta có thể vẽ các đường viền màu trên đó
    # cv2.imwrite("10.png", imgThreshColor)

    for currentChar in listOfMatchingChars:                                         # cho mỗi char trong tấm
        pt1 = (currentChar.intBoundingRectX, currentChar.intBoundingRectY)
        pt2 = ((currentChar.intBoundingRectX + currentChar.intBoundingRectWidth), (currentChar.intBoundingRectY + currentChar.intBoundingRectHeight))

        cv2.rectangle(imgThreshColor, pt1, pt2, Main.SCALAR_GREEN, 2)           # vẽ hộp màu xanh xung quanh char

                # cắt char ra khỏi ngưỡng hình ảnh
        imgROI = imgThresh[currentChar.intBoundingRectY : currentChar.intBoundingRectY + currentChar.intBoundingRectHeight,
                           currentChar.intBoundingRectX : currentChar.intBoundingRectX + currentChar.intBoundingRectWidth]

        imgROIResized = cv2.resize(imgROI, (RESIZED_CHAR_IMAGE_WIDTH, RESIZED_CHAR_IMAGE_HEIGHT))           # Thay đổi kích thước hình ảnh, điều này là cần thiết để nhận dạng char

        npaROIResized = imgROIResized.reshape((1, RESIZED_CHAR_IMAGE_WIDTH * RESIZED_CHAR_IMAGE_HEIGHT))        # làm phẳng hình ảnh thành mảng numpy 1d

        npaROIResized = np.float32(npaROIResized)               # chuyển đổi numpy từ 1d kiểu ints thành numpy 1d kiểu float

        retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k = 15)              # Tìm kiếm láng giềng gần để nhận dạng kí tự

        strCurrentChar = str(chr(int(npaResults[0][0])))            # lấy kí tự từ kết quả

        strChars = strChars + strCurrentChar                        # nối kí tự hiện tại vào chuỗi đầy đủ

    # end for

    if Main.showSteps == True: # show steps 
        cv2.imshow("10", imgThreshColor)
    # end if 

    return strChars
# end function








