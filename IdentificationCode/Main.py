# file main

import cv2
import numpy as np
import os

import DetectChars
import DetectPlates
import PossiblePlate

# biến cấp độ mô-đun ##########################################################################
SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

showSteps = False

###################################################################################################
def main():

    blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()         # Huấn luyện dữ liệu KNN

    if blnKNNTrainingSuccessful == False:                               # nếu không thành công
        print("\nerror: KNN traning was not successful\n")  # in thing báo lỗi
        return                                                          # và thoát chưng trình
    # end if

    imgPhoneCard  = cv2.imread("test/0.jpg")               # mở ảnh mã thẻ cào

    if imgPhoneCard is None:                            # nếu ảnh thẻ cào không mở được
        print("\nerror: image not read from file \n\n")  # in thông báo lỗi
        os.system("pause")                                  # dừng màn hình để đọc lỗi
        return                                              # thoát chương trình

    listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgPhoneCard)           # phát hiện vùng chứa mã thẻ cào

    listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)        # phát hiện kí tự trong vùng chứa mã

    cv2.imshow("imgPhoneCard", imgPhoneCard)            # show ảnh thẻ cào

    if len(listOfPossiblePlates) == 0:                          # nếu không có vùng chữa mã nào đc tìm thấy
        print("\nno license plates were detected\n")  # in thông báo
    else:                                                       # ngược lại
                # nếu trong danh sách có ít nhất 1 tấm

                # sắp xếp các tấm có trong danh sách theo số lượng kí tự có trong tấm.
        listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)

                # giả sử tấm có ký tự được nhận biết nhiều nhất (tấm đầu tiên trong danh sách) là tấm thực tế
        licPlate = listOfPossiblePlates[0]

        cv2.imshow("imgPlate", licPlate.imgPlate)           # hiển thị ảnh cắt của tấm với dạng binary
        cv2.imshow("imgThresh", licPlate.imgThresh)

        if len(licPlate.strChars) == 0:                     # nếu không có ký tự được tìm thấy trong tấm
            print("\nno characters were detected\n\n")  # in thông báo
            return                                          # và thoát chương trình
        # end if

        drawRedRectangleAroundPlate(imgOriginalScene, licPlate)             # vẽ hình chữ nhật màu đỏ xung quanh tấm

        print("\nlicense plate read from image = " + licPlate.strChars + "\n")  # in mã thẻ cào
        print("----------------------------------------")

        writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)           # viết mã thẻ cào lên ảnh

        cv2.imshow("imgPhoneCard", imgOriginalScene)                # hiển thị lại ảnh

        cv2.imwrite("imgCodeCard.png", imgOriginalScene)           # xuất ra ảnh

    # end if else

    cv2.waitKey(0)					# giữ cửa sổ mở cho đến khi nhấn phím bất kì

    return
# end main

###################################################################################################
def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):

    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)            # lấy 4 điểm để vẽ vùng chứa mã thẻ cào

    cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)         # vẽ 4 dòng màu đỏ
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)
# end function

###################################################################################################
def writeLicensePlateCharsOnImage(imgOriginalScene, licPlate):
    ptCenterOfTextAreaX = 0                            # Tọa độ vị trí viết mã lên thẻ
    ptCenterOfTextAreaY = 0

    ptLowerLeftTextOriginX = 0                          
    ptLowerLeftTextOriginY = 0

    sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
    plateHeight, plateWidth, plateNumChannels = licPlate.imgPlate.shape

    intFontFace = cv2.FONT_HERSHEY_SIMPLEX                      # chọn font
    fltFontScale = float(plateHeight) / 30.0                    # tỉ lệ font
    intFontThickness = int(round(fltFontScale * 1.5))           # độ dày
    textSize, baseline = cv2.getTextSize(licPlate.strChars, intFontFace, fltFontScale, intFontThickness)        # gọi ham getTextSize

            
    ( (intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg ) = licPlate.rrLocationOfPlateInScene

    intPlateCenterX = int(intPlateCenterX)              # ép kiểu int
    intPlateCenterY = int(intPlateCenterY)

    ptCenterOfTextAreaX = int(intPlateCenterX)         

    if intPlateCenterY < (sceneHeight * 0.75):                                                  # ảnh lệch trên
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) + int(round(plateHeight * 1.6))      # viết ở dưới
    else:                                                                                       # ngược lại
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) - int(round(plateHeight * 1.6))      
    # end if

    textSizeWidth, textSizeHeight = textSize               

    ptLowerLeftTextOriginX = int(ptCenterOfTextAreaX - (textSizeWidth / 2))           
    ptLowerLeftTextOriginY = int(ptCenterOfTextAreaY + (textSizeHeight / 2))          
            # viết văn bản lên hình ảnh
    cv2.putText(imgOriginalScene, licPlate.strChars, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), intFontFace, fltFontScale, SCALAR_RED, intFontFace)
# end function

###################################################################################################
if __name__ == "__main__":
    main()


















