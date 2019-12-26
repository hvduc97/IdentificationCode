# Phân loại và gán nhãn dữ liệu huấn luyện

import sys
import numpy as np
import cv2
import os

# biến cấp độ mô-đun ##########################################################################
MIN_CONTOUR_AREA = 100

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

################################################################################################
def main():
    imgTrainingNumbers = cv2.imread("datatrain/train1.png")            # đọc hình ảnh đào tạo số

    if imgTrainingNumbers is None:                         
        print ("error: image not read from file \n\n")       
        os.system("pause")                                  
        return                                              
    # end if

    imgGray = cv2.cvtColor(imgTrainingNumbers, cv2.COLOR_BGR2GRAY)          # biến đổi sang hình ảnh thang độ xám
    imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)                        # sử dụng phương pháp Gaussian trong hàm thư viện opencv để làm mịn ảnh

                                                        # lọc hình ảnh từ thang độ xám sang đen trắng
    imgThresh = cv2.adaptiveThreshold(imgBlurred,                           # đưa vào ảnh thang độ xám đã đc làm mịn
                                      255,                                  # làm cho các pixel vượt quá ngưỡng thì thành toàn màu trắng
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,       # giá trị ngưỡng: là tổng trọng số của các giá trị lân cận trong đó trọng số là một cửa sổ gaussian
                                      cv2.THRESH_BINARY_INV,                # đảo ngược để các chữ số có màu trắng và background sẽ có màu đen
                                      11,                                   # kích thước của vùng lân cận pixel được sử dụng để tính giá trị ngưỡng
                                      2)                                    # một hằng số không đổi trừ trung bình hoặc trung bình có trọng số

    cv2.imshow("imgThresh", imgThresh)      # hiển thị hình ảnh binary

    imgThreshCopy = imgThresh.copy()        # tạo một bản sao của hình ảnh để khi tìm contour sẽ không làm thay đổi hình ảnh gốc

    npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,        # input ảnh đc sao chép
                                                 cv2.RETR_EXTERNAL,                 # chỉ lấy các đường viền ngoài cùng
                                                 cv2.CHAIN_APPROX_SIMPLE)           # nén các đoạn ngang, dọc và chéo và chỉ để lại các điểm cuối của chúng

                                # Khai báo mảng numpy trống, sẽ sử dụng để sau này ghi vào tệp
                                # hàng cho bằng 0, và cột thì cho bằng giá trị đủ để chứa tất cả dữ liệu hình ảnh
    npaFlattenedImages =  np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))

    intClassifications = []         # khai báo danh sách phân loại trống để lữu trữ các phân loại kí tự.

                                    # các kí tự đó là các chữ số từ 0-9 được lưu trong danh sách intValidChars
    intValidChars = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9')]

    for npaContour in npaContours:                          # cho mỗi contour
        if cv2.contourArea(npaContour) > MIN_CONTOUR_AREA:          # nếu contour đủ lớn để xem xét
            [intX, intY, intW, intH] = cv2.boundingRect(npaContour)         # nhận giá trị và break

                                                # vẽ hình chữ nhật xung quanh mỗi contour để yêu cầu nhập liệu
            cv2.rectangle(imgTrainingNumbers,           # vẽ hình chữ nhật trên hình ảnh đào tạo ban đầu
                          (intX, intY),                 # góc trên bên trái
                          (intX+intW,intY+intH),        # Góc dưới bên phải
                          (0, 0, 255),                  # màu đỏ
                          2)                            # độ dày

            imgROI = imgThresh[intY:intY+intH, intX:intX+intW]                                  # cắt char ra khỏi threshold image
            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))     # Thay đổi kích thước hình ảnh, điều này sẽ phù hợp hơn để nhận biết và lưu trữ

            cv2.imshow("imgROI", imgROI)                    # hiển thị kí tự cắt đc để tham khảo
            cv2.imshow("imgROIResized", imgROIResized)      # hiển thị hình ảnh đã được resize
            cv2.imshow("training_numbers.png", imgTrainingNumbers)      # hiển thị hình ảnh đào tạo số, bây giờ sẽ có hình chữ nhật màu đỏ được vẽ trên đó
            intChar = cv2.waitKey(0)                     # lấy key từ bàn phím

            if intChar == 27:                   # nếu nhấn phím esc
                sys.exit()                      # thoát chương trình
            elif intChar in intValidChars:      # ngược lại nếu char nằm trong danh sách các ký tự mà chúng ta đang tìm kiếm. . .

                intClassifications.append(intChar)  # nối thêm phân loại char vào danh sách số nguyên của ký tự (sẽ chuyển đổi về kiểu float trước khi ghi vào tệp)

                npaFlattenedImage = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))  # làm phẳng hình ảnh thành mảng numpy 1d để chúng ta có thể ghi vào tập tin sau này
                npaFlattenedImages = np.append(npaFlattenedImages, npaFlattenedImage, 0)                    # thêm mảng numpy hình ảnh phẳng hiện tại vào danh sách các mảng numpy hình ảnh phẳng
            # end if
        # end if
    # end for

    fltClassifications = np.array(intClassifications, np.float32)                   # chuyển đổi danh sách phân loại kiểu int sang float

    npaClassifications = fltClassifications.reshape((fltClassifications.size, 1))   # làm phẳng mảng numpy float 1d để chúng ta có thể ghi vào tệp sau đó

    print ("\n\ntraining complete !!\n")

    np.savetxt("classifications.txt", npaClassifications)           # ghi hình các phân loại kí tự vào tập tin
    np.savetxt("flattened_images.txt", npaFlattenedImages)          # ghi hình ảnh đã được làm phẳng vào tập tin

    cv2.destroyAllWindows()             # giải phóng bộ nhớ

    return

###################################################################################################
if __name__ == "__main__":
    main()
# end if




