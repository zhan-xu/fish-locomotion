import glob
import cv2
import numpy as np

'''
Find the contours of a canny image, returns an image with the 3 largest contours, 
a bounding box around them, th bounding box, the contours and perimeters sorted
'''


def largestContours(canny, img, img_gray):
    _, contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    img_contour = np.copy(img)  # Contours change original image.
    # cv2.drawContours(img_contour, contours, -1, (0,255,0), 3) # Draw all - For visualization only

    # Contours -  maybe the largest perimeters pinpoint to the leaf?
    perimeter = []
    max_perim = [0, 0]
    i = 0

    # Find perimeter for each contour i = id of contour
    for each_cnt in contours:
        prm = cv2.arcLength(each_cnt, False)
        perimeter.append([prm, i])
        i += 1

    # Sort them
    perimeter = quick_sort(perimeter)

    unified = []
    max_index = []
    # Draw max contours
    for i in range(0, 3):
        index = perimeter[i][1]
        max_index.append(index)
        cv2.drawContours(img_contour, contours, index, (255, 0, 0), 3)

    # Get convex hull for max contours and draw them
    cont = np.vstack(contours[i] for i in max_index)
    hull = cv2.convexHull(cont)
    unified.append(hull)
    cv2.drawContours(img_contour, unified, -1, (0, 0, 255), 3)

    return img_contour, contours, perimeter, hull


'''
Given a convex hull apply graph cut to the image
Assumptions: 
- Everything inside convex hull is the foreground object - cv2.GC_FGD or 1
- Everything outside the rectangle is the background -  cv2.GC_BGD or 0
- Between the hull and the rectangle is probably foreground - cv2.GC_PR_FGD or 3
'''


def grCut(chull, img):
    # First create our rectangle that contains the object
    y_corners = np.amax(chull, axis=0)
    x_corners = np.amin(chull, axis=0)
    x_min = x_corners[0][0]
    x_max = x_corners[0][1]
    y_min = y_corners[0][0]
    y_max = y_corners[0][1]
    rect = (x_min, x_max, y_min, y_max)

    # Our mask
    mask = np.zeros(img.shape[:2], np.uint8)

    # Values needed for algorithm
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Grabcut
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == cv2.GC_PR_BGD) | (mask == cv2.GC_BGD), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]

    return img


'''
Sort all perimeters obtained
'''


def quick_sort(p):
    if len(p) <= 1:
        return p

    pivot = p.pop(0)
    low, high = [], []
    for entry in p:
        if entry[0] > pivot[0]:
            high.append(entry)
        else:
            low.append(entry)
    return quick_sort(high) + [pivot] + quick_sort(low)


def filtering(img_gray, esp):
    if esp == "median":
        return cv2.medianBlur(img_gray, 5)
    elif esp == "gaussian":
        return cv2.GaussianBlur(img_gray, (5, 5), 0)
    elif esp == "bilateral":
        return cv2.bilateralFilter(img_gray, 5, 50, 100)


if __name__ == '__main__':
    for img_id in range(82,83):
        img_path = '../example/real/{:03d}.jpg'.format(img_id)
        print img_path
        img = cv2.imread(img_path)
        size_ori = img.shape[:2]
        img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # thresh = threshold(img_gray) 	# Threshold the image
        #filtered = filtering(img_gray, "bilateral")
        canny = cv2.Canny(img_gray, 20, 100)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        canny = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel, iterations=3)

        # cv2.namedWindow('canny')
        # cv2.namedWindow('morphological')
        # cv2.imshow('canny',canny)
        # cv2.imshow('morphological', morph)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        _, contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        max_area = 0
        max_id = 0
        for i in range(len(contours)):
            cnt = contours[i]
            area = cv2.contourArea(cnt)
            # find max countour
            if (area > max_area):
                max_area = area
                max_id = i

        cv2.drawContours(img, contours, max_id, (0, 0, 254), thickness=-1)
        lowerb = np.array([0, 0, 250], np.uint8)
        upperb = np.array([0, 0, 255], np.uint8)
        frame = cv2.inRange(img, lowerb,upperb)

        h,w = frame.shape
        for c in range(w):
            min_p = h
            max_p = 0
            for r in range(h):
                if frame[r,c] > 250:
                    if r<min_p:
                        min_p = r
                    if r>max_p:
                        max_p = r
            for r_ in range(min_p,max_p):
                frame[r_,c] = 255

        # cv2.namedWindow('frame')
        # cv2.imshow('frame',frame)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        frame = cv2.resize(frame, size_ori)
        name_out = "mask_{:03d}.jpg".format(img_id)
        cv2.imwrite(name_out, frame)
        # cv2.imshow('mask', frame)
        # cv2.waitKey(0)
