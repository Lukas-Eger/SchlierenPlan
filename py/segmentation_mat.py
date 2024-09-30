import numpy as np
import cv2
import matplotlib.pyplot as plt

def detect_striae(image, algorithm, *args):
    print('---Stria detection started---')

    curr_img = image.copy()

    if algorithm == "Canny":
        sigma = 1.5
        print(f'Running Canny edge detection with sigma of {sigma}')
        
        # Apply Gaussian Blur (approximate sigma in Canny)
        blurred_img = cv2.GaussianBlur(image, (5, 5), sigma)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred_img, 100, 200)
        im_height, im_width = edges.shape

        # Clean image edges
        print('Cleaning image edges')
        edges[:3, :] = 0
        edges[-3:, :] = 0
        edges[:, :3] = 0
        edges[:, -3:] = 0
        
        plt.figure('Canny-Filter')
        plt.imshow(edges, cmap='gray')
        plt.show()

        # Label connected components using OpenCV
        print('Labeling connected parts')
        num_labels, label_matrix = cv2.connectedComponents(edges)
        
        # Filter small areas
        min_area = 30
        print(f'Deleting areas < {min_area} pixels')
        area_values = np.bincount(label_matrix.flatten())
        large_labels = np.where(area_values > min_area)[0]

        # Rebuild the image based on filtered regions
        curr_img = np.isin(label_matrix, large_labels).astype(np.uint8)

        plt.figure()
        plt.imshow(curr_img, cmap='gray')
        plt.show()

    elif algorithm == "Threshold":
        print('Applying adaptive thresholding level')
        
        # Apply adaptive thresholding using OpenCV
        T = cv2.adaptiveThreshold(curr_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 63, -0.58)
        bw = T / 255  # Normalize to binary
        
        plt.figure()
        plt.imshow(bw, cmap='gray')
        plt.show()

        # Morphological operations
        print('Applying morphological operations to only keep relevant areas')
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cl = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, se)

        # Remove small areas
        num_labels, label_matrix = cv2.connectedComponents(cl.astype(np.uint8))
        
        plt.figure()
        plt.imshow(label_matrix, cmap='gray')
        plt.show()

        print('Skeletonizing image')
        skel = skeletonize_opencv(cl.astype(np.uint8))

        # Fit a curve through the skeleton
        print('Fitting curve through each skeleton')
        curr_img = skel  # Replace this with any curve fitting logic

    # Label connected parts again after processing
    num_labels, label_matrix = cv2.connectedComponents(curr_img.astype(np.uint8))

    print('Looping through all detected striae to get brightness profile for each')
    striae_profiles = []

    for c in range(1, num_labels):
        # Get brightness values for the current stria
        curr_stria_values = image[label_matrix == c]

        # Get coordinates of each pixel of the stria
        rows, cols = np.where(label_matrix == c)

        # Fit a first-order polynomial to the points of the stria
        coeffs = np.polyfit(cols, rows, 1)
        
        # Perpendicular slope
        perpendicular_slope = -1 / coeffs[0]
        direction_vector_perpendicular = np.array([1, perpendicular_slope]) / np.linalg.norm([1, perpendicular_slope])

        # Find maximum brightness point of the stria
        max_ind = np.argmax(curr_stria_values)

        # Create a perpendicular line on the lightest point of the stria
        n_pixel = np.arange(-30, 31)
        points_on_line = np.array([
            [cols[max_ind], rows[max_ind]] + n * direction_vector_perpendicular 
            for n in n_pixel
        ])

        # Read brightness values along this line
        x_points = points_on_line[:, 0]
        y_points = points_on_line[:, 1]
        line_values = improfile(image, x_points, y_points)

        # Remove NaN values from the profile
        line_values = line_values[~np.isnan(line_values)]

        # Store the profile
        striae_profiles.append(line_values)

    print('Stria profiles determined')
    print('---Stria detection done---')
    print('--------------------------')
    
    return striae_profiles, label_matrix

def skeletonize_opencv(image):
    """
    Skeletonization using OpenCV.
    """
    size = np.size(image)
    skel = np.zeros(image.shape, np.uint8)
    
    # Structuring element for morphological operations
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        # Open the image
        open_img = cv2.morphologyEx(image, cv2.MORPH_OPEN, element)
        # Subtract the open image from the original image
        temp = cv2.subtract(image, open_img)
        # Erode the image
        eroded = cv2.erode(image, element)
        # Or the result with the skeleton
        skel = cv2.bitwise_or(skel, temp)
        image = eroded.copy()

        # Stop if there is nothing left to erode
        if cv2.countNonZero(image) == 0:
            break
    
    return skel

def improfile(image, x_points, y_points):
    """
    Sample brightness values along a line between two points.
    """
    num_points = len(x_points)
    x_points = np.clip(np.round(x_points).astype(int), 0, image.shape[1] - 1)
    y_points = np.clip(np.round(y_points).astype(int), 0, image.shape[0] - 1)
    
    return image[y_points, x_points]


img = cv2.imread("h_std_05_raw.jpg",cv2.IMREAD_GRAYSCALE)

detect_striae(img, "Threshold", "args")