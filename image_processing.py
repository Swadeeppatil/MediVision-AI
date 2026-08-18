import cv2
import numpy as np
import os

def enhance_image(image_path):
    """Applies CLAHE contrast enhancement and Gaussian smoothing."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return image_path

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    enhanced_final = cv2.addWeighted(img, 1.4, blurred, -0.4, 0)

    base, ext = os.path.splitext(image_path)
    output_path = f"{base}_enhanced{ext}"
    cv2.imwrite(output_path, enhanced_final)
    return output_path

def generate_thermal_image(image_path):
    """Generates a professional thermal heat-map visualization using OpenCV COLORMAP_JET/INFERNO."""
    img = cv2.imread(image_path)
    if img is None:
        return image_path

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE to boost bone and structural contrast
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Calculate gradient magnitude for structural intensity heat density
    sobelx = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Normalize magnitude
    magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Combine original intensity with high-density gradient variations
    combined = cv2.addWeighted(enhanced, 0.6, magnitude_norm, 0.4, 0)
    
    # Apply JET pseudo-color mapping for authentic thermal heat-mapping
    thermal_map = cv2.applyColorMap(combined, cv2.COLORMAP_JET)
    
    # Smooth thermal transition effect
    thermal_smooth = cv2.GaussianBlur(thermal_map, (7, 7), 0)
    
    base, ext = os.path.splitext(image_path)
    output_path = f"{base}_thermal{ext}"
    cv2.imwrite(output_path, thermal_smooth)
    return output_path

def highlight_fracture_area(image_path):
    """Highlights potential fracture regions with bounding contour and center markers."""
    img = cv2.imread(image_path)
    if img is None:
        return image_path

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 31, 2)
    edges = cv2.Canny(thresh, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest_contour)
        
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [hull], -1, (255), -1)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(sobel_x**2 + sobel_y**2)
        
        smoothed = cv2.GaussianBlur(gradient, (5, 5), 0)
        _, thresholded = cv2.threshold(smoothed, np.mean(smoothed) + np.std(smoothed), 255, cv2.THRESH_BINARY)
        thresholded = cv2.convertScaleAbs(thresholded)
        
        contour_candidates, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_fracture_area = cv2.contourArea(hull) * 0.04
        fracture_candidates = [cnt for cnt in contour_candidates if cv2.contourArea(cnt) >= min_fracture_area]
        
        if fracture_candidates:
            fracture_candidates.sort(key=cv2.contourArea, reverse=True)
            largest_fracture = fracture_candidates[0]
            
            M = cv2.moments(largest_fracture)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Distinctive red target boundary & marker
                cv2.drawContours(img, [largest_fracture], -1, (0, 0, 255), 3)
                cv2.circle(img, (cx, cy), 22, (0, 0, 255), 3)
                cv2.circle(img, (cx, cy), 4, (0, 255, 255), -1)
                
                # Crosshairs
                cv2.line(img, (cx - 30, cy), (cx + 30, cy), (0, 0, 255), 2)
                cv2.line(img, (cx, cy - 30), (cx, cy + 30), (0, 0, 255), 2)
        else:
            cv2.drawContours(img, [hull], -1, (255, 0, 0), 2)

    base, ext = os.path.splitext(image_path)
    output_path = f"{base}_highlighted{ext}"
    cv2.imwrite(output_path, img)
    return output_path
