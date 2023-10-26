import os, math, csv
import streamlit as st
from streamlit_image_select import image_select
import cv2
import numpy as np
from PIL import Image
import matplotlib.colors as mcolors

class DirectoryManager:
    def __init__(self, output_dir):
        self.dir_output = output_dir
        self.mask_flag = os.path.join(output_dir, "mask_flag")
        self.mask_plant = os.path.join(output_dir, "mask_plant")
        self.mask_plant_plot = os.path.join(output_dir, "mask_plant_plot")
        self.plant_rgb = os.path.join(output_dir, "plant_rgb")
        self.plot_rgb = os.path.join(output_dir, "plot_rgb")
        self.plant_rgb_warp = os.path.join(output_dir, "plant_rgb_warp")
        self.plant_mask_warp = os.path.join(output_dir, "plant_mask_warp")
        self.data = os.path.join(output_dir, "data")

    def create_directories(self):
        os.makedirs(self.dir_output, exist_ok=True)
        os.makedirs(self.mask_flag, exist_ok=True)
        os.makedirs(self.mask_plant, exist_ok=True)
        os.makedirs(self.mask_plant_plot, exist_ok=True)
        os.makedirs(self.plant_rgb, exist_ok=True)
        os.makedirs(self.plot_rgb, exist_ok=True)
        os.makedirs(self.plant_rgb_warp, exist_ok=True)
        os.makedirs(self.plant_mask_warp, exist_ok=True)
        os.makedirs(self.data, exist_ok=True)



def hex_to_hsv_bounds(hex_color, sat_value, val_value):
    # Convert RGB hex to color
    rgb_color = mcolors.hex2color(hex_color)
    hsv_color = mcolors.rgb_to_hsv(np.array(rgb_color).reshape(1, 1, 3))
    
    # Adjust the saturation and value components based on user's input
    hsv_color[0][0][1] = sat_value / 255.0  # Saturation
    hsv_color[0][0][2] = val_value / 255.0  # Value

    hsv_bound = tuple((hsv_color * np.array([179, 255, 255])).astype(int)[0][0])
    
    return hsv_bound

def warp_image(img, vertices):
    # Compute distances between the vertices to determine the size of the target square
    distances = [np.linalg.norm(np.array(vertices[i]) - np.array(vertices[i+1])) for i in range(len(vertices)-1)]
    distances.append(np.linalg.norm(np.array(vertices[-1]) - np.array(vertices[0])))  # Add the distance between the last and first point
    max_distance = max(distances)

    # Define target vertices for the square
    dst_vertices = np.array([
        [max_distance - 1, 0],
        [0, 0],
        [0, max_distance - 1],
        [max_distance - 1, max_distance - 1]
    ], dtype="float32")

    # Compute the perspective transform matrix using the provided vertices
    matrix = cv2.getPerspectiveTransform(np.array(vertices, dtype="float32"), dst_vertices)
    
    # Warp the image to the square
    warped_img = cv2.warpPerspective(img, matrix, (int(max_distance), int(max_distance)))

    return warped_img

def process_image(image_path, flag_lower, flag_upper, plant_lower, plant_upper):
    img = cv2.imread(image_path)
    
    # Check if image is valid
    if img is None:
        print(f"Error reading image from path: {image_path}")
        return None, None, None, None, None, None, None, None, None, None

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Convert image to HSV
    
    # Explicitly ensure bounds are integer tuples
    flag_lower = tuple(int(x) for x in flag_lower)
    flag_upper = tuple(int(x) for x in flag_upper)
    plant_lower = tuple(int(x) for x in plant_lower)
    plant_upper = tuple(int(x) for x in plant_upper)

    flag_mask = cv2.inRange(hsv_img, flag_lower, flag_upper)
    plant_mask = cv2.inRange(hsv_img, plant_lower, plant_upper)

    # Find contours
    contours, _ = cv2.findContours(flag_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area and keep only the largest 4
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:4]
    
    # If there are not 4 largest contours, return
    if len(sorted_contours) != 4:
        return None, None, None, None, None, None, None, None, None, None

    # Create a new mask with only the largest 4 contours
    largest_4_flag_mask = np.zeros_like(flag_mask)
    cv2.drawContours(largest_4_flag_mask, sorted_contours, -1, (255), thickness=cv2.FILLED)
    
    # Compute the centroid for each contour
    centroids = []
    for contour in sorted_contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0
        centroids.append((cx, cy))
    
    # Compute the centroid of the centroids
    centroid_x = sum(x for x, y in centroids) / 4
    centroid_y = sum(y for x, y in centroids) / 4

    # Sort the centroids
    centroids.sort(key=lambda point: (-math.atan2(point[1] - centroid_y, point[0] - centroid_x)) % (2 * np.pi))

    # Create a polygon mask using the sorted centroids
    poly_mask = np.zeros_like(flag_mask)
    cv2.fillPoly(poly_mask, [np.array(centroids)], 255)
    
    # Mask the plant_mask with poly_mask
    mask_plant_plot = cv2.bitwise_and(plant_mask, plant_mask, mask=poly_mask)

    # Count the number of black pixels inside the quadrilateral
    total_pixels_in_quad = np.prod(poly_mask.shape)
    white_pixels_in_quad = np.sum(poly_mask == 255)
    black_pixels_in_quad = total_pixels_in_quad - white_pixels_in_quad
    
    # Extract the RGB pixels from the original image using the mask_plant_plot
    plant_rgb = cv2.bitwise_and(img, img, mask=mask_plant_plot)

    # Draw the bounding quadrilateral
    plot_rgb = plant_rgb.copy()
    for i in range(4):
        cv2.line(plot_rgb, centroids[i], centroids[(i+1)%4], (0, 0, 255), 3)

    # Convert the masks to RGB for visualization
    flag_mask_rgb = cv2.cvtColor(flag_mask, cv2.COLOR_GRAY2RGB)
    orange_color = [255, 165, 0]  # RGB value for orange
    flag_mask_rgb[np.any(flag_mask_rgb != [0, 0, 0], axis=-1)] = orange_color

    plant_mask_rgb = cv2.cvtColor(plant_mask, cv2.COLOR_GRAY2RGB)
    mask_plant_plot_rgb = cv2.cvtColor(mask_plant_plot, cv2.COLOR_GRAY2RGB)
    bright_green_color = [0, 255, 0]
    plant_mask_rgb[np.any(plant_mask_rgb != [0, 0, 0], axis=-1)] = bright_green_color
    mask_plant_plot_rgb[np.any(mask_plant_plot_rgb != [0, 0, 0], axis=-1)] = bright_green_color
    
    # Warp the images
    plant_rgb_warp = warp_image(plant_rgb, centroids)
    plant_mask_warp = warp_image(mask_plant_plot_rgb, centroids)

    return flag_mask_rgb, plant_mask_rgb, mask_plant_plot_rgb, plant_rgb, plot_rgb, plant_rgb_warp, plant_mask_warp, plant_mask, mask_plant_plot, black_pixels_in_quad

def calculate_coverage(mask_plant_plot, plant_mask_warp, black_pixels_in_quad):
    # Calculate the percentage of white pixels for mask_plant_plot
    white_pixels_plot = np.sum(mask_plant_plot > 0)
    total_pixels_plot = mask_plant_plot.size
    plot_coverage = (white_pixels_plot / black_pixels_in_quad) * 100

    # Convert plant_mask_warp to grayscale
    plant_mask_warp_gray = cv2.cvtColor(plant_mask_warp, cv2.COLOR_BGR2GRAY)

    # Calculate the percentage of white pixels for plant_mask_warp
    white_pixels_warp = np.sum(plant_mask_warp_gray > 0)
    total_pixels_warp = plant_mask_warp_gray.size
    warp_coverage = (white_pixels_warp / total_pixels_warp) * 100

    # Calculate the area in cm^2 of the mask_plant_plot
    # Given that the real-life size of the square is 2 square meters or 20000 cm^2
    plot_area_cm2 = (white_pixels_warp / total_pixels_warp) * 20000

    return round(plot_coverage,2), round(warp_coverage,2), round(plot_area_cm2,2)

def get_color_parameters():
    # Color pickers for hue component
    FL, FL_S, FL_SS = st.columns([2,4,4])
    with FL:
        flag_lower_hex = st.color_picker("Flag Color Lower Bound Hue", "#33211f")
    with FL_S:
        flag_lower_sat = st.slider("Flag Lower Bound Saturation", 0, 255, 120)
    with FL_SS:
        flag_lower_val = st.slider("Flag Lower Bound Value", 0, 255, 150)

    FU, FU_S, FU_SS = st.columns([2,4,4])
    with FU:
        flag_upper_hex = st.color_picker("Flag Color Upper Bound Hue", "#ff7700")
    with FU_S:
        flag_upper_sat = st.slider("Flag Upper Bound Saturation", 0, 255, 255)
    with FU_SS:
        flag_upper_val = st.slider("Flag Upper Bound Value", 0, 255, 255)

    PL, PL_S, PL_SS = st.columns([2,4,4])
    with PL:
        plant_lower_hex = st.color_picker("Plant Color Lower Bound Hue", "#504F49")
    with PL_S:
        plant_lower_sat = st.slider("Plant Lower Bound Saturation", 0, 255, 30)
    with PL_SS:
        plant_lower_val = st.slider("Plant Lower Bound Value", 0, 255, 30)

    PU, PU_S, PU_SS = st.columns([2,4,4])
    with PU:
        plant_upper_hex = st.color_picker("Plant Color Upper Bound Hue", "#00CFFF")
    with PU_S:
        plant_upper_sat = st.slider("Plant Upper Bound Saturation", 0, 255, 255)
    with PU_SS:
        plant_upper_val = st.slider("Plant Upper Bound Value", 0, 255, 255)  

    # Get HSV bounds using the modified function
    flag_lower_bound = hex_to_hsv_bounds(flag_lower_hex, flag_lower_sat, flag_lower_val)
    flag_upper_bound = hex_to_hsv_bounds(flag_upper_hex, flag_upper_sat, flag_upper_val)
    plant_lower_bound = hex_to_hsv_bounds(plant_lower_hex, plant_lower_sat, plant_lower_val)
    plant_upper_bound = hex_to_hsv_bounds(plant_upper_hex, plant_upper_sat, plant_upper_val)

    return flag_lower_bound, flag_upper_bound, plant_lower_bound, plant_upper_bound

def save_img(directory, base_name, mask):
    mask_name = os.path.join(directory, os.path.basename(base_name))
    cv2.imwrite(mask_name, mask)

def main():
    st.set_page_config(layout="wide", page_title='GreenSight')
    st.title("GreenSight")

    _, R_coverage, R_plot_area_cm2, R_save = st.columns([5,2,2,2])
    img_gallery, img_main, img_seg, img_green, img_warp = st.columns([1,4,2,2,2])

    dir_input = st.text_input("Input directory for images:", value="D:\Dropbox\GreenSight\demo")
    dir_output = st.text_input("Output directory:", value="D:\Dropbox\GreenSight\demo_out")
    
    directory_manager = DirectoryManager(dir_output)
    directory_manager.create_directories()

    run_name = st.text_input("Run name:", value="test")
    file_name = os.path.join(directory_manager.data, f"{run_name}.csv")
    headers = ['image',"plant_coverage_uncorrected_percen", "plant_coverage_corrected_percent", "plant_area_corrected_cm2"]
    file_exists = os.path.isfile(file_name)

    if 'input_list' not in st.session_state:
        input_images = [os.path.join(dir_input, fname) for fname in os.listdir(dir_input) if fname.endswith(('.jpg', '.jpeg', '.png'))]
        st.session_state.input_list = input_images

    if os.path.exists(dir_input):
        
        if len(st.session_state.input_list) == 0 or st.session_state.input_list is None:
            st.balloons()
        else:
            with img_gallery:
                selected_img = image_select("Select an image", st.session_state.input_list, use_container_width=False)
                base_name = os.path.basename(selected_img)
            
            if selected_img:

                selected_img_view = Image.open(selected_img)
                with img_main:
                    st.image(selected_img_view, caption="Selected Image", use_column_width='auto')

                    flag_lower_bound, flag_upper_bound, plant_lower_bound, plant_upper_bound = get_color_parameters()

                flag_mask, plant_mask, mask_plant_plot, plant_rgb, plot_rgb, plant_rgb_warp, plant_mask_warp, plant_mask_bi, mask_plant_plot_bi, black_pixels_in_quad = process_image(selected_img, flag_lower_bound, flag_upper_bound, plant_lower_bound, plant_upper_bound)

                if plant_mask_warp is not None:
                    plot_coverage, warp_coverage, plot_area_cm2 = calculate_coverage(mask_plant_plot_bi, plant_mask_warp, black_pixels_in_quad)

                    with R_coverage:
                        st.markdown(f"Uncorrected Plant Coverage: {plot_coverage}%")
                    with R_plot_area_cm2:
                        st.markdown(f"Corrected Plant Coverage: {warp_coverage}%")
                        st.markdown(f"Corrected Plant Area: {plot_area_cm2}cm2")

                    # Display masks in galleries
                    with img_seg:
                        st.image(plant_mask, caption="Plant Mask", use_column_width=True)
                        st.image(flag_mask, caption="Flag Mask", use_column_width=True)
                    with img_green:
                        st.image(mask_plant_plot, caption="Plant Mask Inside Plot", use_column_width=True)
                        st.image(plant_rgb, caption="Plant Material", use_column_width=True)
                    with img_warp:
                        st.image(plot_rgb, caption="Plant Material Inside Plot", use_column_width=True)
                        st.image(plant_rgb_warp, caption="Plant Mask Inside Plot Warped to Square", use_column_width=True)
                        # st.image(plot_rgb_warp, caption="Flag Mask", use_column_width=True)
                    with R_save:
                        if st.button('Save'):
                            # Save the masks to their respective folders
                            save_img(directory_manager.mask_flag, base_name, flag_mask)
                            save_img(directory_manager.mask_plant, base_name, plant_mask)
                            save_img(directory_manager.mask_plant_plot, base_name, mask_plant_plot)
                            save_img(directory_manager.plant_rgb, base_name, plant_rgb)
                            save_img(directory_manager.plot_rgb, base_name, plot_rgb)
                            save_img(directory_manager.plant_rgb_warp, base_name, plant_rgb_warp)
                            save_img(directory_manager.plant_mask_warp, base_name, plant_mask_warp)

                            # Append the data to the CSV file
                            with open(file_name, mode='a', newline='') as file:
                                writer = csv.writer(file)
                                
                                # If the file doesn't exist, write the headers
                                if not file_exists:
                                    writer.writerow(headers)
                                
                                # Write the data
                                writer.writerow([f"{base_name}",f"{plot_coverage}", f"{warp_coverage}", f"{plot_area_cm2}"])

                            # Remove processed image from the list
                            st.session_state.input_list.remove(selected_img)
                            st.rerun()
                else:
                    with R_save:
                        if st.button('Save as Failure'):
                            # Append the data to the CSV file
                            with open(file_name, mode='a', newline='') as file:
                                writer = csv.writer(file)
                                
                                # If the file doesn't exist, write the headers
                                if not file_exists:
                                    writer.writerow(headers)
                                
                                # Write the data
                                writer.writerow([f"{base_name}",f"NA", f"NA", f"NA"])

                            # Remove processed image from the list
                            st.session_state.input_list.remove(selected_img)
                            st.rerun()


if __name__ == '__main__':
    main()
