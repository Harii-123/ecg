from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from io import BytesIO
from skimage.io import imread
from skimage import color
from skimage.filters import threshold_otsu, gaussian
from skimage.transform import resize
from skimage import measure
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

class ECG:
    def getImage(self, image_stream):
        """
        This function gets the user image from an image stream.
        Returns: User image
        """
        image = imread(image_stream)
        return image

    def GrayImage(self, image):
        """
        This function converts the user image to Gray Scale.
        Returns: Gray scale Image
        """
        image_gray = color.rgb2gray(image)
        image_gray = resize(image_gray, (1572, 2213))
        return image_gray

    def DividingLeads(self, image):
        """
        This function divides the ECG image into 13 Leads.
        Returns: List containing all 13 leads divided
        """
        Lead_1 = image[300:600, 150:643]  # Lead 1
        Lead_2 = image[300:600, 646:1135] # Lead aVR
        Lead_3 = image[300:600, 1140:1625] # Lead V1
        Lead_4 = image[300:600, 1630:2125] # Lead V4
        Lead_5 = image[600:900, 150:643]   # Lead 2
        Lead_6 = image[600:900, 646:1135]  # Lead aVL
        Lead_7 = image[600:900, 1140:1625] # Lead V2
        Lead_8 = image[600:900, 1630:2125] # Lead V5
        Lead_9 = image[900:1200, 150:643]  # Lead 3
        Lead_10 = image[900:1200, 646:1135] # Lead aVF
        Lead_11 = image[900:1200, 1140:1625] # Lead V3
        Lead_12 = image[900:1200, 1630:2125] # Lead V6
        Lead_13 = image[1250:1480, 150:2125] # Long Lead

        Leads = [Lead_1, Lead_2, Lead_3, Lead_4, Lead_5, Lead_6, Lead_7, Lead_8, Lead_9, Lead_10, Lead_11, Lead_12, Lead_13]
        return Leads

    def PreprocessingLeads(self, Leads):
        """
        This function performs preprocessing on the extracted leads.
        Returns: List of preprocessed leads
        """
        preprocessed_leads = []
        for y in Leads[:len(Leads) - 1]:
            grayscale = color.rgb2gray(y)
            blurred_image = gaussian(grayscale, sigma=1)
            global_thresh = threshold_otsu(blurred_image)
            binary_global = blurred_image < global_thresh
            binary_global = resize(binary_global, (300, 450))
            preprocessed_leads.append(binary_global)
        
        # Preprocess the last lead separately
        grayscale = color.rgb2gray(Leads[-1])
        blurred_image = gaussian(grayscale, sigma=1)
        global_thresh = threshold_otsu(blurred_image)
        binary_global = blurred_image < global_thresh
        preprocessed_leads.append(binary_global)
        
        return preprocessed_leads

    def SignalExtraction_Scaling(self, Leads):
        """
        This function performs signal extraction and scaling.
        Returns: List of scaled 1D signals
        """
        scaler = MinMaxScaler()
        all_scaled_signals = []

        for x, y in enumerate(Leads[:len(Leads) - 1]):
            grayscale = color.rgb2gray(y)
            blurred_image = gaussian(grayscale, sigma=0.7)
            global_thresh = threshold_otsu(blurred_image)
            binary_global = blurred_image < global_thresh
            binary_global = resize(binary_global, (300, 450))
            contours = measure.find_contours(binary_global, 0.8)
            contours_shape = sorted([x.shape for x in contours])[::-1][0:1]
            
            for contour in contours:
                if contour.shape in contours_shape:
                    test = resize(contour, (255, 2))
            
            fit_transform_data = scaler.fit_transform(test)
            Normalized_Scaled = pd.DataFrame(fit_transform_data[:, 0], columns=['X'])
            Normalized_Scaled = Normalized_Scaled.T
            all_scaled_signals.append(Normalized_Scaled)

        return all_scaled_signals

    def CombineConvert1Dsignal(self, all_scaled_signals):
        """
        This function combines all 1D signals of leads into one DataFrame.
        Returns: Final combined DataFrame
        """
        test_final = pd.concat(all_scaled_signals, axis=1, ignore_index=True)
        return test_final

    def DimensionalReduction(self, test_final):
        """
        This function performs dimensionality reduction using PCA.
        Returns: DataFrame with reduced dimensions
        """
        pca_loaded_model = joblib.load(r"PCA_ECG (1).pkl")
        result = pca_loaded_model.transform(test_final)
        final_df = pd.DataFrame(result)
        return final_df

    def ModelLoad_predict(self, final_df):
        """
        This function loads the pretrained model and performs ECG classification.
        Returns: Classification result
        """
        loaded_model = joblib.load(r"final.pkl")
        result = loaded_model.predict(final_df)
        if result[0] == 2:
            return "Your ECG is Normal"
        else:
            return "Your ECG corresponds to Myocardial Infarction"

@app.route('/', methods=['GET'])
def index():
    return render_template('input.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        image_stream = BytesIO(file.read())
        ecg = ECG()
        ecg_user_image = ecg.getImage(image_stream)
        ecg_user_gray_image = ecg.GrayImage(ecg_user_image)
        dividing_leads = ecg.DividingLeads(ecg_user_image)
        ecg_preprocessed_leads = ecg.PreprocessingLeads(dividing_leads)
        ec_signal_extraction = ecg.SignalExtraction_Scaling(dividing_leads)
        ecg_1dsignal = ecg.CombineConvert1Dsignal(ec_signal_extraction)
        ecg_final = ecg.DimensionalReduction(ecg_1dsignal)
        ecgmodel = ecg.ModelLoad_predict(ecg_final)
        return render_template('output.html', result=ecgmodel)
if __name__ == '__main__':
    app.run(debug=True)
