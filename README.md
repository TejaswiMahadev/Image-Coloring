# Image Colorization App

The **Image Colorization App** is a Streamlit web application that leverages deep learning models to colorize grayscale images. Two pre-trained models—ECCV16 and SIGGRAPH17—are used to generate vibrant colorizations from user-uploaded images. The app offers GPU acceleration (if available) to speed up processing.

## Features

- **Easy-to-Use Interface:**  
  Upload an image in JPG, JPEG, or PNG format and view the colorization results side by side.
- **Dual Model Comparison:**  
  Compare the colorization outputs from two different models:

  - **ECCV16**
  - **SIGGRAPH17**

- **GPU Support:**  
  Option to use GPU for faster inference when available.

- **Interactive Results:**  
  View the original image, the grayscale input, and the two colorized outputs in a clear and organized layout.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Jnan-py/image-colorization-app.git
   cd image-colorization-app
   ```

2. **Create a Virtual Environment (Recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Application:**

   ```bash
   streamlit run main.py
   ```

2. **Upload an Image:**

   - Click the file uploader to select an image (jpg, jpeg, or png).
   - Optionally, check the **"Use GPU"** checkbox if you have GPU support enabled.

3. **View Results:**
   - The app displays:
     - The grayscale version of your image.
     - The colorized output using the ECCV16 model.
     - The original image.
     - The colorized output using the SIGGRAPH17 model.

## Project Structure

```
image-colorization-app/
│
├── main.py                  # Main Streamlit application
├── models/                 # Folder containing model definitions and preprocessing utilities:
│   ├── eccv16.py           # ECCV16 model definition
│   ├── siggraph17.py       # SIGGRAPH17 model definition│
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies
```

## Technologies Used

- **Streamlit:** For building the interactive web interface.
- **PyTorch:** For loading and running the deep learning models.
- **NumPy:** For numerical operations.
- **Matplotlib:** For plotting (if needed).
- **Pillow:** For image processing.
- **Custom Models:** Pre-trained ECCV16 and SIGGRAPH17 models along with associated preprocessing utilities.

---

Save these files in your project directory. To run the app, use the following command:

```bash
streamlit run main.py
```
