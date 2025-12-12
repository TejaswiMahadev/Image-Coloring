import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from models import eccv16, siggraph17, preprocess_img, postprocess_tens

st.set_page_config(page_icon=":img", page_title= "Image Colorizer", layout= "wide")

st.title("Image Colorization App")
st.write("Upload an image (jpg, jpeg, png) to colorize it:")

use_gpu = st.checkbox("Use GPU", value=False)

@st.cache_resource(show_spinner=False)
def load_models(use_gpu_flag):
    model_eccv16 = eccv16(pretrained=True).eval()
    model_siggraph17 = siggraph17(pretrained=True).eval()
    if use_gpu_flag:
        model_eccv16.cuda()
        model_siggraph17.cuda()
    return model_eccv16, model_siggraph17

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:    
    image = Image.open(uploaded_file).convert('RGB')    

    img_np = np.array(image)
    
    with st.spinner("Processing Image..."):

        tens_l_orig, tens_l_rs = preprocess_img(img_np, HW=(256,256))
        if use_gpu:
            tens_l_rs = tens_l_rs.cuda()
            
        colorizer_eccv16, colorizer_siggraph17 = load_models(use_gpu)
        
        with torch.no_grad():
            img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig, 0*tens_l_orig), dim=1))        
            out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
            out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())
        
        st.write("### Colorization Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(img_bw, caption="Grayscale Input", use_container_width=True)
            st.image(out_img_eccv16, caption="Output (ECCV 16)", use_container_width=True)
        with col2:
            st.image(image, caption="Original", use_container_width=True)
            st.image(out_img_siggraph17, caption="Output (SIGGRAPH 17)", use_container_width=True)
