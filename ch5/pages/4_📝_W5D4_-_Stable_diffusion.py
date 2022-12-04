import os
if not os.path.exists("./images"):
    os.chdir("./ch5")

from st_dependencies import *
styling()

import plotly.io as pio
import plotly.express as px
import re
import json
import numpy as np

def img_to_html(img_path, width):
    with open("images/" + img_path, "rb") as file:
        img_bytes = file.read()
    encoded = base64.b64encode(img_bytes).decode()
    return f"<img style='width:{width}px;max-width:100%;st-bottom:25px' src='data:image/png;base64,{encoded}' class='img-fluid'>"
def st_image(name, width):
    st.markdown(img_to_html(name, width=width), unsafe_allow_html=True)

def read_from_html(filename):
    filename = f"images/{filename}.html"
    with open(filename) as f:
        html = f.read()
    call_arg_str = re.findall(r'Plotly\.newPlot\((.*)\)', html)[0]
    call_args = json.loads(f'[{call_arg_str}]')
    try:
        plotly_json = {'data': call_args[1], 'layout': call_args[2]}
        fig = pio.from_json(json.dumps(plotly_json))
    except:
        del call_args[2]["template"]["data"]["scatter"][0]["fillpattern"]
        plotly_json = {'data': call_args[1], 'layout': call_args[2]}
        fig = pio.from_json(json.dumps(plotly_json))
    return fig

def get_fig_dict():
    names = ["grid_output"]
    return {name: read_from_html(name) for name in names}

if "fig_dict" not in st.session_state:
    fig_dict = get_fig_dict()
    st.session_state["fig_dict"] = fig_dict
else:
    fig_dict = st.session_state["fig_dict"]

arr = np.load(r"images/arr.npy")
fig_dict["animation_output"] = px.imshow(arr, animation_frame=0, color_continuous_scale="gray")

st.markdown("""
<style>
.css-ffhzg2 span[style*="color: blue"] {
    color: rgb(0, 180, 240) !important;
}
.css-ffhzg2 span[style*="color: black"] {
    color: white !important;
}
.css-fg4pbf span[style*="color: orange"] {
    color: rgb(255, 130, 20) !important;
}
</style>""", unsafe_allow_html=True)

def section_home():

    st.markdown("""
Coming soon!
""")

func_list = [section_home]

page_list = ["🏠 Home"]
page_dict = {name: idx for idx, name in enumerate(page_list)}

def page():
    with st.sidebar:
        radio = st.radio("Section", page_list)
        st.markdown("---")
    func_list[page_dict[radio]]()

if is_local or check_password():
    page()


# %%
import numpy as np
import plotly.express as px



# %%