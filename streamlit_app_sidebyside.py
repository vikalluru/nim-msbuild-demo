import streamlit as st
import time
import json
import requests

# Define the API endpoints and keys

# VLLM
api_url_1 = 'https://nim-aml-endpoint-1.westeurope.inference.ml.azure.com'
model_1 = "/var/azureml-app/azureml-models/mistralai-Mixtral-8x7B-Instruct-v01/5/mlflow_model_folder/data/model"
deployment_1 = "vllm-aml-mixtral-deployment"

# NIM
api_url_2 = 'https://nim-aml-endpoint-1.westeurope.inference.ml.azure.com'
model_2 = "mixtral-instruct"
deployment_2 = "nim-aml-mixtral-deployment-1"

API_TOKEN = "mVHiH89aX9KtWvartgwXG5V1fmCAjYEy"

vllm_ttft = 0
vllm_time_to_next_token = []
vllm_tokens_received = 0

nim_ttft = 0
nim_time_to_next_token = []
nim_tokens_received = 0

# Set the title of the Streamlit app
st.set_page_config(layout="wide")
st.header('NIM OFF vs NIM ON')

col_header, col_dropdown, col_toggle = st.columns([2, 2, 2])

# Subheader
col_header.subheader('Mixtral 8x7B - Azure NC A100 v4')

# Injecting custom CSS via Markdown to style around the toggle
col_toggle.markdown(
    """
    <style>
    .stToggle > label {
        font-size: 16px; /* Adjusting label size which can give a perception of a larger toggle */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Dropdown for model selection
model_options = ["Llama 3 7B", "Mistral 7B", "Mixtral 8x7B"]
selected_model = col_dropdown.selectbox("Choose model:", model_options)

# Toggle button
def handle_toggle_change():
    if 'toggle_previous' in st.session_state:
        if st.session_state.toggle_previous != st.session_state.self_hosting:
            st.session_state.toggle_previous = st.session_state.self_hosting
            st.toast(f"Self Hosting is now {'enabled' if st.session_state.self_hosting else 'disabled'}")
    else:
        st.session_state.toggle_previous = st.session_state.self_hosting

self_hosting = col_toggle.toggle(
    "Enable Self Hosting", 
    value=False, 
    key='self_hosting', 
    on_change=handle_toggle_change, 
    label_visibility="visible"
)

col1, col2 = st.columns(2)

def check_vllm_health():
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_TOKEN}',
        'azureml-model-deployment': deployment_1,
    }
    url = api_url_1 + "/health"
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return True
        else:
            return False
    except requests.exceptions.RequestException as e:
        return False
    
def check_nim_health():
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_TOKEN}',
        'azureml-model-deployment': deployment_2,
    }
    url = api_url_2 + "/v1/models"
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return True
        else:
            return False
    except requests.exceptions.RequestException as e:
        return False    
    

def get_vllm_stream_response(messages):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_TOKEN}',
        'azureml-model-deployment': deployment_1,
    }

    body = {
        "model": model_1,
        "messages": messages,
        "max_tokens": 1024,
        "stream": True
    }

    url = api_url_1 + "/v1/chat/completions"

    error_msg = ""
    error_response_code = -1

    global vllm_ttft
    global vllm_tokens_received
    global vllm_time_to_next_token

    start_time = time.monotonic()
    most_recent_received_token_time = time.monotonic()

    try:
        with requests.post(
            url,
            json=body,
            stream=True,
            timeout=180,
            headers=headers,
        ) as response:
            if response.status_code != 200:
                error_msg = response.text
                error_response_code = response.status_code
                response.raise_for_status()
            for chunk in response.iter_lines(chunk_size=None):
                # Parse chunk
                chunk = chunk.strip()
                if not chunk:
                    continue
                stem = "data: "
                chunk = chunk[len(stem) :]
                if chunk == b"[DONE]":
                    continue
                data = json.loads(chunk)                
                
                # Check errors
                if "error" in data:
                    error_msg = data["error"]["message"]
                    error_response_code = data["error"]["code"]
                    raise RuntimeError(data["error"]["message"])                
                
                delta = data["choices"][0]["delta"]
                if delta.get("content", None):
                    vllm_tokens_received += 1
                    if not vllm_ttft:
                        vllm_ttft = time.monotonic() - start_time
                        vllm_time_to_next_token.append(vllm_ttft)
                    else:
                        vllm_time_to_next_token.append(
                            time.monotonic() - most_recent_received_token_time
                        )
                    most_recent_received_token_time = time.monotonic()                    
                    yield delta["content"]

    except Exception as e:
        print(f"Warning Or Error: {error_msg} {error_response_code}")


def get_nim_stream_response(messages):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_TOKEN}',
        'azureml-model-deployment': deployment_2,
    }

    body = {
        "model": model_2,
        "messages": messages,
        "max_tokens": 1024,
        "stream": True
    }

    error_msg = ""
    error_response_code = -1

    url = api_url_2 + "/v1/chat/completions"

    global nim_ttft
    global nim_tokens_received
    global nim_time_to_next_token

    start_time = time.monotonic()
    most_recent_received_token_time = time.monotonic()

    try:
        with requests.post(
            url,
            json=body,
            stream=True,
            timeout=180,
            headers=headers,
        ) as response:
            if response.status_code != 200:
                error_msg = response.text
                error_response_code = response.status_code
                response.raise_for_status()
            for chunk in response.iter_lines(chunk_size=None):
                # Parse chunk
                chunk = chunk.strip()
                if not chunk:
                    continue
                stem = "data: "
                chunk = chunk[len(stem) :]
                if chunk == b"[DONE]":
                    continue
                data = json.loads(chunk)                
                
                # Check errors
                if "error" in data:
                    error_msg = data["error"]["message"]
                    error_response_code = data["error"]["code"]
                    raise RuntimeError(data["error"]["message"])                
                
                delta = data["choices"][0]["delta"]
                if delta.get("content", None):
                    nim_tokens_received += 1
                    if not nim_ttft:
                        nim_ttft = time.monotonic() - start_time
                        nim_time_to_next_token.append(nim_ttft)
                    else:
                        nim_time_to_next_token.append(
                            time.monotonic() - most_recent_received_token_time
                        )
                    most_recent_received_token_time = time.monotonic()                    
                    yield delta["content"]

    except Exception as e:
        st.error(f"Warning Or Error: {error_msg} {error_response_code}")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

with col1:
    if check_vllm_health():
        st.markdown('<p style="color:green; font-size:16px; text-align:right;">NIM-OFF Status: 🟢 Up</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:red; font-size:16px; text-align:right;">NIM-OFF Status: 🔴 Down</p>', unsafe_allow_html=True)

with col2:
    if check_nim_health():
        st.markdown('<p style="color:green; font-size:16px; text-align:right;">NIM-ON Status: 🟢 Up</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:red; font-size:16px; text-align:right;">NIM-ON Status: 🔴 Down</p>', unsafe_allow_html=True)


for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    if role == "user":
        with col1:
            with st.chat_message(role):
                st.markdown(content)
        with col2:
            with st.chat_message(role):
                st.markdown(content)
    elif role == "VLLM":
        with col1:
            with st.chat_message("assistant"):
                st.markdown(content)
    elif role == "NIM":
        with col2:
            with st.chat_message("assistant"):
                st.markdown(content)

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with col1:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            stream = get_vllm_stream_response(
                messages = [
                    {"role": "assistant" if m["role"] == "VLLM" else m["role"], "content": m["content"]}
                    for m in st.session_state.messages if m["role"] in ["user", "VLLM"]
                ]
            )
            response = st.write_stream(stream)
            itl = sum(vllm_time_to_next_token)
            metrics = "Received: " +  "{:.0f}".format(vllm_tokens_received) +  " tokens"  + "\tITL: " + "{:.2f}".format(itl) + " seconds"
            vllm_throughput = vllm_tokens_received/itl
            st.markdown(f'''`{metrics}`''')
    st.session_state.messages.append({"role": "VLLM", "content": response})
    with col2:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            stream = get_nim_stream_response(
                messages = [
                    {"role": "assistant" if m["role"] == "NIM" else m["role"], "content": m["content"]}
                    for m in st.session_state.messages if m["role"] in ["user", "NIM"]
                ]
            )
            response = st.write_stream(stream)
            itl = sum(nim_time_to_next_token)
            nim_throughput = nim_tokens_received/itl
            perf_gain = nim_throughput/vllm_throughput
            metrics = "Received: " +  "{:.0f}".format(nim_tokens_received) +  " tokens"  + "\tITL: " + "{:.2f}".format(itl) + " seconds"
            st.markdown(f'''`{metrics}`''')
            gain = "Perf gain: "+"{:.1f}".format(perf_gain) + "X🚀"
            st.markdown(f'''**{gain}**''')
    st.session_state.messages.append({"role": "NIM", "content": response})
