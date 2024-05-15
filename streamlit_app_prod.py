import streamlit as st
import time
import json
import requests

from transformers import AutoTokenizer

# Define the API endpoints and keys

nim_off_config = {
    "url": "https://nim-aml-endpoint-1.westeurope.inference.ml.azure.com",
    "key": "mVHiH89aX9KtWvartgwXG5V1fmCAjYEy",
    "model": "/var/azureml-app/azureml-models/mistralai-Mixtral-8x7B-Instruct-v01/5/mlflow_model_folder/data/model",
    "deployment_name": "os-aml-mixtral-deployment-1"
}

nim_on_config = {
    "url": "https://nim-aml-endpoint-1.westeurope.inference.ml.azure.com",
    "key": "mVHiH89aX9KtWvartgwXG5V1fmCAjYEy",
    "model": "mixtral-instruct",
    "deployment_name": "nim-aml-mixtral-deployment-1"
}

vllm_ttft = 0
vllm_time_to_next_token = []
vllm_tokens_received = 0

nim_ttft = 0
nim_time_to_next_token = []
nim_tokens_received = 0

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1", token="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")

# Set the title of the Streamlit app
st.set_page_config(layout="wide")
st.header('NIM OFF vs NIM ON')

def is_promptflow(url):
    return  "/score" in url

def generate_headers(url, key, deployment_name):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {key}'
    }

    if deployment_name:
        headers.update({'azureml-model-deployment': deployment_name})
    
    if is_promptflow(url):
        headers.update({"Accept": "text/event-stream"})
    
    return headers

def generate_body(url, model, messages):
    if is_promptflow(url):
        assert messages[-1]["role"] == "user"
        return {
            "question" : messages[-1]["content"]
        }
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": 1024,
        "stream": True
    }
    return body

def check_vllm_health(url, key, deployment_name):
    headers = generate_headers(url, key, deployment_name)
    try:
        response = requests.get(url + "/health", headers=headers)
        if response.status_code == 200:
            return True
        else:
            return False
    except requests.exceptions.RequestException as e:
        return False
    
def check_nim_health(url, key, deployment_name):
    headers = generate_headers(url, key, deployment_name)
    try:
        response = requests.get(url + "/v1/models", headers=headers)
        if response.status_code == 200:
            return True
        else:
            return False
    except requests.exceptions.RequestException as e:
        return False    
    

def get_vllm_stream_response(url, key, model, deployment_name, messages):
    headers = generate_headers(url, key, deployment_name)
    body = generate_body(url, model, messages)

    error_msg = ""
    error_response_code = -1

    global vllm_ttft
    global vllm_tokens_received
    global vllm_time_to_next_token

    start_time = time.monotonic()
    most_recent_received_token_time = time.monotonic()

    try:
        with requests.post(
            url + "/v1/chat/completions",
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


def get_nim_stream_response(url, key, model, deployment_name, messages):
    headers = generate_headers(url, key, deployment_name)
    body = generate_body(url, model, messages)

    error_msg = ""
    error_response_code = -1

    global nim_ttft
    global nim_tokens_received
    global nim_time_to_next_token

    start_time = time.monotonic()
    most_recent_received_token_time = time.monotonic()

    try:
        with requests.post(
            url + "/v1/chat/completions",
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

col1, col2, col3, col4, col5 = st.columns([2,2,2,2,1])

st.markdown("""
    <style>
    .custom-button {
        padding-top: 10px;
        padding-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

with col1:
    url = st.text_input("Endpoint", value=nim_on_config["url"])
    nim_on_config["url"] = url
with col2:
    key = st.text_input("API key", value=nim_on_config["key"])
    nim_on_config["key"] = key    
with col3:
    model = st.text_input("Model", value=nim_on_config["model"])
    nim_on_config["model"] = model
with col4:
    deployment_name = st.text_input("Deployment", value=nim_on_config["deployment_name"])
    nim_on_config["deployment_name"] = deployment_name
col5.markdown('<div class="custom-button"></div>', unsafe_allow_html=True)
if col5.button('Go'):
    st.session_state.messages = []

col1, col2 = st.columns(2)        

with col1:
    if check_vllm_health(nim_off_config["url"], nim_off_config["key"], nim_off_config["deployment_name"]):
        st.markdown('<p style="color:green; font-size:16px; text-align:right;">NIM-OFF Status: 🟢 Up</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:red; font-size:16px; text-align:right;">NIM-OFF Status: 🔴 Down</p>', unsafe_allow_html=True)

with col2:
    if check_nim_health(nim_on_config["url"], nim_on_config["key"], nim_on_config["deployment_name"]):
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
                nim_off_config["url"], nim_off_config["key"], nim_off_config["model"], nim_off_config["deployment_name"],
                messages = [
                    {"role": "assistant" if m["role"] == "VLLM" else m["role"], "content": m["content"]}
                    for m in st.session_state.messages if m["role"] in ["user", "VLLM"]
                ]
            )
            response = st.write_stream(stream)
            if len(response) > 0:
                itl = sum(vllm_time_to_next_token)
                vllm_tokens_received = len(tokenizer([response])['input_ids'][0])
                metrics = "Received: " +  "{:.0f}".format(vllm_tokens_received) +  " tokens"  + "\tITL: " + "{:.2f}".format(itl) + " seconds"
                vllm_throughput = vllm_tokens_received/itl
                st.markdown(f'''`{metrics}`''')
    st.session_state.messages.append({"role": "VLLM", "content": response})

    with col2:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            stream = get_nim_stream_response(
                nim_on_config["url"], nim_on_config["key"], nim_on_config["model"], nim_on_config["deployment_name"],
                messages = [
                    {"role": "assistant" if m["role"] == "NIM" else m["role"], "content": m["content"]}
                    for m in st.session_state.messages if m["role"] in ["user", "NIM"]
                ]
            )
            response = st.write_stream(stream)
            if len(response) > 0:
                itl = sum(nim_time_to_next_token)
                nim_tokens_received = len(tokenizer([response])['input_ids'][0])
                nim_throughput = nim_tokens_received/itl
                perf_gain = nim_throughput/vllm_throughput
                metrics = "Received: " +  "{:.0f}".format(nim_tokens_received) +  " tokens"  + "\tITL: " + "{:.2f}".format(itl) + " seconds"
                st.markdown(f'''`{metrics}`''')
                gain = "Perf gain: "+"{:.1f}".format(perf_gain) + "X🚀"
                st.markdown(f'''**{gain}**''')
    st.session_state.messages.append({"role": "NIM", "content": response})
