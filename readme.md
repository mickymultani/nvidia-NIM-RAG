# RAG using NVIDIA NIM with LangChain

## Overview

This project demonstrates the power and simplicity of NVIDIA NIM (NVIDIA Inference Model), a suite of optimized cloud-native microservices, by setting up and running a Retrieval-Augmented Generation (RAG) pipeline. NVIDIA NIM is designed to streamline the deployment and time-to-market of generative AI models across various environments, including cloud platforms, data centers, and GPU-accelerated workstations. By abstracting the complexities of AI model development and leveraging industry-standard APIs, NIM makes advanced AI technologies accessible to a broader range of developers.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip and virtualenv
- API key from https://build.nvidia.com/mistralai/mixtral-8x7b-instruct

### Installation

1. **Clone the repository**

    ```
    git clone https://github.com/mickymultani/nvidia-NIM-RAG.git
    cd nvidia-NIM-RAG
    ```

2. **Set up a virtual environment**

    Create a virtual environment named `nvidia`:

    ```
    python -m venv nvidia
    ```

    Activate the virtual environment:

    - On Windows:
        ```bash
        nvidia\Scripts\activate
        ```
    - On macOS/Linux:
        ```
        source nvidia/bin/activate
        ```

3. **Install dependencies**

    Install the required packages using pip:

    ```
    pip install -r requirements.txt
    ```

4. **Environment Variables**

    Create a `.env` file in the root directory of the project, and add your NVIDIA API key:

    ```
    NVIDIA_API_KEY=your_nvidia_api_key_here
    ```

    Replace `your_nvidia_api_key_here` with your actual NVIDIA API key.

### Usage

To run the project, execute the following command:

```bash
python nim.py
```

### Contributing
Contributions to this project are welcome!

### License
Distributed under the MIT License. 