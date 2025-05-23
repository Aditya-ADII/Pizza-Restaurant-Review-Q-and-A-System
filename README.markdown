# Pizza Restaurant Review Q&A System

This project uses **LangChain**, **Ollama**, and **Chroma** to answer questions about pizza restaurants based on reviews in `realistic_restaurant_reviews.csv`. It leverages **CUDA**, **cuDNN**, and **PyTorch** for GPU acceleration (tested on RTX 3060, 12 GB VRAM) and monitors GPU usage with `pynvml` in `Monitor_cuda.py`.

## Features
- Answers questions (e.g., “What’s the best pizza in town?”) using review data.
- GPU-accelerated with `llama3.2:latest` (2.0 GB) and `mxbai-embed-large:latest` (669 MB).
- Uses Chroma for vector search (top 5 reviews).
- Monitors VRAM with `pynvml` and `nvidia-smi`.
- Interactive CLI for questions.

## Prerequisites
- **Hardware:** NVIDIA GPU (e.g., RTX 3060, 12 GB VRAM), 16 GB RAM.
- **Software:** Windows 10/11 (tested), Python 3.10+, NVIDIA driver 566.36+, CUDA 12.6/12.7, Ollama.
- **Dataset:** `realistic_restaurant_reviews.csv` with `Title`, `Review`, `Rating`, `Date`.

## Setup Instructions
1. **Install NVIDIA Drivers and CUDA:**
   - Get NVIDIA driver from [NVIDIA](https://www.nvidia.com/Download/index.aspx).
   - Verify: `nvidia-smi` (should show CUDA 12.6/12.7).
   - cuDNN is bundled with Py pytorch.

2. **Install Python and Virtual Environment:**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1  # Windows
   ```

3. **Install Dependencies:**
   ```powershell
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   pip install -r requirements.txt
   ```
   `requirements.txt`:
   ```
   langchain
   langchain-ollama
   langchain-chroma
   pandas
   pynvml
   ```

4. **Install Ollama:**
   - Download from [ollama.ai](https://ollama.ai/download).
   - Pull models:
     ```powershell
     ollama pull llama3.2:latest
     ollama pull mxbai-embed-large:latest
     ```
   - Verify:
     ```powershell
     ollama list
     ```
     Should show:
     ```
     NAME                          ID              SIZE      MODIFIED
     llama3.2:latest               a----------5    2.0 GB    Recently
     mxbai-embed-large:latest      4----------7    669 MB    Recently
     ```

5. **Prepare Dataset:**
   - Place `realistic_restaurant_reviews.csv` in project root.
   - Format:
     ```csv
     Title,Review,Rating,Date
     "Great Pizza","Crispy crust, fresh toppings!",5,"2023-10-01"
     ```

## Project Structure
```
pizza-restaurant-review/
├── main.py                  # Runs Q&A system
├── vector.py                # Handles embeddings and vector database
├── Monitor_cuda.py          # Monitors GPU memory
├── requirements.txt         # Python dependencies
├── realistic_restaurant_reviews.csv  # Review dataset
├── chrome_langchain_db/     # Chroma database (auto-generated)
└── venv/                    # Virtual environment
```

## How It Works
1. **Data Loading (`vector.py`):**
   - Reads CSV, combines `Title` and `Review`, creates `Document` objects with `rating`, `date`.
2. **Embedding (`vector.py`):**
   - Uses `mxbai-embed-large:latest` (669 MB, 1024 dimensions) for review embeddings.
   - Stores in Chroma (`chrome_langchain_db`), retrieves top 5 reviews.
3. **Q&A (`main.py`):**
   - Takes user question, retrieves reviews, uses `llama3.2:latest` (2.0 GB) to answer.
4. **GPU Acceleration:**
   - Ollama uses PyTorch with CUDA/cuDNN, ~4-5 GB VRAM.
5. **Monitoring:**
   - `Monitor_cuda.py` uses `pynvml` for GPU stats.
   - `nvidia-smi` shows `ollama.exe`/`python.exe` usage.

## Running the Project
1. **Activate Environment:**
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
2. **Run Q&A:**
   ```powershell
   python main.py
   ```
   - Ask questions (e.g., “What’s the best pizza in town?”), type `q` to quit.
   - Example:
     ```
     -------------------------------
     Ask your question (q to quit): whats the best pizza in town
     Based on reviews, [Pizza Place] has the best pizza for its crispy crust.
     ```
3. **Run GPU Monitor:**
   ```powershell
   python Monitor_cuda.py
   ```
   - Outputs:
     ```
     Total GPU memory: 12288.00 MB
     Free GPU memory: ~7292.00 MB
     Used GPU memory: ~4824.00 MB
     ```

## Monitoring GPU Usage
- **In-Script (`Monitor_cuda.py`):**
  ```python
  import pynvml
  try:
      pynvml.nvmlInit()
      handle = pynvml.nvmlDeviceGetHandleByIndex(0)
      mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
      print(f"Total GPU memory: {mem_info.total / 1024**2:.2f} MB")
      print(f"Free GPU memory: {mem_info.free / 1024**2:.2f} MB")
      print(f"Used GPU memory: {mem_info.used / 1024**2:.2f} MB")
  except pynvml.NVMLError as e:
      print(f"NVML Error: {e}")
  finally:
      pynvml.nvmlShutdown()
  ```
- **External (`nvidia-smi`):**
  ```powershell
  nvidia-smi
  ```
  - Look for `ollama.exe`/`python.exe` using ~4-5 GB VRAM.
  - Continuous:
    ```powershell
    nvidia-smi --query --display=COMPUTE,MEMORY -l 2
    ```

## Dependencies
- **Python Packages** (`requirements.txt`):
  - `langchain`
  - `langchain-ollama`
  - `langchain-chroma`
  - `vector`
  - `pandas`
  - `pynvml`
- **PyTorch with CUDA:**
  ```powershell
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
  ```
- **Ollama Models:**
  - `llama3.2:latest` (a----------5, 2.0 GB)
  - `mxbai-embed-large:latest` (4----------7, 669 MB)
- **NVIDIA Stack:**
  - CUDA 12.6/12.7
  - cuDNN (bundled with PyTorch)
  - NVIDIA driver 566.36+

## Dataset
- **File:** `realistic_restaurant_reviews.csv`
- **Format:** CSV with:
  - `Title`: Review title
  - `Review`: Review text
  - `Rating`: 1-5
  - `Date`: e.g., “2023-10-01”
- **Usage:** Loaded by `vector.py` for embeddings.
