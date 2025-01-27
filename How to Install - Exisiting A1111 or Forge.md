# Installing Diffusers in Stable Diffusion WebUI Virtual Environment

Follow these steps to install the **Diffusers** library inside the virtual environment (venv) used by **Stable Diffusion WebUI (A1111)** or **Forge**.

---

## Step 1: Locate the Virtual Environment

Stable Diffusion WebUI (A1111) and Forge both set up their own Python virtual environment under the project directory, typically:

- **For Stable Diffusion WebUI (A1111):**  
  ```
  stable-diffusion-webui/venv/
  ```
- **For Forge:**  
  ```
  forge-ai/venv/
  ```

Ensure you are inside the correct project folder before proceeding.

---

## Step 2: Activate the Virtual Environment

### On Windows:

1. Open **Command Prompt (CMD)** or **PowerShell** and navigate to your Stable Diffusion WebUI directory:

   ```
   cd D:\stable-diffusion-webui
   ```

2. Activate the virtual environment:

   - **If using Command Prompt (CMD):**
     ```
     venv\Scripts\activate
     ```

   - **If using PowerShell:**
     ```
     .\venv\Scripts\Activate.ps1
     ```

---

### On Linux/MacOS:

1. Open a terminal and navigate to your Stable Diffusion WebUI directory:

   ```
   cd /path/to/stable-diffusion-webui
   ```

2. Activate the virtual environment:

   ```
   source venv/bin/activate
   ```

---

## Step 3: Install Diffusers in the Virtual Environment

Once the virtual environment is activated (you should see `(venv)` in the command prompt/terminal), run the following command to install the **Diffusers** library:

```
pip install diffusers
```

If you need a specific version for compatibility reasons, you can install it with:

```
pip install diffusers==0.25.0
```

---

## Step 4: Verify Installation

After installation, check if the `diffusers` module is available by running:

```
python -c "import diffusers; print(diffusers.__version__)"
```

If installed correctly, it should display the version number.

---

## Step 5: Deactivate the Virtual Environment (Optional)

Once installation is successful, deactivate the virtual environment:

```
deactivate
```

---

## Step 6: Modify WebUI to Use Diffusers

Once installed, your WebUI should be able to recognize and use the `diffusers` package. If it's not found, you may need to add the path manually in your Python script, for example:

```
import sys
sys.path.append("D:/stable-diffusion-webui/venv/Lib/site-packages")

from diffusers.schedulers.scheduling_utils import SchedulerMixin
```

---

## Troubleshooting Installation Issues

1. **If `pip` fails to install due to outdated packages**, try updating `pip` first:

   ```
   pip install --upgrade pip
   ```

2. **If the virtual environment is not recognized, ensure you are in the right directory.**  
   Use:

   ```
   where python  # Windows
   which python  # Linux/Mac
   ```

   It should point to the `venv` directory.

3. **If `diffusers` still cannot be found in WebUI**, manually add the package path in `webui-user.bat`:

   ```
   set PYTHONPATH=%CD%\venv\Lib\site-packages
   ```

---

## Summary of Steps

1. **Navigate to your WebUI folder:**  
   ```
   cd D:\stable-diffusion-webui
   ```

2. **Activate the virtual environment:**  
   ```
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   ```

3. **Install Diffusers:**  
   ```
   pip install diffusers
   ```

4. **Verify installation:**  
   ```
   python -c "import diffusers; print(diffusers.__version__)"
   ```

5. **Deactivate the virtual environment (optional):**  
   ```
   deactivate
   ```



