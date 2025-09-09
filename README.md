## Paper Mining

This project extracts and analyzes patterns from research papers in PDF format.

### Setup Instructions

1. **Google API Key**
	- Set your Google API key as an environment variable:
	  ```bash
	  export GOOGLE_API_KEY=your_api_key_here
	  ```

2. **Add Papers**
	- Place your PDF papers in the `papers/` folder.
	- To use a different folder, update the `paper_folder` variable in `main.py`.

3. **Disable Cleaning Step (Optional)**
	- To skip the paper cleaning step, comment out the `clean_papers()` function call at the bottom of `main.py`.

### Usage

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the main script:
```bash
python main.py
```

---
For more details, see the code and comments in `main.py` and related modules.