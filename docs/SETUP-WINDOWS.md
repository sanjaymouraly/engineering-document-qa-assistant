# Windows setup

Tested commands are PowerShell 5.1 / Windows Terminal; the `cmd.exe` equivalents are noted where
they differ. Python 3.11 or newer is required. No build tools are needed — every dependency
ships wheels for Windows (pypdfium2, Pillow, numpy, reportlab, pdfplumber).

## 1. Install Python

Either:

* **python.org** — download the 64-bit installer, tick *Add python.exe to PATH*, then
  `py -3.12 --version`; or
* **winget**:

```powershell
winget install -e --id Python.Python.3.12
winget install -e --id Git.Git
```

Close and reopen the terminal so PATH updates apply.

## 2. Get the code and install

```powershell
git clone <your-fork-url> engineering-rag-assistant
cd engineering-rag-assistant
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # cmd.exe:  .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -e .
```

PowerShell needs the quotes for extras: `pip install -e ".[dev]"`.

If script activation is blocked:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Optional extras:

```powershell
pip install -e ".[dev]"         # pytest, coverage, playwright
pip install -e ".[semantic]"    # sentence-transformers (first use downloads ~90 MB)
pip install -e ".[ocr]"         # pytesseract; also install the binaries below
```

For OCR, install the tesseract Windows installer from
<https://github.com/UB-Mannheim/tesseract/wiki> and add its folder to `PATH`
(e.g. `C:\Program Files\Tesseract-OCR`). Without it, scanned pages are flagged
`needs_ocr` instead of appearing empty.

## 3. Try it with the synthetic corpus

```powershell
engrag build-demo
engrag ask "What are the rated voltage and frequency of DEMO-TR-01?"
```

The CLI prints units such as `mm²` and `°C`. If your console shows `?` or raises a
`UnicodeEncodeError`, switch the code page before running:

```powershell
chcp 65001
$env:PYTHONIOENCODING = "utf-8"
```

## 4. Run the app

Two terminals (or `Start-Process`):

```powershell
engrag api          # FastAPI on http://127.0.0.1:8502   (optional)
engrag web          # Streamlit on http://127.0.0.1:8501
```

Then open <http://127.0.0.1:8501>. To restrict the bind address:

```powershell
$env:ENGRAG_HOST = "127.0.0.1"
engrag web
```

The app binds `0.0.0.0` by default; Windows Firewall may prompt on first launch. Allow it only
on *Private* networks, or set `ENGRAG_HOST=127.0.0.1` — this tool has no accounts, so do not
expose it to a workgroup/public network.

## 5. Data location

Everything lives in the data directory (default `.\data` next to where you start the process):

```
data\engrag.db          index, pages, chunks, history, verification results
data\uploads\           your stored PDFs, named <sha256-prefix>_<filename>
data\renders\           cached page PNGs (safe to delete; they are regenerated)
data\exports\           reserved (created for file exports; the UI's *Download
                        answer+citations* hands the file to your browser instead)
data\demo_generated\    the synthetic demo PDFs
```

Move it with `$env:ENGRAG_DATA_DIR = "D:\engrag-data"`. Back up the whole folder to migrate; the
SQLite file plus `uploads\` is all state there is.

## 6. Start on login (optional)

```powershell
$action  = New-ScheduledTaskAction -Execute "D:\engineering-rag-assistant\.venv\Scripts\python.exe" `
          -Argument "-m engrag web" -WorkingDirectory "D:\engineering-rag-assistant"
$trigger = New-ScheduledTaskTrigger -AtLogOn
Register-ScheduledTask -TaskName "EngRagWeb" -Action $action -Trigger $trigger
```

## 7. Verify the installation

```powershell
python -m pytest tests -q
python scripts\run_eval.py --json-out $env:TEMP\eval.json
```

Expected: 86 tests pass, evaluation reports `18/18`. The suite uses temporary data directories
and a mocked HTTP transport, so it neither needs a network nor touches `.\data`.

## 8. API calls from PowerShell

`curl` in PowerShell is an alias for `Invoke-WebRequest`; use the real binary:

```powershell
curl.exe -F "file=@C:\docs\manual.pdf" http://127.0.0.1:8502/api/v1/documents

curl.exe -X POST http://127.0.0.1:8502/api/v1/ask -H "content-type: application/json" -d "{\"question\":\"What is the rated current on the HV side?\",\"top_k\":8}"
```

or with `Invoke-RestMethod`:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8502/api/v1/ask -Method Post `
  -ContentType "application/json" `
  -Body (@{ question = "What is the rated current on the HV side?"; top_k = 8 } | ConvertTo-Json)
```

## 9. Docker instead of a venv (optional)

With Docker Desktop running (WSL2 backend):

```powershell
docker compose up --build      # UI → http://localhost:8501, API → http://localhost:8502/docs
```

State lives in the `engrag-data` volume, so `docker compose down` keeps your documents.

## Troubleshooting (Windows)

| Symptom | Cause and fix |
|---|---|
| `engrag` is not recognized | The venv is not activated, or you installed without `-e` into the venv. Run `.\.venv\Scripts\Activate.ps1` first, or use `python -m engrag …`. |
| `UnicodeEncodeError: 'charmap' codec …` | Console code page is not UTF-8: `chcp 65001` and `$env:PYTHONIOENCODING="utf-8"`. |
| `PermissionError` writing `data\` | The folder is inside `Program Files` or a roaming profile. Set `ENGRAG_DATA_DIR` to a writable path, e.g. `$env:LOCALAPPDATA\engrag`. |
| Antivirus blocks/deletes uploads | Exclude your data directory; the app only reads PDFs and writes PNG caches, but scanning large PDF bursts is slow. |
| Long path errors on deep temp dirs | Enable Win32 long paths (`git config --system core.longpaths true`, plus the `LongPathsEnabled` registry value) or keep the repo near a drive root. |
| `sentence-transformers` install pulls huge CUDA wheels | Use the CPU index: `pip install torch --index-url https://download.pytorch.org/whl/cpu` first, then `pip install -e ".[semantic]"`. |
| Port already allocated | `netstat -ano | findstr :8501`, then `engrag web --port 8601`. |
