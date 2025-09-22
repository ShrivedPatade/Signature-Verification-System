@echo off
setlocal enabledelayedexpansion

REM --- Configuration ---
set "GDRIVE_FILE_ID=1tQOuGeYJmZOhw9v9HdUQIRwiPYSTRlSa"

REM --- Script ---
set "TARGET_DIR=SigVer\models"
set "ZIP_FILE_PATH=%TARGET_DIR%\models.zip"
set "COOKIE_FILE=%TARGET_DIR%\cookies.txt"
set "TEMP_HTML_FILE=%TARGET_DIR%\temp_response.html"

echo Checking for model directory...
if not exist "%TARGET_DIR%" (
    echo Directory not found. Creating %TARGET_DIR%...
    mkdir "%TARGET_DIR%"
    if !errorlevel! neq 0 (
        echo Failed to create directory. Please check permissions.
        exit /b 1
    )
    echo Directory created successfully.
) else (
    echo Directory %TARGET_DIR% already exists.
)

echo.
echo Starting download from Google Drive...
echo This may take a moment.

REM --- Step 1: Get cookies and download the confirmation page ---
curl -c "%COOKIE_FILE%" -L "https://drive.google.com/uc?export=download&id=%GDRIVE_FILE_ID%" > "%TEMP_HTML_FILE%"

REM --- Step 2: Find the confirmation URL in the downloaded HTML by looking for the download form ---
set "CONFIRM_URL_PART="
for /f "usebackq tokens=4 delims==\" " %%i in (`findstr /i "download-form" "%TEMP_HTML_FILE%"`) do (
    set "CONFIRM_URL_PART=%%i"
)

REM --- Step 3: Download the actual file using the confirmation link if it was found ---
if defined CONFIRM_URL_PART (
    echo Confirmation required. Proceeding with download...
    set "CONFIRM_URL_PART=!CONFIRM_URL_PART:amp;=^&!"
    set "FINAL_URL=https://drive.google.com!CONFIRM_URL_PART!"
    
    curl -b "%COOKIE_FILE%" -L -o "%ZIP_FILE_PATH%" "!FINAL_URL!"
) else (
    echo No confirmation page detected. Assuming direct download.
    REM If no form is found, the downloaded HTML is likely the actual file
    move /Y "%TEMP_HTML_FILE%" "%ZIP_FILE_PATH%"
)

REM --- Cleanup temporary files ---
if exist "%COOKIE_FILE%" del "%COOKIE_FILE%"
if exist "%TEMP_HTML_FILE%" del "%TEMP_HTML_FILE%"

REM --- Verify that the download was successful ---
if not exist "%ZIP_FILE_PATH%" (
    echo.
    echo ERROR: Download failed. The final zip file was not created.
    exit /b 1
)

rem Check file size. If it's still tiny (<10KB), the download likely failed.
for %%F in ("%ZIP_FILE_PATH%") do set "FILESIZE=%%~zF"

if !FILESIZE! LSS 10000 (
    echo.
    echo ERROR: Download failed. The downloaded file is too small.
    echo This can happen if Google Drive changed its download mechanism.
    echo Please try downloading the file manually from your browser.
    del "%ZIP_FILE_PATH%"
    exit /b 1
)

echo Download complete.
echo.
echo Extracting models...
REM Use tar to extract the contents of the zip file.
tar -xf "%ZIP_FILE_PATH%" -C "%TARGET_DIR%"

if !errorlevel! neq 0 (
    echo ERROR: Extraction failed. The downloaded file might be corrupt or not a valid zip archive.
    exit /b 1
)

echo Deleting temporary zip file...
del "%ZIP_FILE_PATH%"

echo.
echo ---
echo Successfully downloaded and installed models to %TARGET_DIR%
echo ---
echo.
pause

