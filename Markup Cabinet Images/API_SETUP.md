# API Key Setup Instructions

## Security Notice
**NEVER commit API keys directly in code!** They will be exposed publicly on GitHub.

## Setup Instructions

1. **Install required package:**
   ```bash
   pip install python-dotenv
   ```

2. **Create a .env file** in the `Markup Cabinet Images` folder with your API key:
   ```
   GOOGLE_VISION_API_KEY=your-api-key-here
   ```

3. **The .env file is already in .gitignore** so it won't be committed to GitHub

## Alternative: Using System Environment Variables

Instead of a .env file, you can set the environment variable directly:

**Windows (Command Prompt):**
```cmd
set GOOGLE_VISION_API_KEY=your-api-key-here
```

**Windows (PowerShell):**
```powershell
$env:GOOGLE_VISION_API_KEY="your-api-key-here"
```

**Linux/Mac:**
```bash
export GOOGLE_VISION_API_KEY=your-api-key-here
```

## Getting a Google Vision API Key

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable the Vision API
4. Go to Credentials → Create Credentials → API Key
5. Restrict the API key to Vision API only for security

## If Your Key Gets Exposed

1. **Immediately delete the exposed key** in Google Cloud Console
2. **Create a new key**
3. **Update your .env file** with the new key
4. **Never commit the .env file to git**