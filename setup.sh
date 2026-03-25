#!/bin/bash

# Create .streamlit directory if it doesn't exist
mkdir -p ~/.streamlit/

# Create config.toml
cat > ~/.streamlit/config.toml <<EOF
[theme]
primaryColor = "#667eea"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[client]
showErrorDetails = false
toolbarMode = "minimal"

[server]
headless = true
port = \$PORT
enableCORS = false
enableXsrfProtection = true
maxUploadSize = 500

[logger]
level = "info"
EOF

echo "✅ Streamlit configuration created"