#!/bin/bash

# Installation script for mistral-ocr

echo "🔧 Installing mistral-ocr..."

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create a wrapper script
WRAPPER_PATH="$HOME/.local/bin/mistral-ocr"
mkdir -p "$HOME/.local/bin"

cat > "$WRAPPER_PATH" << EOF
#!/bin/bash
SCRIPT_DIR="$SCRIPT_DIR"
CURRENT_DIR="\$(pwd)"
cd "\$SCRIPT_DIR" && MISTRAL_OCR_CWD="\$CURRENT_DIR" poetry run mistral-ocr "\$@"
EOF

chmod +x "$WRAPPER_PATH"

echo "✅ Installation complete!"
echo ""
echo "Make sure ~/.local/bin is in your PATH by adding this to your ~/.zshrc or ~/.bash_profile:"
echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
echo ""
echo "Then reload your shell or run:"
echo "  source ~/.zshrc"
echo ""
echo "You can now use: mistral-ocr <file>"