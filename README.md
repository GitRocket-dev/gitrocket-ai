# GitRocket-AI 🚀

<div align="center">

![GitRocket AI Logo](https://via.placeholder.com/800x200/000000/FFFFFF?text=GitRocket+AI+Terminal+Assistant)

**Access 52+ FREE AI Models | Built-in Diagnostics | API Debugger**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![OpenRouter](https://img.shields.io/badge/OpenRouter-API-green.svg)](https://openrouter.ai)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Free](https://img.shields.io/badge/Free-Tier-success.svg)](https://openrouter.ai)

*A powerful terminal-based AI assistant that gives you access to dozens of free AI models through OpenRouter*

</div>

## ✨ Features

### 🎯 **Core Capabilities**
- **52+ Free AI Models** - Access to Google, Meta, Microsoft, Mistral, and more
- **Smart Model Selection** - Interactive browser with recommendations
- **Real-time Chat** - Clean, formatted conversations with AI
- **Built-in Diagnostics** - Comprehensive API testing and debugging
- **Rate Limit Handling** - Automatic retry and optimization

### 🎨 **User Experience**
- **Beautiful Terminal UI** - Colors, borders, and ASCII art
- **Multiple Interfaces** - Choose your preferred way to select models
- **Session Management** - Conversation history and context
- **Quick Commands** - Easy-to-remember chat commands

### 🔧 **Advanced Features**
- **API Debugger** - Comprehensive testing and troubleshooting
- **Model History** - Track and quickly access recently used models
- **Smart Categorization** - Models organized by use case
- **Search & Filter** - Find the perfect model for your needs
- **Configuration Management** - Easy settings customization

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- OpenRouter API account (free)

### Installation

1. **Clone or download the script:**
```bash
# Download the script directly
wget https://raw.githubusercontent.com/your-repo/gitrocket-ai/main/gitrocket_ai.py

# Or clone the repository
git clone https://github.com/your-repo/gitrocket-ai.git
cd gitrocket-ai
```

2. **Make it executable:**
```bash
chmod +x gitrocket_ai.py
```

3. **Install required dependencies:**
```bash
pip install requests
```

### First-Time Setup

Run the script and follow the interactive setup wizard:

```bash
./gitrocket_ai.py
```

The setup wizard will guide you through:
1. **API Key Configuration** - Get your free OpenRouter API key
2. **Model Selection** - Choose from 52+ free AI models
3. **Initial Testing** - Verify everything works correctly

#### Getting Your API Key

1. Visit [OpenRouter Keys](https://openrouter.ai/keys)
2. Sign up or log in to your account
3. Create a new API key
4. Copy the key (starts with `sk-`) and enter it in the setup wizard

## 🎮 Usage

### Starting a Chat Session

```bash
./gitrocket_ai.py
```

The application will start with a beautiful ASCII art intro and ready for chatting.

### Chat Commands

| Command | Description |
|---------|-------------|
| `help` | Show available commands |
| `clear` | Reset conversation history |
| `test` | Run API diagnostic test |
| `settings` | Open settings menu |
| `quit`/`exit`/`bye` | End the session |

### Example Session

```
💬 You: Hello! Can you help me write a Python function to calculate factorial?

🤔 Thinking...

🤖 AI:
Sure! Here's a Python function to calculate factorial:

```python
def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n-1)

# Example usage:
print(factorial(5))  # Output: 120
```

This function uses recursion to calculate the factorial. For large numbers, you might want to use an iterative approach to avoid recursion limits.
```

## 🎯 Model Selection

GitRocket-AI offers three ways to choose your AI model:

### 1. 🎯 Smart Recommendations
**Perfect for beginners** - Models are categorized by use case:
- **💬 Chat & Conversation** - Best for general conversations
- **📝 Writing & Content** - Optimized for creative writing
- **🔍 Analysis & Reasoning** - Great for logic and analysis
- **💻 Coding & Technical** - Specialized for programming
- **🌐 Multilingual** - Excellent for multiple languages

### 2. 🔍 Interactive Browser
**For power users** - Full-featured browser with:
- **Search** - Find models by name or description
- **Filtering** - Show only free models or all models
- **Pagination** - Browse through all available models
- **Detailed Info** - Context length, pricing, descriptions

### 3. ⚡ Quick Pick
**Fast and reliable** - Choose from pre-tested reliable models:
- `google/gemma-2-9b-it:free`
- `microsoft/wizardlm-2-8x22b:free`
- `meta-llama/llama-3.1-8b-instruct:free`
- `mistralai/mistral-nemo:free`
- `cognitivecomputations/dolphin3.0-mistral-24b:free`

## ⚙️ Settings & Configuration

Access settings with the `settings` command during chat:

### Available Settings
1. **Change Model** - Switch to a different AI model
2. **Change API Key** - Update your OpenRouter API key
3. **Max Tokens** - Adjust response length (1-4000)
4. **Temperature** - Control creativity (0.0-2.0)
5. **Debug Mode** - Enable/disable detailed logging
6. **API Debugger** - Comprehensive API testing tool

### Configuration Files
- **Main Config**: `~/.config/gitrocket_ai/config.json`
- **Debug Logs**: `~/.config/gitrocket_ai/debug_log.json`
- **Model History**: `~/.config/gitrocket_ai/model_history.json`

## 🔧 Troubleshooting

### Common Issues

#### ❌ "Invalid API Key"
1. Verify your API key at [OpenRouter Keys](https://openrouter.ai/keys)
2. Ensure the key starts with `sk-`
3. Check that your account is active

#### ❌ "Rate Limited"
- Free tier has rate limits
- Wait 10-30 seconds between requests
- Use the `test` command to check current limits

#### ❌ "Cannot Reach OpenRouter"
- Check your internet connection
- Verify `https://openrouter.ai` is accessible
- Try the built-in connectivity test

### Using the API Debugger

Run comprehensive diagnostics with the API Debugger:

1. Go to **Settings** → **API Debugger**
2. The tool will test:
   - Network connectivity
   - API key validity
   - Model availability
   - Chat functionality
   - Configuration settings

### Debug Mode

Enable debug mode for detailed logging:

1. Go to **Settings** → **Toggle Debug Mode**
2. Choose debug level:
   - **Basic** - Minimal logging
   - **Detailed** - More information
   - **Full** - Complete request/response data

## 🆓 Free Tier Information

### What's Available
- **52+ Free Models** from top providers
- **Generous Rate Limits** - Suitable for personal use
- **No Credit Card Required** - Completely free to start
- **Various Capabilities** - Chat, coding, writing, analysis

### Rate Limits
- **Requests**: Limited per minute/hour
- **Tokens**: Generous but limited monthly
- **Models**: Some models have individual limits

### Tips for Free Tier
- Use smaller models for faster responses
- Keep conversations concise
- Use the `test` command to check availability
- Switch models if one is rate-limited

## 🛠️ Technical Details

### Supported Models
GitRocket-AI automatically detects and supports all free models available through OpenRouter, including:

- **Google**: Gemma, Gemma-2
- **Meta**: Llama 3.1, Code Llama
- **Microsoft**: WizardLM, Phi-3
- **Mistral**: Mistral-Nemo, Mixtral
- **And many more...**

### API Integration
- **OpenRouter API** - Unified interface to multiple AI providers
- **Automatic Retry** - Handles rate limits gracefully
- **Error Handling** - Comprehensive error messages and recovery
- **Session Management** - Maintains conversation context

### Security & Privacy
- **Local Configuration** - All settings stored locally
- **Secure File Permissions** - Config files protected
- **No Data Storage** - Conversations not stored permanently
- **API Key Encryption** - Keys stored in user config directory

## 📁 Project Structure

```
gitrocket_ai.py          # Main application script
~/.config/gitrocket_ai/  # Configuration directory
├── config.json          # User settings and API key
├── debug_log.json       # Debug session logs
└── model_history.json   # Recently used models
```

## 🐛 Bug Reports & Feature Requests

Found a bug? Have a feature idea? Please open an issue with:

1. **Description** of the problem or feature
2. **Steps to reproduce** (for bugs)
3. **Expected behavior**
4. **Environment details** (OS, Python version)

## 🤝 Contributing

We welcome contributions! Areas needing help:

- **Additional AI Providers** - Expand model support
- **UI Enhancements** - Improve terminal experience
- **Documentation** - Improve guides and examples
- **Testing** - Add unit and integration tests

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenRouter** for providing free access to multiple AI models
- **All AI Model Providers** for their incredible work
- **The Python Community** for excellent libraries and tools

---

<div align="center">

**Ready to launch?** 🚀

```bash
./gitrocket_ai.py
```

*Start your AI journey today with GitRocket-AI!*

</div>
