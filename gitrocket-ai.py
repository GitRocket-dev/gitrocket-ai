#!/usr/bin/env python3

import requests
import json
import os
import readline
import time
from pathlib import Path
from datetime import datetime

# Configuration
CONFIG_DIR = Path.home() / '.config' / 'gitrocket_ai'
CONFIG_FILE = CONFIG_DIR / 'config.json'
DEBUG_FILE = CONFIG_DIR / 'debug_log.json'
MODELS_URL = "https://openrouter.ai/api/v1/models"
API_KEY_URL = "https://openrouter.ai/keys"
HISTORY_FILE = CONFIG_DIR / 'model_history.json'

# Default configuration
DEFAULT_CONFIG = {
    "api_key": "",
    "model": "deepseek/deepseek-r1-distill-qwen-7b:free",
    "api_url": "https://openrouter.ai/api/v1/chat/completions",
    "max_tokens": 800,
    "temperature": 0.7,
    "debug_mode": False,
    "debug_level": "basic"  # basic, detailed, full
}

# Rate limiting
last_request_time = 0
MIN_REQUEST_INTERVAL = 10  # Reduced to 10 seconds for testing

# Debugging
debug_session = {
    "session_start": datetime.now().isoformat(),
    "requests": [],
    "errors": [],
    "config_changes": []
}

def ensure_config_dir():
    """Ensure configuration directory exists"""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

def load_config():
    """Load configuration from file"""
    ensure_config_dir()
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
            # Merge with defaults to ensure all keys exist
            return {**DEFAULT_CONFIG, **config}
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            return DEFAULT_CONFIG
    return DEFAULT_CONFIG

def save_config(config):
    """Save configuration to file"""
    ensure_config_dir()
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        os.chmod(CONFIG_FILE, 0o600)  # Secure file permissions
        return True
    except Exception as e:
        print(f"❌ Error saving config: {e}")
        return False

def get_model_history():
    """Get recently used models"""
    ensure_config_dir()
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_model_history(model_id):
    """Save model to history"""
    ensure_config_dir()
    history = get_model_history()
    
    # Remove if already exists
    history = [m for m in history if m['id'] != model_id]
    
    # Add to beginning
    history.insert(0, {
        'id': model_id,
        'last_used': datetime.now().isoformat()
    })
    
    # Keep only last 10
    history = history[:10]
    
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
        return True
    except:
        return False

def log_debug_event(event_type, data, config):
    """Log debug events if debug mode is enabled"""
    if not config.get('debug_mode', False):
        return
    
    debug_level = config.get('debug_level', 'basic')
    event = {
        "timestamp": datetime.now().isoformat(),
        "type": event_type,
        "data": data
    }
    
    # Filter data based on debug level
    if debug_level == "basic":
        if event_type == "api_request":
            filtered_data = {
                "model": data.get("model"),
                "messages_count": len(data.get("messages", [])),
                "max_tokens": data.get("max_tokens"),
                "temperature": data.get("temperature")
            }
            event["data"] = filtered_data
        elif event_type == "api_response":
            filtered_data = {
                "status_code": data.get("status_code"),
                "response_time": data.get("response_time"),
                "has_content": bool(data.get("content")),
                "error": data.get("error")
            }
            event["data"] = filtered_data
    
    debug_session["requests"].append(event)
    
    # Save to debug log file
    try:
        with open(DEBUG_FILE, 'w') as f:
            json.dump(debug_session, f, indent=2, default=str)
    except Exception as e:
        print(f"❌ Debug log error: {e}")

def check_connectivity():
    """Check if we can reach OpenRouter"""
    print("🔍 Checking connectivity to OpenRouter...")
    try:
        response = requests.get("https://openrouter.ai", timeout=10)
        print(f"📡 Connectivity: ✅ (Status {response.status_code})")
        return True
    except Exception as e:
        print(f"❌ Cannot reach OpenRouter: {e}")
        return False

def quick_api_test():
    """Simple direct test of OpenRouter API"""
    config = load_config()
    api_key = config.get('api_key', '')
    
    if not api_key:
        print("❌ No API key configured")
        return
    
    print("\n🧪 Running direct API test...")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/gitrocket-ai",
        "X-Title": "GitRocket-AI Test"
    }
    
    # Test with a very simple request
    payload = {
        "model": "google/gemma-2-9b-it:free",
        "messages": [{"role": "user", "content": "Say 'TEST OK' only."}],
        "max_tokens": 10,
        "temperature": 0.1
    }
    
    try:
        print("📤 Sending test request...")
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            print("✅ SUCCESS! Response:")
            print(f"   '{content}'")
            return True
        elif response.status_code == 429:
            retry_after = response.headers.get('Retry-After', 'unknown')
            print(f"❌ RATE LIMITED - Retry after: {retry_after}s")
            print(f"📋 Response: {response.text}")
        else:
            print(f"❌ FAILED - Status {response.status_code}")
            print(f"📋 Full response: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    return False

def get_available_models(api_key):
    """Fetch available models from OpenRouter - only show free models"""
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        start_time = time.time()
        response = requests.get(MODELS_URL, headers=headers, timeout=10)
        response_time = time.time() - start_time
        
        # Log API call for debugging
        log_debug_event("models_request", {
            "url": MODELS_URL,
            "headers": {k: "***" if "Authorization" in k else v for k, v in headers.items()},
            "response_time": response_time,
            "status_code": response.status_code
        }, {"debug_mode": True, "debug_level": "basic"})
        
        if response.status_code == 200:
            models_data = response.json()
            models = []
            for model in models_data.get('data', []):
                model_id = model.get('id', '')
                # Only include models that have ":free" at the end
                if model_id.endswith(':free'):
                    models.append({
                        'id': model_id,
                        'name': model.get('name', model_id),
                        'description': model.get('description', ''),
                        'context_length': model.get('context_length', 'Unknown'),
                        'pricing': model.get('pricing', {})
                    })
            return models
        else:
            print(f"❌ Error fetching models: {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ Error fetching models: {e}")
        log_debug_event("models_error", {"error": str(e)}, {"debug_mode": True, "debug_level": "basic"})
        return []

def get_reliable_free_models():
    """Return a list of known reliable free models"""
    return [
        "google/gemma-2-9b-it:free",
        "microsoft/wizardlm-2-8x22b:free", 
        "meta-llama/llama-3.1-8b-instruct:free",
        "mistralai/mistral-nemo:free",
        "cognitivecomputations/dolphin3.0-mistral-24b:free"
    ]

def switch_to_reliable_model(config):
    """Switch to a known reliable model"""
    reliable_models = get_reliable_free_models()
    current_model = config['model']
    
    if current_model in reliable_models:
        # Try a different one
        for model in reliable_models:
            if model != current_model:
                config['model'] = model
                print(f"🔄 Switching to reliable model: {model}")
                return config
    
    # If all else fails, use the first reliable model
    config['model'] = reliable_models[0]
    print(f"🔄 Switching to reliable model: {reliable_models[0]}")
    return config

def display_model_info(model, index):
    """Display model information with appealing formatting"""
    model_id = model['id']
    context = model['context_length']
    description = model['description'] or "No description available"
    
    # Color code based on model provider
    if 'google' in model_id:
        color = "\033[1;34m"  # Blue for Google
    elif 'meta' in model_id or 'llama' in model_id:
        color = "\033[1;33m"  # Yellow for Meta/Llama
    elif 'microsoft' in model_id:
        color = "\033[1;32m"  # Green for Microsoft
    elif 'mistral' in model_id:
        color = "\033[1;35m"  # Magenta for Mistral
    else:
        color = "\033[1;36m"  # Cyan for others
    
    print(f"{color}┌─ {index}. {model_id}\033[0m")
    print(f"\033[1;90m│   Context: {context} tokens\033[0m")
    
    # Format description with word wrapping
    desc_lines = []
    words = description.split()
    current_line = ""
    for word in words:
        if len(current_line + " " + word) <= 80:
            current_line += " " + word if current_line else word
        else:
            desc_lines.append(current_line)
            current_line = word
    if current_line:
        desc_lines.append(current_line)
    
    if desc_lines:
        print(f"\033[1;97m│   Description: {desc_lines[0]}\033[0m")
        for line in desc_lines[1:]:
            print(f"\033[1;97m│                 {line}\033[0m")
    
    # Show pricing info if available
    pricing = model.get('pricing', {})
    if pricing.get('prompt') == 0 and pricing.get('completion') == 0:
        print(f"\033[1;92m│   💰 Price: FREE\033[0m")
    else:
        prompt_price = pricing.get('prompt', '?')
        completion_price = pricing.get('completion', '?')
        print(f"\033[1;93m│   💰 Price: ${prompt_price}/1M tokens\033[0m")
    
    print("\033[1;90m└─────────────────────────────────────────────────────────────────\033[0m")
    print("")

def interactive_model_browser(config):
    """Interactive model browser with filtering and search"""
    print("\033[1;36m")
    print("┌─────────────────────────────────────────────────────┐")
    print("│ 🔍 Interactive Model Browser │")
    print("└─────────────────────────────────────────────────────┘")
    print("\033[0m")
    
    print("⏳ Loading available models...")
    models = get_available_models(config['api_key'])
    if not models:
        print("❌ Could not fetch models.")
        return config
    
    filtered_models = models
    current_page = 0
    models_per_page = 5
    
    while True:
        print("\033[1;36m")
        print("┌─────────────────────────────────────────────────────┐")
        print(f"│ 📋 Models ({len(filtered_models)} found) - Page {current_page + 1} │")
        print("└─────────────────────────────────────────────────────┘")
        print("\033[0m")
        
        # Display current page
        start_idx = current_page * models_per_page
        end_idx = min(start_idx + models_per_page, len(filtered_models))
        
        for i in range(start_idx, end_idx):
            display_model_info(filtered_models[i], i + 1)
        
        # Show navigation info
        print(f"\033[1;33mShowing {start_idx + 1}-{end_idx} of {len(filtered_models)} models\033[0m")
        print("")
        
        # Show commands
        print("\033[1;94mNavigation:\033[0m")
        print(" • \033[1;36m[number]\033[0m - Select model")
        print(" • \033[1;36mn\033[0m - Next page")
        print(" • \033[1;36mp\033[0m - Previous page") 
        print(" • \033[1;36ms [term]\033[0m - Search models")
        print(" • \033[1;36mf free\033[0m - Show only free models")
        print(" • \033[1;36mf all\033[0m - Show all models")
        print(" • \033[1;36mr\033[0m - Reset filters")
        print(" • \033[1;36mc\033[0m - Cancel")
        print("")
        
        choice = input("\033[1;35mChoose action: \033[0m").strip().lower()
        
        if choice == 'c':
            print("Model selection cancelled.")
            return config
        elif choice == 'n':
            if end_idx < len(filtered_models):
                current_page += 1
            else:
                print("❌ Already on last page")
        elif choice == 'p':
            if current_page > 0:
                current_page -= 1
            else:
                print("❌ Already on first page")
        elif choice.startswith('s '):
            search_term = choice[2:].lower()
            filtered_models = [m for m in models if search_term in m['id'].lower() or 
                             search_term in (m.get('description', '').lower())]
            current_page = 0
            print(f"🔍 Found {len(filtered_models)} models matching '{search_term}'")
        elif choice == 'f free':
            filtered_models = [m for m in models if m['id'].endswith(':free')]
            current_page = 0
            print(f"💰 Showing {len(filtered_models)} free models")
        elif choice == 'f all':
            filtered_models = models
            current_page = 0
            print("📋 Showing all models")
        elif choice == 'r':
            filtered_models = models
            current_page = 0
            print("🔄 Filters reset")
        elif choice.isdigit():
            model_index = int(choice) - 1
            if 0 <= model_index < len(filtered_models):
                selected_model = filtered_models[model_index]
                config['model'] = selected_model['id']
                save_model_history(selected_model['id'])
                print(f"✅ Selected: \033[1;32m{selected_model['id']}\033[0m")
                
                # Show confirmation with model details
                print("\n\033[1;36m" + "─" * 50 + "\033[0m")
                print(f"\033[1;94m🎯 Model Activated:\033[0m")
                print(f"   \033[1;36mName:\033[0m {selected_model['id']}")
                if selected_model['description']:
                    desc = selected_model['description'][:100] + "..." if len(selected_model['description']) > 100 else selected_model['description']
                    print(f"   \033[1;36mAbout:\033[0m {desc}")
                print(f"   \033[1;36mContext:\033[0m {selected_model['context_length']} tokens")
                print("\033[1;36m" + "─" * 50 + "\033[0m")
                return config
            else:
                print("❌ Invalid model number")
        else:
            print("❌ Invalid command")
        
        print("")

def get_model_recommendations(models):
    """Get recommended models based on use case"""
    recommendations = {
        "💬 Chat & Conversation": [],
        "📝 Writing & Content": [], 
        "🔍 Analysis & Reasoning": [],
        "💻 Coding & Technical": [],
        "🌐 Multilingual": []
    }
    
    for model in models:
        model_id = model['id'].lower()
        desc = (model.get('description') or '').lower()
        
        # Categorize models
        if any(word in model_id + desc for word in ['chat', 'conversation', 'instruct']):
            recommendations["💬 Chat & Conversation"].append(model)
        elif any(word in model_id + desc for word in ['write', 'content', 'creative', 'story']):
            recommendations["📝 Writing & Content"].append(model)
        elif any(word in model_id + desc for word in ['reason', 'analysis', 'logic', 'math']):
            recommendations["🔍 Analysis & Reasoning"].append(model)
        elif any(word in model_id + desc for word in ['code', 'program', 'technical', 'developer']):
            recommendations["💻 Coding & Technical"].append(model)
        elif any(word in model_id + desc for word in ['multilingual', 'translate', 'language']):
            recommendations["🌐 Multilingual"].append(model)
    
    return recommendations

def show_model_recommendations(config):
    """Show models by category with recommendations"""
    print("\033[1;36m")
    print("┌─────────────────────────────────────────────────────┐")
    print("│ 🎯 Model Recommendations │")
    print("└─────────────────────────────────────────────────────┘")
    print("\033[0m")
    
    models = get_available_models(config['api_key'])
    if not models:
        return config
    
    recommendations = get_model_recommendations(models)
    
    print("🤔 What will you primarily use the AI for?\n")
    
    categories = list(recommendations.keys())
    for i, category in enumerate(categories, 1):
        count = len(recommendations[category])
        if count > 0:
            print(f"\033[1;36m{i}. {category} ({count} models)\033[0m")
    
    print("\n\033[1;36m0. Show all models\033[0m")
    print("")
    
    choice = input("Select category (0-5): ").strip()
    
    if choice == '0':
        return interactive_model_browser(config)
    elif choice.isdigit() and 1 <= int(choice) <= len(categories):
        selected_category = categories[int(choice) - 1]
        category_models = recommendations[selected_category]
        
        print(f"\n\033[1;32m🎯 {selected_category} - Top Models:\033[0m\n")
        
        # Show top 3 models from this category
        for i, model in enumerate(category_models[:3], 1):
            print(f"\033[1;36m{i}. {model['id']}\033[0m")
            if model['description']:
                desc = model['description'][:80] + "..." if len(model['description']) > 80 else model['description']
                print(f"   \033[1;90m{desc}\033[0m")
            print(f"   \033[1;33mContext: {model['context_length']} tokens\033[0m")
            print("")
        
        if len(category_models) > 3:
            print(f"\033[1;90m... and {len(category_models) - 3} more models in this category\033[0m")
            print("")
        
        use_browser = input("Browse all models in this category? (y/N): ").strip().lower()
        if use_browser == 'y':
            # Create a modified config for the browser with filtered models
            temp_config = config.copy()
            return interactive_model_browser(temp_config)
        else:
            # Let user pick from top 3
            model_choice = input("Select model (1-3) or 'b' to browse all: ").strip().lower()
            if model_choice.isdigit() and 1 <= int(model_choice) <= 3:
                selected_model = category_models[int(model_choice) - 1]
                config['model'] = selected_model['id']
                save_model_history(selected_model['id'])
                print(f"✅ Selected: \033[1;32m{selected_model['id']}\033[0m")
                return config
            elif model_choice == 'b':
                return interactive_model_browser(config)
    
    return config

def change_model(config):
    """Enhanced model selection with multiple interfaces"""
    print("\033[1;36m")
    print("┌─────────────────────────────────────────────────────┐")
    print("│ 🤖 Model Selection │")
    print("└─────────────────────────────────────────────────────┘")
    print("\033[0m")
    
    # Show recently used models
    history = get_model_history()
    if history:
        print("\033[1;33m🕐 Recently Used Models:\033[0m")
        for i, model in enumerate(history[:3], 1):
            print(f"   \033[1;36m{i}. {model['id']}\033[0m")
        print("")
    
    print("Choose your model selection interface:\n")
    print("\033[1;36m1. 🎯 Smart Recommendations\033[0m")
    print("   - Get models tailored to your use case")
    print("   - Perfect for beginners")
    print("")
    print("\033[1;36m2. 🔍 Interactive Browser\033[0m") 
    print("   - Browse all models with search & filters")
    print("   - Great for power users")
    print("")
    print("\033[1;36m3. ⚡ Quick Pick\033[0m")
    print("   - Choose from reliable, tested models")
    print("   - Fast and simple")
    print("")
    print("\033[1;36m4. ↩️  Cancel\033[0m")
    print("")
    
    choice = input("Select interface (1-4): ").strip()
    
    if choice == '1':
        return show_model_recommendations(config)
    elif choice == '2':
        return interactive_model_browser(config)
    elif choice == '3':
        # Quick pick from reliable models
        reliable_models = get_reliable_free_models()
        print("\n\033[1;32m⚡ Reliable Models:\033[0m\n")
        for i, model in enumerate(reliable_models, 1):
            print(f"\033[1;36m{i}. {model}\033[0m")
        print("")
        model_choice = input("Select model (1-5): ").strip()
        if model_choice.isdigit() and 1 <= int(model_choice) <= 5:
            config['model'] = reliable_models[int(model_choice) - 1]
            save_model_history(config['model'])
            print(f"✅ Selected: \033[1;32m{config['model']}\033[0m")
        return config
    elif choice == '4':
        print("Model selection cancelled.")
        return config
    else:
        print("❌ Invalid choice")
        return config

def request_api_key():
    """Display API key request information"""
    print("\033[1;36m")
    print("┌─────────────────────────────────────────────────────┐")
    print("│ 🔑 Request API Key │")
    print("└─────────────────────────────────────────────────────┘")
    print("\033[0m")
    print("To use this application, you need an OpenRouter API key.")
    print("")
    print("\033[1;33m📋 Steps to get your API key:\033[0m")
    print("1. Visit: \033[1;34mhttps://openrouter.ai/keys\033[0m")
    print("2. Sign up or log in to your account")
    print("3. Create a new API key")
    print("4. Copy the key and enter it when prompted")
    print("")
    print("\033[1;32m💡 Tip: The API key typically starts with 'sk-' followed by a long string of characters\033[0m")
    print("")
    input("Press Enter to continue...")

def setup_wizard():
    """Interactive setup wizard for first-time configuration"""
    print("\033[1;36m")
    print("┌─────────────────────────────────────────────────────┐")
    print("│ 🔧 Setup Wizard │")
    print("└─────────────────────────────────────────────────────┘")
    print("\033[0m")
    print("Welcome to GitRocket-AI Terminal Assistant!")
    print("Let's get you set up...")
    print("")
    config = load_config()
    
    # API Key Setup with option to request one
    print("\033[1;33m📋 Step 1: API Key Setup\033[0m")
    print("You need an OpenRouter API key to use this application.")
    print("")
    
    while True:
        print("Do you already have an OpenRouter API key?")
        print("1. Yes, I have an API key")
        print("2. No, I need to get one")
        choice = input("Select option (1-2): ").strip()
        
        if choice == '1':
            # User has API key - proceed with input
            break
        elif choice == '2':
            # User needs API key - show request info
            request_api_key()
            print("")
            print("Let's continue with setup...")
            print("")
        else:
            print("❌ Invalid option. Please select 1 or 2.")
    
    # Get API Key
    while True:
        api_key = input("🔑 Enter your OpenRouter API key: ").strip()
        if api_key:
            # Test the API key by fetching models
            print("⏳ Testing API key...")
            models = get_available_models(api_key)
            if models:
                print("✅ API key validated successfully!")
                config['api_key'] = api_key
                break
            else:
                print("❌ Invalid API key or network error. Please try again.")
        else:
            print("❌ API key cannot be empty.")
    
    # Model Selection using enhanced interface
    print("")
    print("\033[1;33m📋 Step 2: Model Selection\033[0m")
    config = change_model(config)
    
    # Save configuration
    print("")
    print("⏳ Saving configuration...")
    if save_config(config):
        print("✅ Configuration saved successfully!")
        print(f"📁 Config location: {CONFIG_FILE}")
    else:
        print("❌ Failed to save configuration.")
    
    return config

def display_intro():
    """Display the GitRocket AI logo and intro"""
    # Center the ASCII art
    art_lines = [
        "@@@@@@@@@@@@@%*:  .           . .   . -#%@@@@@@@@@@",
        "@@@@@@@%*.     .                 .  ..   #@@@@@@@",
        "@@@@%*    . .   .         .   ..      .   . #@@@@",
        "@@%         .    ....::-----:::::.    .       :%@",
        "% .       . . .::-----::::::--::---:. .       . .",
        " . .       ..:---::::-:::::::::::----:      .  . ",
        ". .    .....::--:::....::.....::..:-=-:          ",
        "        :---::::::..........::::-::--=-..      . ",
        "   .   .-=====--:..:....:--=====-==---=-.  .... .",
        "    .   :=*****=:......--=+*******+-:--=-. .     ",
        "      ..:+*****+-.....:-****+++***+---===: .. .  ",
        "    .   -+****+=:......-======+***=---==+=.   .  ",
        "       .---==--:::....:...::--------==+++=.      ",
        "        :-==-----:..:-::...:-----===++=++=     . ",
        "      . .--=-==========-:...:-====+*==+*+:.      ",
        "         .---:-===--===-:::::---==+****+:        ",
        ".  .  .   :===-.......:::-====++===+***-. .      ",
        "   .     ..=+*=::...:--=+*********+***-   .    . ",
        "            -+*+=---++++************+:.          ",
        " .        . . .+*******************=  . .      . ",
        "   .    .       :+***************+-             .",
        "                  .=**+*******+-   .  ...      . ",
        "    .    .    .    .=***++++-:.  . .             ",
        "               .    .-+++---.      ..            ",
        "          ...   . . ...:::-- .  . ....:...:....  ",
        "  .         ..     ..:....==:.. .:...............",
        "          ..+***++**++--::==-..--...:====-:......",
        "      .. . .  :+*++======-===----:..:-==+*+==::..",
        "   . .      .. .======--::.:-====-::.:-=====+----"
    ]
    
    # Calculate center padding for the art
    max_art_width = max(len(line) for line in art_lines)
    art_padding = " " * ((80 - max_art_width) // 2)
    
    print("\033[1;92m")  # Bright green color
    for line in art_lines:
        print(art_padding + line)
    print("\033[0m")  # Reset color
    
    # Center the GitRocket AI text logo
    text_logo_lines = [
        "    ___ _ _                  _        _              _ ",
        "   / _ (_) |_ _ __ ___   ___| | _____| |_       __ _(_)",
        "  / /_\\/ | __| '__/ _ \\ / __| |/ / _ \\ __|____ / _` | |",
        " / /_\\\\| | |_| | | (_) | (__|   <  __/ ||_____| (_| | |",
        " \\____/|_|\\__|_|  \\___/ \\___|_|\\_\\___|\\__|     \\__,_|_|",
        "                                                      "
    ]
    
    # Calculate center padding for the text logo
    max_text_width = max(len(line) for line in text_logo_lines)
    text_padding = " " * ((80 - max_text_width) // 2)
    
    print("\033[1;92m")  # Bright green color
    for line in text_logo_lines:
        print(text_padding + line)
    print("\033[0m")
    
    print("")
    
    # Centered version text
    version_text = "🚀 GitRocket-AI Free Terminal Assistant v1.0 🚀"
    version_padding = " " * ((80 - len(version_text)) // 2)
    print(version_padding + "\033[1;95m" + version_text + "\033[0m")
    print("")
    
    # Centered description with border
    description = "🎯 Access 52+ FREE AI Models | 🛠️ Built-in Diagnostics | 🔧 API Debugger"
    border_line = "┌" + "─" * (len(description) + 2) + "┐"
    empty_line = "│" + " " * (len(description) + 2) + "│"
    text_line = "│ " + description + " │"
    padding = " " * ((80 - len(border_line)) // 2)
    
    print(padding + "\033[1;35m" + border_line + "\033[0m")
    print(padding + "\033[1;35m" + empty_line + "\033[0m")
    print(padding + "\033[1;35m" + text_line + "\033[0m")
    print(padding + "\033[1;35m" + empty_line + "\033[0m")
    print(padding + "\033[1;35m" + "└" + "─" * (len(description) + 2) + "┘" + "\033[0m")
    print("")

def display_chat_header(config):
    """Display chat session header"""
    print("\033[1;34m" + "─" * 80 + "\033[0m")
    print("\033[1;92m" + " " * 28 + "💬 Chat Session Started" + " " * 28 + "\033[0m")
    print(f"\033[1;37m" + " " * 25 + f"Model: {config['model']}" + " " * 25 + "\033[0m")
    print("\033[1;34m" + "─" * 80 + "\033[0m")
    print("")

def display_commands():
    """Display available commands"""
    print("\033[1;33m📋 Quick Commands:\033[0m")
    print(" • Type your message to chat")
    print(" • \033[1;31mquit\033[0m, \033[1;31mexit\033[0m, \033[1;31mbye\033[0m - End session")
    print(" • \033[1;36mclear\033[0m - Reset conversation")
    print(" • \033[1;35mhelp\033[0m - Show commands")
    print(" • \033[1;32msettings\033[0m - Change API settings")
    print(" • \033[1;33mtest\033[0m - Run API diagnostic test")
    print("")

def display_thinking():
    """Display thinking animation"""
    print("")
    print("\033[1;90m" + "🤔 Thinking" + "\033[0m", end="", flush=True)
    for i in range(3):
        time.sleep(0.4)
        print("\033[1;90m.\033[0m", end="", flush=True)
    print("")

def change_api_key(config):
    """Change the API key"""
    print("\033[1;36m")
    print("┌─────────────────────────────────────────────────────┐")
    print("│ 🔑 Change API Key │")
    print("└─────────────────────────────────────────────────────┘")
    print("\033[0m")
    while True:
        api_key = input("Enter your new OpenRouter API key: ").strip()
        if api_key:
            print("⏳ Testing API key...")
            models = get_available_models(api_key)
            if models:
                print("✅ API key validated successfully!")
                config['api_key'] = api_key
                return config
            else:
                print("❌ Invalid API key or network error. Please try again.")
        else:
            print("❌ API key cannot be empty.")

def api_debugger(config):
    """Comprehensive API Debugger"""
    print("\033[1;36m")
    print("┌─────────────────────────────────────────────────────┐")
    print("│ 🐛 API Debugger │")
    print("└─────────────────────────────────────────────────────┘")
    print("\033[0m")
    
    if not config['api_key']:
        print("❌ No API key configured")
        return
    
    print("🔍 Running diagnostic tests...")
    print("")
    
    # Test 1: Connectivity
    print("1. 📡 Network Connectivity...", end=" ")
    if check_connectivity():
        print("✅ Connected to OpenRouter")
    else:
        print("❌ Cannot reach OpenRouter")
        return
    
    # Test 2: API Key Format
    print("2. 🔑 API Key Format Check...", end=" ")
    if config['api_key'].startswith('sk-'):
        print("✅ Valid format (starts with 'sk-')")
    else:
        print("❌ Invalid format (should start with 'sk-')")
    
    # Test 3: Models Endpoint
    print("3. 📋 Models API Test...", end=" ")
    models = get_available_models(config['api_key'])
    if models:
        print(f"✅ Success! Found {len(models)} free models")
    else:
        print("❌ Failed to fetch models")
    
    # Test 4: Quick Chat Test
    print("4. 💬 Quick API Test...")
    if quick_api_test():
        print("   ✅ API is working correctly!")
    else:
        print("   ❌ API test failed")
    
    # Test 5: Configuration Check
    print("5. ⚙️ Configuration Check...", end=" ")
    issues = []
    if not config['api_key']:
        issues.append("Missing API key")
    if not config['model']:
        issues.append("Missing model")
    if config['max_tokens'] > 8000:
        issues.append("Max tokens too high")
    if not 0 <= config['temperature'] <= 2:
        issues.append("Temperature out of range (0-2)")
    
    if not issues:
        print("✅ All settings valid")
    else:
        print("❌ Issues found:")
        for issue in issues:
            print(f"   - {issue}")
    
    # Display Debug Info
    print("")
    print("\033[1;33m📊 Debug Information:\033[0m")
    print(f"   API URL: {config['api_url']}")
    print(f"   Model: {config['model']}")
    print(f"   Max Tokens: {config['max_tokens']}")
    print(f"   Temperature: {config['temperature']}")
    
    print("")
    input("Press Enter to continue...")

def view_debug_log(config):
    """View the debug log"""
    print("\033[1;36m")
    print("┌─────────────────────────────────────────────────────┐")
    print("│ 📋 Debug Log Viewer │")
    print("└─────────────────────────────────────────────────────┘")
    print("\033[0m")
    
    if not DEBUG_FILE.exists():
        print("No debug log found.")
        return
    
    try:
        with open(DEBUG_FILE, 'r') as f:
            log_data = json.load(f)
        
        print(f"Session started: {log_data.get('session_start', 'Unknown')}")
        print(f"Total requests: {len(log_data.get('requests', []))}")
        print(f"Total errors: {len(log_data.get('errors', []))}")
        print("")
        
        print("\033[1;33mRecent Events:\033[0m")
        events = log_data.get('requests', [])[-10:]  # Last 10 events
        for event in events:
            timestamp = event.get('timestamp', '')[:19]
            event_type = event.get('type', '')
            data = event.get('data', {})
            
            print(f"\n[{timestamp}] {event_type}:")
            if event_type == "api_request":
                print(f"   Model: {data.get('model', 'N/A')}")
                print(f"   Messages: {data.get('messages_count', 0)}")
            elif event_type == "api_response":
                status = data.get('status_code', 'N/A')
                status_display = f"✅ {status}" if status == 200 else f"❌ {status}"
                print(f"   Status: {status_display}")
                print(f"   Response Time: {data.get('response_time', 0):.2f}s")
            elif event_type in ["models_error", "api_error"]:
                print(f"   Error: {data.get('error', 'N/A')}")
    
    except Exception as e:
        print(f"❌ Error reading debug log: {e}")
    
    print("")
    input("Press Enter to continue...")

def clear_debug_log():
    """Clear the debug log"""
    print("\033[1;36m")
    print("┌─────────────────────────────────────────────────────┐")
    print("│ 🗑️ Clear Debug Log │")
    print("└─────────────────────────────────────────────────────┘")
    print("\033[0m")
    
    if not DEBUG_FILE.exists():
        print("No debug log to clear.")
        return
    
    try:
        confirm = input("Are you sure you want to clear the debug log? (y/N): ").strip().lower()
        if confirm == 'y':
            DEBUG_FILE.unlink()
            global debug_session
            debug_session = {
                "session_start": datetime.now().isoformat(),
                "requests": [],
                "errors": [],
                "config_changes": []
            }
            print("✅ Debug log cleared.")
        else:
            print("Clear cancelled.")
    except Exception as e:
        print(f"❌ Error clearing debug log: {e}")

def chat_with_ai(message, conversation_history, config):
    global last_request_time
    
    print("🔍 Debug: Starting API call...")
    
    # Add new message to conversation history
    conversation_history.append({"role": "user", "content": message})
    
    # Basic rate limiting
    current_time = time.time()
    time_since_last = current_time - last_request_time
    
    if time_since_last < MIN_REQUEST_INTERVAL:
        sleep_time = MIN_REQUEST_INTERVAL - time_since_last
        print(f"⏳ Waiting {sleep_time:.1f}s...")
        time.sleep(sleep_time)
    
    last_request_time = time.time()
    
    headers = {
        "Authorization": f"Bearer {config['api_key']}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/gitrocket-ai",
        "X-Title": "GitRocket-AI Terminal Chat"
    }
    
    # Conservative payload for free tier
    payload = {
        "model": config['model'],
        "messages": conversation_history[-3:],  # Only last 3 messages
        "max_tokens": min(config.get('max_tokens', 800), 500),  # Max 500 tokens
        "temperature": config.get('temperature', 0.7)
    }
    
    # Log the API request for debugging
    log_debug_event("api_request", {
        "headers": {k: "***" if "Authorization" in k else v for k, v in headers.items()},
        "payload": payload,
        "conversation_length": len(conversation_history)
    }, config)
    
    try:
        start_time = time.time()
        print("📤 Sending request to OpenRouter...")
        response = requests.post(config['api_url'], headers=headers, json=payload, timeout=60)
        response_time = time.time() - start_time
        
        print(f"📊 Received response: HTTP {response.status_code}")
        
        # Log the API response
        log_debug_event("api_response", {
            "status_code": response.status_code,
            "response_time": response_time,
            "headers": dict(response.headers)
        }, config)
        
        if response.status_code == 429:
            # Rate limited
            retry_after = response.headers.get('Retry-After', 30)
            try:
                retry_after = int(retry_after)
            except:
                retry_after = 30
                
            log_debug_event("api_error", {
                "error": f"Rate limited - retry after {retry_after}s", 
                "status_code": 429,
                "retry_after": retry_after
            }, config)
            
            # Remove the last user message since it failed
            conversation_history.pop()
            
            return f"⚠️ Rate limited. Please wait {retry_after} seconds.\n💡 Try running 'test' command to diagnose.", conversation_history
        
        elif response.status_code == 400:
            error_msg = response.json().get('error', {}).get('message', 'Bad request')
            log_debug_event("api_error", {
                "error": error_msg,
                "status_code": 400
            }, config)
            
            # Remove the last user message since it failed
            conversation_history.pop()
            
            return f"❌ API Error: {error_msg}", conversation_history
        
        response.raise_for_status()
        result = response.json()
        assistant_message = result['choices'][0]['message']['content']
        
        # Log successful response
        log_debug_event("api_success", {
            "content_length": len(assistant_message),
            "response_time": response_time
        }, config)
        
        # Add assistant response to conversation history
        conversation_history.append({"role": "assistant", "content": assistant_message})
        return assistant_message, conversation_history
        
    except requests.exceptions.Timeout:
        error_msg = "❌ Request timeout - API is very busy"
        log_debug_event("api_error", {"error": "Request timeout"}, config)
        # Remove the last user message since it failed
        conversation_history.pop()
        return error_msg, conversation_history
        
    except Exception as e:
        error_msg = f"❌ Error: {e}"
        log_debug_event("api_error", {"error": str(e)}, config)
        # Remove the last user message since it failed
        conversation_history.pop()
        return error_msg, conversation_history

def show_settings(config):
    """Show and update settings with a menu - Enhanced with debug options"""
    while True:
        print("\033[1;36m")
        print("┌─────────────────────────────────────────────────────┐")
        print("│ ⚙️ Settings Menu │")
        print("└─────────────────────────────────────────────────────┘")
        print("\033[0m")
        
        print("Current Settings:")
        print(f"1. Model: {config['model']}")
        print(f"2. API Key: {'*' * 20}{config['api_key'][-4:] if config['api_key'] else 'Not set'}")
        print(f"3. Max Tokens: {config['max_tokens']}")
        print(f"4. Temperature: {config['temperature']}")
        print(f"5. Debug Mode: {config.get('debug_mode', False)}")
        print(f"6. Debug Level: {config.get('debug_level', 'basic')}")
        print("")
        
        print("Options:")
        print("1. Change Model")
        print("2. Change API Key")
        print("3. Change Max Tokens")
        print("4. Change Temperature")
        print("5. Toggle Debug Mode")
        print("6. Change Debug Level")
        print("7. API Debugger")
        print("8. View Debug Log")
        print("9. Clear Debug Log")
        print("10. Switch to Reliable Model")
        print("11. Request API Key")
        print("12. Back to Chat")
        print("")
        
        choice = input("Select option (1-12): ").strip()
        
        if choice == '1':
            config = change_model(config)
        elif choice == '2':
            config = change_api_key(config)
        elif choice == '3':
            try:
                new_tokens = int(input(f"Enter new max tokens (current: {config['max_tokens']}): "))
                if 1 <= new_tokens <= 4000:
                    config['max_tokens'] = new_tokens
                    print(f"✅ Max tokens changed to: {new_tokens}")
                else:
                    print("❌ Max tokens must be between 1 and 4000 for free models.")
            except ValueError:
                print("❌ Please enter a valid number.")
        elif choice == '4':
            try:
                new_temp = float(input(f"Enter new temperature (current: {config['temperature']}): "))
                if 0 <= new_temp <= 2:
                    config['temperature'] = new_temp
                    print(f"✅ Temperature changed to: {new_temp}")
                else:
                    print("❌ Temperature must be between 0 and 2.")
            except ValueError:
                print("❌ Please enter a valid number.")
        elif choice == '5':
            config['debug_mode'] = not config.get('debug_mode', False)
            status = "enabled" if config['debug_mode'] else "disabled"
            print(f"✅ Debug mode {status}")
        elif choice == '6':
            print("Debug Levels:")
            print("1. basic - Minimal logging")
            print("2. detailed - More detailed logging")
            print("3. full - Complete request/response logging")
            level_choice = input("Select debug level (1-3): ").strip()
            levels = { '1': 'basic', '2': 'detailed', '3': 'full' }
            if level_choice in levels:
                config['debug_level'] = levels[level_choice]
                print(f"✅ Debug level set to: {config['debug_level']}")
            else:
                print("❌ Invalid choice")
        elif choice == '7':
            api_debugger(config)
        elif choice == '8':
            view_debug_log(config)
        elif choice == '9':
            clear_debug_log()
        elif choice == '10':
            config = switch_to_reliable_model(config)
        elif choice == '11':
            request_api_key()
        elif choice == '12':
            return config
        else:
            print("❌ Invalid option. Please select 1-12.")
        
        # Save after each change
        if save_config(config):
            print("✅ Settings saved!")
        else:
            print("❌ Failed to save settings.")
        print("")

def interactive_chat():
    global last_request_time, debug_session
    # Initialize the last request time to current time
    last_request_time = time.time()
    
    # Initialize debug session
    debug_session = {
        "session_start": datetime.now().isoformat(),
        "requests": [],
        "errors": [],
        "config_changes": []
    }
    
    # Load or setup configuration
    config = load_config()
    if not config['api_key']:
        print("🔧 First-time setup required...")
        config = setup_wizard()
        if not config['api_key']:
            print("❌ Setup failed. Please run the script again.")
            return
    
    # Display clean intro with the new logo
    display_intro()
    display_chat_header(config)
    
    # Show debug status
    if config.get('debug_mode', False):
        print("\033[1;33m🔧 Debug mode is ENABLED\033[0m")
        print("")
    
    display_commands()
    
    conversation_history = []
    
    while True:
        try:
            # Clean spacing
            print("")
            print("\033[1;34m" + "─" * 80 + "\033[0m")
            user_input = input("\033[1;36m💬 You: \033[0m").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("")
                print("\033[1;35m" + "─" * 80 + "\033[0m")
                print("\033[1;92m" + " " * 25 + "✨ Session Ended - Goodbye! 👋" + " " * 25 + "\033[0m")
                print("\033[1;35m" + "─" * 80 + "\033[0m")
                print("")
                break
            elif user_input.lower() == 'clear':
                conversation_history = []
                print("")
                print("\033[1;92m🔄 Conversation cleared!\033[0m")
                print("")
                continue
            elif user_input.lower() == 'help':
                print("")
                display_commands()
                continue
            elif user_input.lower() == 'test':
                print("")
                quick_api_test()
                continue
            elif user_input.lower() == 'settings':
                config = show_settings(config)
                # Redisplay the intro and header when returning from settings
                display_intro()
                display_chat_header(config)
                
                # Show debug status
                if config.get('debug_mode', False):
                    print("\033[1;33m🔧 Debug mode is ENABLED\033[0m")
                    print("")
                    
                display_commands()
                continue
            
            if not user_input:
                continue
            
            # Display thinking animation
            display_thinking()
            
            # Get AI response
            response, conversation_history = chat_with_ai(user_input, conversation_history, config)
            
            # Display AI response with clean formatting
            print("")
            print("\033[1;95m" + "─" * 80 + "\033[0m")
            print("\033[1;94m🤖 AI:\033[0m")
            print("")
            print("\033[0;97m" + response + "\033[0m")
            print("")
            print("\033[1;95m" + "─" * 80 + "\033[0m")
            
        except KeyboardInterrupt:
            print("")
            print("\033[1;35m" + "─" * 80 + "\033[0m")
            print("\033[1;92m" + " " * 30 + "✨ Session Ended ✨" + " " * 30 + "\033[0m")
            print("\033[1;35m" + "─" * 80 + "\033[0m")
            print("")
            break
        except Exception as e:
            print("")
            print(f"\033[1;91m❌ Error: {e}\033[0m")
            print("")

if __name__ == "__main__":
    interactive_chat()