#!/usr/bin/env python3

import requests
import json
import os
import readline
import time
from pathlib import Path
from datetime import datetime

# Try to import Groq SDK
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("⚠️  Groq SDK not installed. Run: pip install groq")

# Configuration
CONFIG_DIR = Path.home() / '.config' / 'gitrocket_ai'
CONFIG_FILE = CONFIG_DIR / 'config.json'
DEBUG_FILE = CONFIG_DIR / 'debug_log.json'
HISTORY_FILE = CONFIG_DIR / 'model_history.json'

# Provider Configuration
PROVIDERS = {
    "openrouter": {
        "name": "OpenRouter",
        "models_url": "https://openrouter.ai/api/v1/models",
        "api_url": "https://openrouter.ai/api/v1/chat/completions",
        "api_key_url": "https://openrouter.ai/keys",
        "free_suffix": ":free",
        "headers": {
            "Authorization": "Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/gitrocket-ai",
            "X-Title": "GitRocket-AI"
        }
    },
    "huggingface": {
        "name": "Hugging Face",
        "models_url": "https://huggingface.co/api/models?filter=text-generation-inference&sort=downloads",
        "api_url": "https://api-inference.huggingface.co/models/{model}",
        "api_key_url": "https://huggingface.co/settings/tokens",
        "free_suffix": "",
        "headers": {
            "Authorization": "Bearer {api_key}",
            "Content-Type": "application/json"
        },
        "popular_models": [
            "google/gemma-2-2b",
            "microsoft/DialoGPT-large",
            "microsoft/DialoGPT-medium",
            "microsoft/DialoGPT-small",
            "bert-base-uncased",
            "gpt2",
            "facebook/blenderbot-400M-distill",
            "microsoft/DialogRPT-updown"
        ]
    },
    "groq": {
        "name": "Groq",
        "models_url": "https://api.groq.com/openai/v1/models",
        "api_url": "https://api.groq.com/openai/v1/chat/completions",
        "api_key_url": "https://console.groq.com/keys",
        "free_suffix": "",
        "headers": {
            "Authorization": "Bearer {api_key}",
            "Content-Type": "application/json"
        },
        "free_models": [
            "llama2-70b-4096",
            "mixtral-8x7b-32768",
            "gemma-7b-it",
            "llama-3.1-8b-instant",
            "llama-3.2-1b-preview",
            "llama-3.2-3b-preview"
        ]
    }
}

# Default configuration
DEFAULT_CONFIG = {
    "api_key": "",
    "model": "deepseek/deepseek-r1-distill-qwen-7b:free",
    "api_url": "https://openrouter.ai/api/v1/chat/completions",
    "provider": "openrouter",
    "max_tokens": 800,
    "temperature": 0.7,
    "debug_mode": False,
    "debug_level": "basic",
    "provider_configs": {
        "openrouter": {"api_key": ""},
        "huggingface": {"api_key": ""},
        "groq": {"api_key": ""}
    }
}

# Rate limiting
last_request_time = 0
MIN_REQUEST_INTERVAL = 10

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
            merged_config = {**DEFAULT_CONFIG, **config}
            # Ensure provider_configs exists and has all providers
            if "provider_configs" not in merged_config:
                merged_config["provider_configs"] = DEFAULT_CONFIG["provider_configs"]
            else:
                for provider in DEFAULT_CONFIG["provider_configs"]:
                    if provider not in merged_config["provider_configs"]:
                        merged_config["provider_configs"][provider] = {"api_key": ""}
            return merged_config
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
        os.chmod(CONFIG_FILE, 0o600)
        return True
    except Exception as e:
        print(f"❌ Error saving config: {e}")
        return False

def get_current_provider_config(config):
    """Get current provider configuration"""
    provider = config.get('provider', 'openrouter')
    provider_config = config.get('provider_configs', {}).get(provider, {})
    return provider, provider_config

def get_current_api_key(config):
    """Get API key for current provider"""
    provider, provider_config = get_current_provider_config(config)
    return provider_config.get('api_key', '')

def set_current_api_key(config, api_key):
    """Set API key for current provider"""
    provider = config.get('provider', 'openrouter')
    if 'provider_configs' not in config:
        config['provider_configs'] = {}
    if provider not in config['provider_configs']:
        config['provider_configs'][provider] = {}
    config['provider_configs'][provider]['api_key'] = api_key
    return config

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

def check_connectivity(provider=None):
    """Check if we can reach the provider"""
    if provider is None:
        provider = 'openrouter'
    
    provider_info = PROVIDERS[provider]
    print(f"🔍 Checking connectivity to {provider_info['name']}...")
    
    test_urls = {
        'openrouter': 'https://openrouter.ai',
        'huggingface': 'https://huggingface.co',
        'groq': 'https://groq.com'
    }
    
    try:
        response = requests.get(test_urls[provider], timeout=10)
        print(f"📡 Connectivity: ✅ (Status {response.status_code})")
        return True
    except Exception as e:
        print(f"❌ Cannot reach {provider_info['name']}: {e}")
        return False

def quick_api_test():
    """Simple direct test of the current provider API"""
    config = load_config()
    provider = config.get('provider', 'openrouter')
    api_key = get_current_api_key(config)
    
    if not api_key:
        print("❌ No API key configured for current provider")
        return
    
    print(f"\n🧪 Running direct API test for {PROVIDERS[provider]['name']}...")
    
    if provider == 'groq' and GROQ_AVAILABLE:
        # Test Groq using their SDK
        try:
            client = Groq(api_key=api_key)
            completion = client.chat.completions.create(
                model="llama2-70b-4096",
                messages=[{"role": "user", "content": "Say 'TEST OK' only."}],
                temperature=0.1,
                max_tokens=10
            )
            content = completion.choices[0].message.content
            print("✅ SUCCESS! Response:")
            print(f"   '{content}'")
            return True
        except Exception as e:
            print(f"❌ Groq API test failed: {e}")
            return False
    else:
        # Test other providers using requests
        headers = get_headers(provider, api_key)
        
        # Test with provider-specific payload
        test_payloads = {
            'openrouter': {
                "model": "google/gemma-2-9b-it:free",
                "messages": [{"role": "user", "content": "Say 'TEST OK' only."}],
                "max_tokens": 10,
                "temperature": 0.1
            },
            'huggingface': {
                "inputs": "Say 'TEST OK' only.",
                "parameters": {
                    "max_new_tokens": 10,
                    "temperature": 0.1,
                    "return_full_text": False
                }
            },
            'groq': {
                "model": "llama2-70b-4096",
                "messages": [{"role": "user", "content": "Say 'TEST OK' only."}],
                "max_tokens": 10,
                "temperature": 0.1
            }
        }
        
        try:
            print("📤 Sending test request...")
            
            if provider == 'huggingface':
                # For Hugging Face, we need to use a specific model in the URL
                test_model = "microsoft/DialoGPT-small"
                api_url = PROVIDERS[provider]['api_url'].format(model=test_model)
                payload = test_payloads[provider]
            else:
                api_url = PROVIDERS[provider]['api_url']
                payload = test_payloads[provider]
            
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            print(f"📊 Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                if provider == 'huggingface':
                    content = result[0]['generated_text']
                else:
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

def get_headers(provider, api_key):
    """Get headers for the specified provider"""
    headers_template = PROVIDERS[provider]['headers']
    headers = {}
    for key, value in headers_template.items():
        headers[key] = value.format(api_key=api_key)
    return headers

def get_detailed_model_description(model_id, provider):
    """Get detailed descriptions for models"""
    descriptions = {
        # OpenRouter Models
        "google/gemma-2-9b-it:free": "Gemma 2 9B Instruct is Google's lightweight, state-of-the-art open model built for text generation tasks. It excels at following instructions, answering questions, and engaging in conversational AI. With 9 billion parameters, it offers a great balance between performance and efficiency, making it ideal for various text-based applications.",
        
        "microsoft/wizardlm-2-8x22b:free": "WizardLM 2 8x22B is a massive language model with 176 billion parameters using mixture-of-experts architecture. It demonstrates exceptional performance in complex reasoning, coding tasks, and mathematical problem-solving. The model is particularly strong at following complex instructions and providing detailed, accurate responses.",
        
        "meta-llama/llama-3.1-8b-instruct:free": "Llama 3.1 8B Instruct is Meta's latest 8-billion parameter model optimized for instruction following. It features improved reasoning capabilities, better multilingual support, and enhanced safety features. Excellent for general chat, coding assistance, and creative writing tasks.",
        
        "mistralai/mistral-nemo:free": "Mistral Nemo is a versatile model designed for both chat and instruction following. It combines Mistral's efficient architecture with strong performance across various benchmarks. Particularly good at creative tasks, summarization, and general conversation.",
        
        "cognitivecomputations/dolphin3.0-mistral-24b:free": "Dolphin 3.0 Mistral 24B is an uncensored model fine-tuned for maximum helpfulness. It's designed to be less restrictive while maintaining quality, making it excellent for creative writing, roleplaying, and unfiltered conversations.",
        
        # Groq Models
        "llama2-70b-4096": "Llama 2 70B running on Groq's lightning-fast LPU inference engine. This model delivers exceptional performance with 70 billion parameters, optimized for complex reasoning, coding, and detailed conversations. Groq's hardware acceleration makes it incredibly responsive.",
        
        "mixtral-8x7b-32768": "Mixtral 8x7B MoE model with 32K context length on Groq's platform. This mixture-of-experts model offers excellent performance across diverse tasks including coding, writing, and analysis. The large context window allows for extensive conversations and document processing.",
        
        "gemma-7b-it": "Google's Gemma 7B Instruct model optimized for Groq's hardware. A lightweight but capable model perfect for quick responses, chat applications, and general AI assistance with fast inference times.",
        
        "llama-3.1-8b-instant": "Meta's Llama 3.1 8B model running on Groq for instant responses. Balanced performance with quick inference, ideal for real-time applications and responsive chat interfaces.",
        
        "llama-3.2-1b-preview": "Ultra-fast 1B parameter model from Meta's Llama 3.2 series. Designed for maximum speed and efficiency while maintaining reasonable quality for simple tasks and quick interactions.",
        
        "llama-3.2-3b-preview": "Lightweight 3B parameter model offering a great balance between speed and capability. Perfect for applications requiring both responsiveness and decent reasoning abilities.",
        
        # Hugging Face Models
        "microsoft/DialoGPT-large": "Microsoft's large-scale conversational AI model trained on Reddit discussions. Excellent for engaging in natural, multi-turn conversations with human-like responses and good contextual understanding.",
        
        "microsoft/DialoGPT-medium": "Medium-sized version of DialoGPT, balancing performance and efficiency. Well-suited for chat applications requiring coherent and contextually appropriate responses.",
        
        "microsoft/DialoGPT-small": "Compact version of DialoGPT for fast inference. Ideal for applications where response speed is crucial while maintaining reasonable conversation quality.",
        
        "facebook/blenderbot-400M-distill": "Facebook's distilled BlenderBot model optimized for engaging conversations. Trained on a diverse dataset to handle various topics while being computationally efficient.",
        
        "google/gemma-2-2b": "Google's ultra-lightweight Gemma 2 model with 2 billion parameters. Perfect for resource-constrained environments while still offering decent performance for basic text generation tasks.",
        
        "bert-base-uncased": "Classic BERT model for understanding and generating text. While primarily a encoder model, it can be used for various NLP tasks including question answering and text classification.",
        
        "gpt2": "The original GPT-2 model from OpenAI. A foundational transformer model that started the modern LLM revolution, capable of coherent text generation and creative writing.",
        
        "microsoft/DialogRPT-updown": "DialogRPT model trained to predict upvotes/downvotes for responses. Useful for generating engaging and well-received conversational responses."
    }
    
    # Return the specific description or a generic one
    if model_id in descriptions:
        return descriptions[model_id]
    else:
        # Generate a generic description based on model characteristics
        if 'llama' in model_id.lower():
            return f"{model_id} is part of Meta's Llama family of language models, known for strong performance across various tasks including conversation, coding, and reasoning. This model offers a balance of capability and efficiency."
        elif 'gemma' in model_id.lower():
            return f"{model_id} is from Google's Gemma family of lightweight, open language models. It's designed to be efficient while maintaining strong performance for instruction following and text generation tasks."
        elif 'mistral' in model_id.lower():
            return f"{model_id} utilizes Mistral AI's efficient architecture, known for delivering strong performance with optimized resource usage. Excellent for various natural language processing tasks."
        elif 'gpt' in model_id.lower():
            return f"{model_id} is based on the GPT architecture, capable of generating coherent and contextually relevant text across a wide range of topics and applications."
        else:
            return f"{model_id} is a capable language model suitable for various text generation and understanding tasks. It can handle conversation, content creation, and information processing effectively."

def get_available_models(provider, api_key):
    """Fetch available models from the specified provider"""
    try:
        headers = get_headers(provider, api_key)
        models_url = PROVIDERS[provider]['models_url']
        
        start_time = time.time()
        
        if provider == 'huggingface':
            # Hugging Face has a different API structure
            response = requests.get(models_url, timeout=15)
            models_data = response.json()
            
            # Filter for popular text generation models
            models = []
            for model in models_data:
                model_id = model.get('id', '')
                # Include popular models and some free ones
                if (model_id in PROVIDERS[provider]['popular_models'] or 
                    any(keyword in model_id.lower() for keyword in ['gpt', 'dialo', 'blender', 'bert'])):
                    models.append({
                        'id': model_id,
                        'name': model_id.split('/')[-1],
                        'description': get_detailed_model_description(model_id, provider),
                        'context_length': 2048,  # Default
                        'pricing': {'prompt': 0, 'completion': 0}
                    })
            return models[:50]  # Limit to top 50
            
        elif provider == 'groq':
            # Groq uses OpenAI-compatible API
            models = []
            for model_id in PROVIDERS[provider]['free_models']:
                models.append({
                    'id': model_id,
                    'name': model_id,
                    'description': get_detailed_model_description(model_id, provider),
                    'context_length': 4096 if '32768' in model_id else 8192 if '4096' in model_id else 2048,
                    'pricing': {'prompt': 0, 'completion': 0}
                })
            return models
                
        else:  # openrouter
            response = requests.get(models_url, headers=headers, timeout=10)
            if response.status_code == 200:
                models_data = response.json()
                models = []
                for model in models_data.get('data', []):
                    model_id = model.get('id', '')
                    if model_id.endswith(PROVIDERS[provider]['free_suffix']):
                        models.append({
                            'id': model_id,
                            'name': model.get('name', model_id),
                            'description': get_detailed_model_description(model_id, provider),
                            'context_length': model.get('context_length', 'Unknown'),
                            'pricing': model.get('pricing', {})
                        })
                return models
            else:
                print(f"❌ Error fetching {provider} models: {response.status_code}")
                return []
                
    except Exception as e:
        print(f"❌ Error fetching {provider} models: {e}")
        return []

def get_reliable_free_models(provider):
    """Return a list of known reliable free models for the provider"""
    reliable_models = {
        'openrouter': [
            "google/gemma-2-9b-it:free",
            "microsoft/wizardlm-2-8x22b:free", 
            "meta-llama/llama-3.1-8b-instruct:free",
            "mistralai/mistral-nemo:free",
            "cognitivecomputations/dolphin3.0-mistral-24b:free"
        ],
        'huggingface': [
            "microsoft/DialoGPT-medium",
            "microsoft/DialoGPT-small",
            "facebook/blenderbot-400M-distill",
            "google/gemma-2-2b"
        ],
        'groq': [
            "llama2-70b-4096",
            "mixtral-8x7b-32768",
            "gemma-7b-it",
            "llama-3.1-8b-instant"
        ]
    }
    return reliable_models.get(provider, [])

def switch_to_reliable_model(config):
    """Switch to a known reliable model for current provider"""
    provider = config.get('provider', 'openrouter')
    reliable_models = get_reliable_free_models(provider)
    current_model = config['model']
    
    if current_model in reliable_models:
        # Try a different one
        for model in reliable_models:
            if model != current_model:
                config['model'] = model
                print(f"🔄 Switching to reliable model: {model}")
                return config
    
    # If all else fails, use the first reliable model
    if reliable_models:
        config['model'] = reliable_models[0]
        print(f"🔄 Switching to reliable model: {reliable_models[0]}")
    return config

def display_model_info(model, index, provider):
    """Display model information with appealing formatting"""
    model_id = model['id']
    context = model['context_length']
    description = model['description']
    
    # Color code based on provider
    color_codes = {
        'openrouter': "\033[1;34m",   # Blue
        'huggingface': "\033[1;33m",  # Yellow
        'groq': "\033[1;35m"          # Magenta
    }
    color = color_codes.get(provider, "\033[1;36m")
    
    print(f"{color}┌─ {index}. {model_id}\033[0m")
    print(f"\033[1;90m│   Provider: {PROVIDERS[provider]['name']}\033[0m")
    print(f"\033[1;90m│   Context: {context} tokens\033[0m")
    
    # Format description with word wrapping
    desc_lines = []
    words = description.split()
    current_line = ""
    for word in words:
        if len(current_line + " " + word) <= 70:  # Slightly narrower for better readability
            current_line += " " + word if current_line else word
        else:
            desc_lines.append(current_line)
            current_line = word
    if current_line:
        desc_lines.append(current_line)
    
    if desc_lines:
        print(f"\033[1;97m│   Description:\033[0m")
        for i, line in enumerate(desc_lines):
            if i == 0:
                print(f"\033[1;97m│     {line}\033[0m")
            else:
                print(f"\033[1;97m│     {line}\033[0m")
    
    # Show pricing info
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
    """Interactive model browser with provider selection"""
    print("\033[1;36m")
    print("┌─────────────────────────────────────────────────────┐")
    print("│ 🔍 Interactive Model Browser │")
    print("└─────────────────────────────────────────────────────┘")
    print("\033[0m")
    
    # Provider selection
    print("Select AI Provider:\n")
    providers = list(PROVIDERS.keys())
    for i, provider in enumerate(providers, 1):
        print(f"\033[1;36m{i}. {PROVIDERS[provider]['name']}\033[0m")
    print("")
    
    provider_choice = input("Select provider (1-3): ").strip()
    if not provider_choice.isdigit() or not (1 <= int(provider_choice) <= 3):
        print("❌ Invalid provider selection")
        return config
    
    selected_provider = providers[int(provider_choice) - 1]
    api_key = config.get('provider_configs', {}).get(selected_provider, {}).get('api_key', '')
    
    if not api_key:
        print(f"❌ No API key configured for {PROVIDERS[selected_provider]['name']}")
        print(f"💡 Please set up the API key in settings first")
        return config
    
    print(f"⏳ Loading available models from {PROVIDERS[selected_provider]['name']}...")
    models = get_available_models(selected_provider, api_key)
    if not models:
        print("❌ Could not fetch models.")
        return config
    
    # Continue with the existing browser logic but for the selected provider
    filtered_models = models
    current_page = 0
    models_per_page = 5
    
    while True:
        print("\033[1;36m")
        print("┌─────────────────────────────────────────────────────┐")
        print(f"│ 📋 {PROVIDERS[selected_provider]['name']} Models ({len(filtered_models)} found) - Page {current_page + 1} │")
        print("└─────────────────────────────────────────────────────┘")
        print("\033[0m")
        
        # Display current page
        start_idx = current_page * models_per_page
        end_idx = min(start_idx + models_per_page, len(filtered_models))
        
        for i in range(start_idx, end_idx):
            display_model_info(filtered_models[i], i + 1, selected_provider)
        
        # Show navigation info
        print(f"\033[1;33mShowing {start_idx + 1}-{end_idx} of {len(filtered_models)} models\033[0m")
        print("")
        
        # Show commands
        print("\033[1;94mNavigation:\033[0m")
        print(" • \033[1;36m[number]\033[0m - Select model")
        print(" • \033[1;36mn\033[0m - Next page")
        print(" • \033[1;36mp\033[0m - Previous page") 
        print(" • \033[1;36ms [term]\033[0m - Search models")
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
        elif choice.isdigit():
            model_index = int(choice) - 1
            if 0 <= model_index < len(filtered_models):
                selected_model = filtered_models[model_index]
                config['model'] = selected_model['id']
                config['provider'] = selected_provider
                config['api_url'] = PROVIDERS[selected_provider]['api_url']
                if selected_provider == 'huggingface':
                    config['api_url'] = config['api_url'].format(model=selected_model['id'])
                
                save_model_history(selected_model['id'])
                print(f"✅ Selected: \033[1;32m{selected_model['id']}\033[0m")
                print(f"✅ Provider: \033[1;32m{PROVIDERS[selected_provider]['name']}\033[0m")
                
                return config
            else:
                print("❌ Invalid model number")
        else:
            print("❌ Invalid command")
        
        print("")

def change_provider(config):
    """Change the current AI provider"""
    print("\033[1;36m")
    print("┌─────────────────────────────────────────────────────┐")
    print("│ 🔄 Change AI Provider │")
    print("└─────────────────────────────────────────────────────┘")
    print("\033[0m")
    
    print("Select AI Provider:\n")
    providers = list(PROVIDERS.keys())
    for i, provider in enumerate(providers, 1):
        provider_name = PROVIDERS[provider]['name']
        has_key = bool(config.get('provider_configs', {}).get(provider, {}).get('api_key', ''))
        status = "✅ Configured" if has_key else "❌ Needs API Key"
        print(f"\033[1;36m{i}. {provider_name} - {status}\033[0m")
        print(f"   {PROVIDERS[provider]['api_key_url']}")
    print("")
    
    choice = input("Select provider (1-3): ").strip()
    if not choice.isdigit() or not (1 <= int(choice) <= 3):
        print("❌ Invalid provider selection")
        return config
    
    selected_provider = providers[int(choice) - 1]
    api_key = config.get('provider_configs', {}).get(selected_provider, {}).get('api_key', '')
    
    if not api_key:
        print(f"❌ No API key configured for {PROVIDERS[selected_provider]['name']}")
        setup_api_key = input("Would you like to set it up now? (y/N): ").strip().lower()
        if setup_api_key == 'y':
            config = setup_provider_api_key(config, selected_provider)
        else:
            return config
    
    # Set the provider and update model to a reliable one
    config['provider'] = selected_provider
    config['api_url'] = PROVIDERS[selected_provider]['api_url']
    
    # Set a reliable model for the provider
    reliable_models = get_reliable_free_models(selected_provider)
    if reliable_models:
        config['model'] = reliable_models[0]
        if selected_provider == 'huggingface':
            config['api_url'] = config['api_url'].format(model=reliable_models[0])
    
    print(f"✅ Switched to {PROVIDERS[selected_provider]['name']}")
    print(f"✅ Model: {config['model']}")
    
    return config

def setup_provider_api_key(config, provider):
    """Set up API key for a specific provider"""
    print(f"\n🔑 Setting up {PROVIDERS[provider]['name']} API Key")
    print(f"📋 Get your API key from: {PROVIDERS[provider]['api_key_url']}")
    print("")
    
    while True:
        api_key = input(f"Enter your {PROVIDERS[provider]['name']} API key: ").strip()
        if api_key:
            print("⏳ Testing API key...")
            
            if provider == 'groq' and GROQ_AVAILABLE:
                # Test Groq using SDK
                try:
                    client = Groq(api_key=api_key)
                    # Try to list models or make a simple call
                    models = client.models.list()
                    print("✅ API key validated successfully!")
                    if 'provider_configs' not in config:
                        config['provider_configs'] = {}
                    config['provider_configs'][provider] = {'api_key': api_key}
                    return config
                except Exception as e:
                    print(f"❌ Invalid Groq API key: {e}")
            else:
                # Test other providers
                models = get_available_models(provider, api_key)
                if models or provider == 'huggingface':  # Hugging Face might not return models immediately
                    print("✅ API key validated successfully!")
                    if 'provider_configs' not in config:
                        config['provider_configs'] = {}
                    config['provider_configs'][provider] = {'api_key': api_key}
                    return config
                else:
                    print("❌ Invalid API key or network error. Please try again.")
        else:
            print("❌ API key cannot be empty.")

def get_model_recommendations(models, provider):
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
        if any(word in model_id + desc for word in ['chat', 'conversation', 'instruct', 'dialo', 'blender']):
            recommendations["💬 Chat & Conversation"].append(model)
        elif any(word in model_id + desc for word in ['write', 'content', 'creative', 'story']):
            recommendations["📝 Writing & Content"].append(model)
        elif any(word in model_id + desc for word in ['reason', 'analysis', 'logic', 'math', 'wizard']):
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
    
    provider = config.get('provider', 'openrouter')
    api_key = get_current_api_key(config)
    models = get_available_models(provider, api_key)
    if not models:
        return config
    
    recommendations = get_model_recommendations(models, provider)
    
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
                desc = model['description'][:100] + "..." if len(model['description']) > 100 else model['description']
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
        provider = config.get('provider', 'openrouter')
        reliable_models = get_reliable_free_models(provider)
        print(f"\n\033[1;32m⚡ Reliable {PROVIDERS[provider]['name']} Models:\033[0m\n")
        for i, model in enumerate(reliable_models, 1):
            print(f"\033[1;36m{i}. {model}\033[0m")
        print("")
        model_choice = input(f"Select model (1-{len(reliable_models)}): ").strip()
        if model_choice.isdigit() and 1 <= int(model_choice) <= len(reliable_models):
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

def request_api_key(provider='openrouter'):
    """Display API key request information for specific provider"""
    provider_info = PROVIDERS[provider]
    print(f"\033[1;36m")
    print("┌─────────────────────────────────────────────────────┐")
    print(f"│ 🔑 Request {provider_info['name']} API Key │")
    print("└─────────────────────────────────────────────────────┘")
    print("\033[0m")
    print(f"To use {provider_info['name']}, you need an API key.")
    print("")
    print("\033[1;33m📋 Steps to get your API key:\033[0m")
    print(f"1. Visit: \033[1;34m{provider_info['api_key_url']}\033[0m")
    print("2. Sign up or log in to your account")
    print("3. Create a new API key")
    print("4. Copy the key and enter it when prompted")
    print("")
    
    if provider == 'huggingface':
        print("💡 For Hugging Face, you might not need an API key for some models,")
        print("   but having one gives you higher rate limits.")
    elif provider == 'groq':
        print("💡 Groq offers free tier with rate limits.")
    
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
    print("Now supporting multiple AI providers!")
    print("")
    config = load_config()
    
    # Provider Selection
    print("\033[1;33m📋 Step 1: Select AI Provider\033[0m")
    print("Choose which AI service you want to use:\n")
    
    providers = list(PROVIDERS.keys())
    for i, provider in enumerate(providers, 1):
        print(f"\033[1;36m{i}. {PROVIDERS[provider]['name']}\033[0m")
        if provider == 'openrouter':
            print("   - Access to 100+ models including free ones")
        elif provider == 'huggingface':
            print("   - Direct access to open-source models")
        elif provider == 'groq':
            print("   - Ultra-fast inference with free tier")
        print("")
    
    provider_choice = input("Select provider (1-3): ").strip()
    if not provider_choice.isdigit() or not (1 <= int(provider_choice) <= 3):
        print("❌ Invalid provider selection, using OpenRouter as default")
        selected_provider = 'openrouter'
    else:
        selected_provider = providers[int(provider_choice) - 1]
    
    config['provider'] = selected_provider
    
    # API Key Setup
    print(f"\n\033[1;33m📋 Step 2: {PROVIDERS[selected_provider]['name']} API Key Setup\033[0m")
    
    while True:
        print(f"Do you already have a {PROVIDERS[selected_provider]['name']} API key?")
        print("1. Yes, I have an API key")
        print("2. No, I need to get one")
        choice = input("Select option (1-2): ").strip()
        
        if choice == '1':
            break
        elif choice == '2':
            request_api_key(selected_provider)
            print("")
            print("Let's continue with setup...")
            print("")
        else:
            print("❌ Invalid option. Please select 1 or 2.")
    
    # Get API Key
    config = setup_provider_api_key(config, selected_provider)
    if not config:
        return config
    
    # Model Selection
    print("")
    print("\033[1;33m📋 Step 3: Model Selection\033[0m")
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
    """Display chat session header with provider info"""
    provider = config.get('provider', 'openrouter')
    print("\033[1;34m" + "─" * 80 + "\033[0m")
    print("\033[1;92m" + " " * 28 + "💬 Chat Session Started" + " " * 28 + "\033[0m")
    print(f"\033[1;37m" + " " * 25 + f"Provider: {PROVIDERS[provider]['name']}" + " " * 25 + "\033[0m")
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
    """Change the API key for current provider"""
    provider = config.get('provider', 'openrouter')
    print(f"\033[1;36m")
    print("┌─────────────────────────────────────────────────────┐")
    print(f"│ 🔑 Change {PROVIDERS[provider]['name']} API Key │")
    print("└─────────────────────────────────────────────────────┘")
    print("\033[0m")
    
    return setup_provider_api_key(config, provider)

def api_debugger(config):
    """Comprehensive API Debugger for current provider"""
    provider = config.get('provider', 'openrouter')
    print(f"\033[1;36m")
    print("┌─────────────────────────────────────────────────────┐")
    print(f"│ 🐛 {PROVIDERS[provider]['name']} API Debugger │")
    print("└─────────────────────────────────────────────────────┘")
    print("\033[0m")
    
    api_key = get_current_api_key(config)
    if not api_key:
        print("❌ No API key configured for current provider")
        return
    
    print("🔍 Running diagnostic tests...")
    print("")
    
    # Test 1: Connectivity
    print(f"1. 📡 Network Connectivity...", end=" ")
    if check_connectivity(provider):
        print(f"✅ Connected to {PROVIDERS[provider]['name']}")
    else:
        print(f"❌ Cannot reach {PROVIDERS[provider]['name']}")
        return
    
    # Test 2: API Key
    print(f"2. 🔑 API Key Check...", end=" ")
    if provider == 'groq' and GROQ_AVAILABLE:
        try:
            client = Groq(api_key=api_key)
            models = client.models.list()
            print(f"✅ API key is valid")
        except Exception as e:
            print(f"❌ API key validation failed: {e}")
    else:
        models = get_available_models(provider, api_key)
        if models or provider == 'huggingface':  # Hugging Face might be lenient
            print(f"✅ API key is valid")
        else:
            print(f"❌ API key validation failed")
    
    # Test 3: Quick Chat Test
    print(f"3. 💬 Quick API Test...")
    if quick_api_test():
        print("   ✅ API is working correctly!")
    else:
        print("   ❌ API test failed")
    
    # Display current configuration
    print("")
    print("\033[1;33m📊 Current Configuration:\033[0m")
    print(f"   Provider: {PROVIDERS[provider]['name']}")
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
    
    provider = config.get('provider', 'openrouter')
    api_key = get_current_api_key(config)
    
    print(f"🔍 Debug: Starting API call to {PROVIDERS[provider]['name']}...")
    
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
    
    # Log the API request for debugging
    log_debug_event("api_request", {
        "provider": provider,
        "conversation_length": len(conversation_history),
        "message": message[:100] + "..." if len(message) > 100 else message
    }, config)
    
    try:
        start_time = time.time()
        
        if provider == 'groq' and GROQ_AVAILABLE:
            # Use Groq SDK
            print(f"📤 Sending request to {PROVIDERS[provider]['name']}...")
            client = Groq(api_key=api_key)
            
            completion = client.chat.completions.create(
                model=config['model'],
                messages=conversation_history[-3:],  # Only last 3 messages
                temperature=config.get('temperature', 0.7),
                max_tokens=min(config.get('max_tokens', 800), 2000),
                stream=False
            )
            
            response_time = time.time() - start_time
            assistant_message = completion.choices[0].message.content
            
        else:
            # Use requests for other providers
            headers = get_headers(provider, api_key)
            
            # Provider-specific payload
            if provider == 'huggingface':
                # Hugging Face uses a different format
                payload = {
                    "inputs": message,
                    "parameters": {
                        "max_new_tokens": min(config.get('max_tokens', 800), 500),
                        "temperature": config.get('temperature', 0.7),
                        "return_full_text": False
                    }
                }
                api_url = config['api_url']
            else:
                # OpenRouter and Groq (fallback) use OpenAI-compatible format
                payload = {
                    "model": config['model'],
                    "messages": conversation_history[-3:],  # Only last 3 messages
                    "max_tokens": min(config.get('max_tokens', 800), 500),
                    "temperature": config.get('temperature', 0.7)
                }
                api_url = config['api_url']
            
            print(f"📤 Sending request to {PROVIDERS[provider]['name']}...")
            response = requests.post(api_url, headers=headers, json=payload, timeout=60)
            response_time = time.time() - start_time
            
            print(f"📊 Received response: HTTP {response.status_code}")
            
            # Log the API response
            log_debug_event("api_response", {
                "provider": provider,
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
            
            # Extract response based on provider
            if provider == 'huggingface':
                assistant_message = result[0]['generated_text']
            else:
                assistant_message = result['choices'][0]['message']['content']
        
        # Log successful response
        log_debug_event("api_success", {
            "provider": provider,
            "content_length": len(assistant_message),
            "response_time": response_time
        }, config)
        
        # Add assistant response to conversation history
        conversation_history.append({"role": "assistant", "content": assistant_message})
        return assistant_message, conversation_history
        
    except requests.exceptions.Timeout:
        error_msg = "❌ Request timeout - API is very busy"
        log_debug_event("api_error", {"error": "Request timeout", "provider": provider}, config)
        # Remove the last user message since it failed
        conversation_history.pop()
        return error_msg, conversation_history
        
    except Exception as e:
        error_msg = f"❌ Error: {e}"
        log_debug_event("api_error", {"error": str(e), "provider": provider}, config)
        # Remove the last user message since it failed
        conversation_history.pop()
        return error_msg, conversation_history

def setup_all_provider_keys(config):
    """Set up API keys for all providers"""
    print("\033[1;36m")
    print("┌─────────────────────────────────────────────────────┐")
    print("│ 🔑 Setup All Provider API Keys │")
    print("└─────────────────────────────────────────────────────┘")
    print("\033[0m")
    
    for provider in PROVIDERS:
        print(f"\n📋 Setting up {PROVIDERS[provider]['name']}...")
        current_key = config.get('provider_configs', {}).get(provider, {}).get('api_key', '')
        
        if current_key:
            change = input(f"{PROVIDERS[provider]['name']} API key already configured. Change it? (y/N): ").strip().lower()
            if change != 'y':
                continue
        
        config = setup_provider_api_key(config, provider)
    
    return config

def show_settings(config):
    """Show and update settings with a menu - Enhanced with multi-provider support"""
    while True:
        provider = config.get('provider', 'openrouter')
        print("\033[1;36m")
        print("┌─────────────────────────────────────────────────────┐")
        print("│ ⚙️ Settings Menu │")
        print("└─────────────────────────────────────────────────────┘")
        print("\033[0m")
        
        print("Current Settings:")
        print(f"1. Provider: {PROVIDERS[provider]['name']}")
        print(f"2. Model: {config['model']}")
        print(f"3. API Key: {'*' * 20}{get_current_api_key(config)[-4:] if get_current_api_key(config) else 'Not set'}")
        print(f"4. Max Tokens: {config['max_tokens']}")
        print(f"5. Temperature: {config['temperature']}")
        print(f"6. Debug Mode: {config.get('debug_mode', False)}")
        print(f"7. Debug Level: {config.get('debug_level', 'basic')}")
        print("")
        
        print("Options:")
        print("1. Change AI Provider")
        print("2. Change Model")
        print("3. Change API Key (Current Provider)")
        print("4. Setup All Provider API Keys")
        print("5. Change Max Tokens")
        print("6. Change Temperature")
        print("7. Toggle Debug Mode")
        print("8. Change Debug Level")
        print("9. API Debugger")
        print("10. View Debug Log")
        print("11. Clear Debug Log")
        print("12. Switch to Reliable Model")
        print("13. Back to Chat")
        print("")
        
        choice = input("Select option (1-13): ").strip()
        
        if choice == '1':
            config = change_provider(config)
        elif choice == '2':
            config = change_model(config)
        elif choice == '3':
            config = change_api_key(config)
        elif choice == '4':
            config = setup_all_provider_keys(config)
        elif choice == '5':
            try:
                new_tokens = int(input(f"Enter new max tokens (current: {config['max_tokens']}): "))
                if 1 <= new_tokens <= 4000:
                    config['max_tokens'] = new_tokens
                    print(f"✅ Max tokens changed to: {new_tokens}")
                else:
                    print("❌ Max tokens must be between 1 and 4000 for free models.")
            except ValueError:
                print("❌ Please enter a valid number.")
        elif choice == '6':
            try:
                new_temp = float(input(f"Enter new temperature (current: {config['temperature']}): "))
                if 0 <= new_temp <= 2:
                    config['temperature'] = new_temp
                    print(f"✅ Temperature changed to: {new_temp}")
                else:
                    print("❌ Temperature must be between 0 and 2.")
            except ValueError:
                print("❌ Please enter a valid number.")
        elif choice == '7':
            config['debug_mode'] = not config.get('debug_mode', False)
            status = "enabled" if config['debug_mode'] else "disabled"
            print(f"✅ Debug mode {status}")
        elif choice == '8':
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
        elif choice == '9':
            api_debugger(config)
        elif choice == '10':
            view_debug_log(config)
        elif choice == '11':
            clear_debug_log()
        elif choice == '12':
            config = switch_to_reliable_model(config)
        elif choice == '13':
            return config
        else:
            print("❌ Invalid option. Please select 1-13.")
        
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
    if not get_current_api_key(config):
        print("🔧 First-time setup required...")
        config = setup_wizard()
        if not get_current_api_key(config):
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