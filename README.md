# 🎮 Game Character Recognition AI System

An advanced AI-powered system that identifies game characters from images using Claude's vision capabilities and provides comprehensive information about them.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Flask](https://img.shields.io/badge/flask-v3.0+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 📋 Table of Contents

- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [API Configuration](#api-configuration)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [API Endpoints](#api-endpoints)
- [Technologies Used](#technologies-used)
- [Troubleshooting](#troubleshooting)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

### 🔍 Core Functionality
- **AI-Powered Character Recognition** - Uses Claude Sonnet 4 vision API to identify game characters
- **Multi-Character Support** - Recognizes characters from any video game
- **Detailed Information Retrieval** - Provides comprehensive character data including:
  - Character name and game origin
  - Visual description
  - Abilities and powers
  - Background and lore
  - Popularity metrics
  - Cultural impact analysis
  - Source citations

### 🎨 User Interface
- **Beautiful Modern Design** - Gradient-based responsive UI
- **Drag & Drop Upload** - Easy image upload with drag-and-drop support
- **Real-time Analysis** - Live progress indicators and animations
- **Confidence Scoring** - Visual confidence meter for recognition accuracy
- **Mobile Responsive** - Works on desktop, tablet, and mobile devices

### ⚙️ Technical Features
- **Demo Mode** - Works without API key for testing
- **Performance Metrics** - Tracks success rate and usage statistics
- **Error Handling** - Graceful fallbacks and error messages
- **CORS Support** - Can be integrated with external frontends
- **Image Format Support** - JPG, PNG, GIF up to 10MB

## 🎬 Demo

### Screenshot
```
┌─────────────────────────────────────┐
│   🎮 Game Character Recognition AI  │
│   Upload any game character image   │
├─────────────────────────────────────┤
│                                     │
│         [Drop image here]           │
│          📸                         │
│       or click to browse            │
│                                     │
│      [Choose Image Button]          │
│                                     │
│    [🔍 Analyze Character]           │
│                                     │
└─────────────────────────────────────┘
```

### Example Output
```
Character: Mario
Game: Super Mario Bros.
Confidence: 95.2%

✓ Super Jump - Exceptional jumping ability
✓ Power-ups - Fire Flower, Super Star
✓ Ground Pound - Powerful downward attack
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/game-character-recognition.git
cd game-character-recognition
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install flask flask-cors pillow requests
```

### Step 4: Configuration (Optional)
For full AI recognition, add your Anthropic API key:

1. Open `main.py`
2. Find the line: `ANTHROPIC_API_KEY = ""`
3. Add your API key: `ANTHROPIC_API_KEY = "sk-ant-your-key-here"`

**Get your API key:** https://console.anthropic.com/

## 💻 Usage

### Running the Application

```bash
# Make sure virtual environment is activated
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows

# Run the application
python3 main.py
```

The server will start on **http://localhost:8080**

### Using the Web Interface

1. **Open your browser** and go to `http://localhost:8080`
2. **Upload an image** by:
   - Clicking "Choose Image" button, or
   - Dragging and dropping an image file
3. **Click "Analyze Character"**
4. **View results** with detailed character information

### Command Line Options

```bash
# Run on different port
PORT=5000 python3 main.py

# Run in production mode (requires gunicorn)
gunicorn -w 4 -b 0.0.0.0:8080 main:app
```

## 🔑 API Configuration

### Getting an Anthropic API Key

1. Go to https://console.anthropic.com/
2. Sign up or log in
3. Navigate to **API Keys** section
4. Click **Create Key**
5. Copy the key and add it to `main.py`

### Demo Mode vs Full Mode

| Feature | Demo Mode | Full Mode (with API key) |
|---------|-----------|--------------------------|
| Character Recognition | ❌ Generic | ✅ Accurate AI Vision |
| Any Character | ❌ Limited | ✅ All characters |
| Confidence Score | ⚠️ Low (30%) | ✅ High (80-95%) |
| Real-time Analysis | ❌ | ✅ |

## 📁 Project Structure

```
game-character-recognition/
│
├── main.py                 # Main application file
├── requirements.txt        # Python dependencies
├── README.md              # This file
│
├── venv/                  # Virtual environment (not in git)
│
└── static/                # (Optional) Static assets
    └── images/
```

### Key Files

**main.py** - Contains:
- Flask web server
- Character recognition system
- API endpoints
- HTML/CSS/JavaScript interface

**requirements.txt**:
```
flask>=3.0.0
flask-cors>=4.0.0
pillow>=10.0.0
requests>=2.31.0
```

## 🔧 How It Works

### System Architecture

```
┌─────────────┐
│   Browser   │
│  (Upload)   │
└──────┬──────┘
       │
       ▼
┌─────────────┐      ┌──────────────┐
│  Flask API  │─────▶│  Claude API  │
│   Server    │◀─────│   (Vision)   │
└──────┬──────┘      └──────────────┘
       │
       ▼
┌─────────────┐      ┌──────────────┐
│ Character   │─────▶│  Web Search  │
│ Enrichment  │◀─────│  (Future)    │
└─────────────┘      └──────────────┘
```

### Recognition Pipeline

1. **Image Upload** - User uploads character image
2. **Preprocessing** - Image converted to base64
3. **AI Analysis** - Claude Vision API analyzes the image
4. **Character Identification** - AI identifies character name and game
5. **Information Enrichment** - System gathers additional details
6. **Results Display** - Comprehensive information shown to user

### Claude API Integration

```python
# Example API call structure
{
  "model": "claude-sonnet-4-20250514",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "image", "source": {"data": base64_image}},
      {"type": "text", "text": "Identify this game character"}
    ]
  }]
}
```

## 🌐 API Endpoints

### POST /api/identify
Identify a character from an uploaded image.

**Request:**
```bash
curl -X POST http://localhost:8080/api/identify \
  -F "image=@character.jpg"
```

**Response:**
```json
{
  "success": true,
  "character": {
    "name": "Mario",
    "game": "Super Mario Bros.",
    "description": "Italian plumber...",
    "abilities": ["Super Jump", "Fire Flower"],
    "background": "First appeared in...",
    "popularity": "800M+ copies sold",
    "cultural_impact": "Gaming icon...",
    "sources": ["https://..."],
    "confidence": 0.95,
    "timestamp": "2026-01-25T00:42:10"
  }
}
```

### GET /api/metrics
Get system performance metrics.

**Response:**
```json
{
  "total_requests": 15,
  "successful": 14,
  "failed": 1,
  "success_rate": 93.33
}
```

### GET /api/check-api
Check if API key is configured.

**Response:**
```json
{
  "has_api_key": true
}
```

## 🛠️ Technologies Used

### Backend
- **Python 3.11** - Core programming language
- **Flask 3.0** - Web framework
- **Pillow (PIL)** - Image processing
- **Requests** - HTTP client for API calls
- **Flask-CORS** - Cross-origin resource sharing

### Frontend
- **HTML5** - Structure
- **CSS3** - Styling (Gradients, Animations, Flexbox)
- **JavaScript (ES6+)** - Interactivity
- **Fetch API** - AJAX requests

### AI/ML
- **Claude Sonnet 4** - Vision and language model
- **Anthropic API** - AI service provider

## 🐛 Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Error: Address already in use
# Solution: Use different port
PORT=8080 python3 main.py

# Or kill process on port 5000
lsof -ti:5000 | xargs kill -9
```

#### PIL Import Error
```bash
# Error: No module named 'PIL'
# Solution: Install Pillow
pip install pillow
```

#### API Key Issues
```bash
# Error: 401 Unauthorized
# Solution: Check API key is correct
# Make sure it starts with "sk-ant-"
```

#### Image Upload Fails
```bash
# Error: File too large
# Solution: Ensure image is under 10MB
# Resize if necessary
```

### Debug Mode

Enable detailed logging:
```python
# In main.py, set:
app.run(debug=True, host='0.0.0.0', port=8080)
```

### Testing Without API Key

The system works in demo mode without an API key:
- Shows generic character information
- Lower confidence scores
- Good for testing UI/UX

## 🚀 Future Enhancements

### Planned Features

#### Phase 1 - Enhanced Recognition
- [ ] Multi-character detection in single image
- [ ] Character pose/outfit variation handling
- [ ] Fan art and cosplay recognition
- [ ] Video game screenshot analysis

#### Phase 2 - Information Expansion
- [ ] Real-time web scraping for character info
- [ ] Integration with gaming databases (IGDB, Giant Bomb)
- [ ] Community-sourced character data
- [ ] Character comparison features

#### Phase 3 - Advanced Features
- [ ] Multilingual support (10+ languages)
- [ ] Voice character recognition (from audio clips)
- [ ] Character similarity finder
- [ ] Build your own character database
- [ ] Mobile app (iOS/Android)

#### Phase 4 - Performance
- [ ] Caching system for common characters
- [ ] Batch processing
- [ ] Real-time video analysis
- [ ] GPU acceleration

### Integration Ideas
- Discord bot
- Twitch extension
- Browser extension
- Mobile AR app

## 🤝 Contributing

Contributions are welcome! Here's how to help:

### Reporting Bugs
1. Check existing issues
2. Create new issue with:
   - Clear title
   - Steps to reproduce
   - Expected vs actual behavior
   - Screenshots if applicable

### Suggesting Features
1. Open an issue with `[FEATURE]` tag
2. Describe the feature
3. Explain use cases
4. Provide examples

### Code Contributions
1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

### Development Setup
```bash
# Fork and clone
git clone https://github.com/yourusername/game-character-recognition.git
cd game-character-recognition

# Create branch
git checkout -b feature/my-feature

# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest

# Make changes and commit
git add .
git commit -m "Description of changes"
git push origin feature/my-feature
```

## 📄 License

This project is licensed under the MIT License.

```
MIT License

Copyright (c) 2026 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 📞 Contact & Support

- **Author**: Aidin Obolbekov
- **Email**: your.email@example.com
- **GitHub**: [@yourusername](https://github.com/yourusername)
- **Issues**: [GitHub Issues](https://github.com/yourusername/game-character-recognition/issues)

## 🙏 Acknowledgments

- **Anthropic** - For Claude AI API
- **Flask Team** - For the excellent web framework
- **Pillow Contributors** - For image processing capabilities
- **Gaming Community** - For inspiration and testing

## 📊 Project Stats

![GitHub stars](https://img.shields.io/github/stars/yourusername/game-character-recognition?style=social)
![GitHub forks](https://img.shields.io/github/forks/yourusername/game-character-recognition?style=social)
![GitHub issues](https://img.shields.io/github/issues/yourusername/game-character-recognition)
![GitHub last commit](https://img.shields.io/github/last-commit/yourusername/game-character-recognition)

---

**Made with ❤️ for gamers and AI enthusiasts**

*Star ⭐ this repo if you find it useful!*
