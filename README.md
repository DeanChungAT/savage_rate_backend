# Savage Rate Backend

Backend service for the Savage Rate application.

## Setup

1. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Set up environment variables
```bash
cp .env.example .env
# Edit .env with your configurations
```

## Development

Start the development server:
```bash
python main.py
```

## API Documentation

API endpoints will be documented here.

## Error Handling

All errors are logged and returned with appropriate HTTP status codes.

## License

MIT 