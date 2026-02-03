# Antigravity Cut - Claude Development Guide

## Project Context
A specialized video processing application using React (Vite), Remotion, and a Python-based backend for processing logic.

## Build and Development Commands
### Frontend (React/Vite)
- `npm run dev`: Start local development server
- `npm run build`: Build for production
- `npm run lint`: Run ESLint checks
- `npm run preview`: Preview production build

### Backend (Python)
- `python server.py`: Start the Python development server
- `pip install -r requirements.txt`: Install backend dependencies

## Code Style Guidelines
- **Framework**: React 19+ with Vite.
- **Styling**: Vanilla CSS with premium aesthetics (dark mode, glassmorphism).
- **TypeScript**: Strict typing encouraged. Use `zod` for validation.
- **Backend**: Python 3.x, use type hints.

## Development Workflow (from Everything-Claude-Code)
1. **Planning**: Use `/plan` for all non-trivial features. Create an implementation plan before coding.
2. **Testing**: Implement critical logic with TDD principles.
3. **Commit**: Use descriptive commit messages.
4. **Safety**: Do not commit secrets. Use environmental variables.

## Key Files
- `src/`: Frontend React source
- `server.py`: Main backend entry point
- `processor.py`: Video processing logic
- `.claude/`: Project-specific AI agents and rules
