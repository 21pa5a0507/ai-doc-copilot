# HexPilot AI Frontend

React + Vite frontend for HexPilot AI, a chat interface for asking questions across Hexnode documentation and Keka policy sources.

## Local Setup

Install dependencies:

```bash
npm install
```

Create a local `.env` file:

```bash
VITE_API_URL=http://localhost:8000
```

Start the development server:

```bash
npm run dev
```

## Environment Variables

| Variable | Description |
| --- | --- |
| `VITE_API_URL` | Backend base URL. Do not include `/ask`; the frontend appends it automatically. |

Example production value:

```bash
VITE_API_URL=https://your-backend-domain.com
```

## Available Scripts

```bash
npm run dev
npm run lint
npm run build
npm run preview
```

## Deployment Notes

For Vercel, use this directory as the project root:

```bash
frontend/ragline-ui
```

Set `VITE_API_URL` in Vercel before deploying. Redeploy after changing environment variables.
