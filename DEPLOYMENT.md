# Vercel frontend + Render backend

This deployment keeps the Flask frontend on Vercel and runs the FastAPI backend from the repository's Dockerfile on Render.

## Render backend

1. In Render, create a Blueprint from this GitHub repository and select branch `v3`.
2. Render reads `render.yaml` and creates only the FastAPI backend service.
3. Set `GROQ_API_KEY` in the service's environment variables. Keep it out of Git and the Docker image.
4. Wait for the first deploy to finish, then copy the service's `https://…onrender.com` URL. Its health check is `/health`.

The free service has limited memory and sleeps after inactivity. This backend loads PyTorch and a Sentence Transformers model; if it runs out of memory, use a larger instance or reduce the model footprint. Local filesystem data on a free instance is not persistent.

## Vercel frontend

1. Import the same GitHub repository into Vercel and select branch `v3`.
2. Set the project root directory to `flask_frontend`.
3. Add `BACKEND_URL` with the Render URL, without a trailing slash.
4. Deploy the project. The Vercel production domain stays stable across future deployments.

The frontend and backend both enforce a 4 MB upload limit so PDF requests remain below Vercel's function payload limit. Recruiter batches must total 4 MB or less.
