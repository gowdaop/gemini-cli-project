# ── Build stage ─────────────────────────────────────────
FROM node:20-bookworm AS builder
WORKDIR /app
COPY src/legal-document-simplifier/src/frontend/package*.json ./
RUN npm ci
COPY src/legal-document-simplifier/src/frontend .
RUN npm run build           # Vite; emits /app/dist

# ── Serve stage ────────────────────────────────────────
FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
