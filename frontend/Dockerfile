# VITE_API_URL='.' keeps api.js on the same origin: nginx proxies /v1/ to the backend service.
FROM node:20.19.0-alpine3.21 AS build
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
ARG CABINET_MODE=full
ENV VITE_CABINET_MODE=$CABINET_MODE
ENV VITE_API_URL=.
ENV VITE_FEEDBACK_API=/feedback
RUN npm run build

FROM nginx:1.27-alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
