FROM oven/bun:1-alpine

WORKDIR /app

# Copy package files
COPY package.json bun.lockb* ./

# Install dependencies
RUN bun install --frozen-lockfile

# Copy source code
COPY . .

# Build the application
RUN bun run build

# Install serve for serving static files
RUN bun add -g serve

# Expose port
EXPOSE 3000

# Start the application
CMD ["bunx", "serve", "-s", "dist", "-l", "3000"]