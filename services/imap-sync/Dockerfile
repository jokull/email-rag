FROM golang:1.21-alpine AS builder

# Install git and ca-certificates (needed for downloading Go modules)
RUN apk add --no-cache git ca-certificates

WORKDIR /app

# Copy source code first so go mod tidy can analyze dependencies
COPY . .

# Download dependencies and tidy up (this will create go.sum)
RUN go mod download && go mod tidy

# Build the application
RUN CGO_ENABLED=0 GOOS=linux go build -o email-imap-server .

# Final stage
FROM alpine:latest

# Install ca-certificates for HTTPS/TLS connections
RUN apk --no-cache add ca-certificates

WORKDIR /root/

# Copy the binary from builder stage
COPY --from=builder /app/email-imap-server .

# Expose IMAP port
EXPOSE 1143

# Run the server
CMD ["./email-imap-server"]