FROM golang:1.21-alpine AS builder

# Install git and ca-certificates (needed for downloading Go modules)
RUN apk add --no-cache git ca-certificates

WORKDIR /app

# Copy go mod files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY . .

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