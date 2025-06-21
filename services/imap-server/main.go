package main

import (
	"database/sql"
	"fmt"
	"log"
	"net"
	"os"
	"strconv"
	"time"

	"github.com/emersion/go-imap/server"
	imapsql "github.com/foxcpp/go-imap-sql"
	_ "github.com/lib/pq"
	"github.com/sirupsen/logrus"
)

type Config struct {
	DatabaseURL     string
	IMAPPort        string
	UpstreamIMAP    UpstreamIMAPConfig
	SyncInterval    time.Duration
	LogLevel        string
}

type UpstreamIMAPConfig struct {
	Host     string
	Port     int
	Username string
	Password string
	TLS      bool
}

func getConfig() (*Config, error) {
	config := &Config{
		DatabaseURL:  os.Getenv("DATABASE_URL"),
		IMAPPort:     getEnvDefault("IMAP_PORT", "1143"),
		LogLevel:     getEnvDefault("LOG_LEVEL", "info"),
		SyncInterval: parseDuration(getEnvDefault("SYNC_INTERVAL", "300s")),
	}

	// Upstream IMAP configuration
	config.UpstreamIMAP = UpstreamIMAPConfig{
		Host:     os.Getenv("UPSTREAM_IMAP_HOST"),
		Port:     parseInt(getEnvDefault("UPSTREAM_IMAP_PORT", "993")),
		Username: os.Getenv("UPSTREAM_IMAP_USER"),
		Password: os.Getenv("UPSTREAM_IMAP_PASS"),
		TLS:      getEnvDefault("UPSTREAM_IMAP_TLS", "true") == "true",
	}

	// Validate required fields
	if config.DatabaseURL == "" {
		return nil, fmt.Errorf("DATABASE_URL is required")
	}
	if config.UpstreamIMAP.Host == "" {
		return nil, fmt.Errorf("UPSTREAM_IMAP_HOST is required")
	}
	if config.UpstreamIMAP.Username == "" {
		return nil, fmt.Errorf("UPSTREAM_IMAP_USER is required")
	}
	if config.UpstreamIMAP.Password == "" {
		return nil, fmt.Errorf("UPSTREAM_IMAP_PASS is required")
	}

	return config, nil
}

func getEnvDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func parseInt(s string) int {
	if i, err := strconv.Atoi(s); err == nil {
		return i
	}
	return 0
}

func parseDuration(s string) time.Duration {
	if d, err := time.ParseDuration(s); err == nil {
		return d
	}
	return 5 * time.Minute
}

func setupLogger(level string) *logrus.Logger {
	logger := logrus.New()
	
	logLevel, err := logrus.ParseLevel(level)
	if err != nil {
		logLevel = logrus.InfoLevel
	}
	logger.SetLevel(logLevel)
	
	logger.SetFormatter(&logrus.TextFormatter{
		FullTimestamp: true,
	})
	
	return logger
}

func main() {
	config, err := getConfig()
	if err != nil {
		log.Fatalf("Configuration error: %v", err)
	}

	logger := setupLogger(config.LogLevel)
	logger.Info("Starting Email RAG IMAP Server")

	// Connect to PostgreSQL
	db, err := sql.Open("postgres", config.DatabaseURL)
	if err != nil {
		logger.Fatalf("Failed to connect to database: %v", err)
	}
	defer db.Close()

	// Test database connection
	if err := db.Ping(); err != nil {
		logger.Fatalf("Failed to ping database: %v", err)
	}
	logger.Info("Connected to PostgreSQL database")

	// Create IMAP backend using go-imap-sql
	backend, err := imapsql.NewBackend(db, &imapsql.Opts{
		// Use PostgreSQL-specific options
		MaxConns:        10,
		MaxIdleConns:    5,
		ConnMaxLifetime: time.Hour,
		
		// Enable compression for better performance
		CompressAlgo: imapsql.CompressionNone,
		
		// Use random UIDVALIDITY generation
		PRNG: nil,
	})
	if err != nil {
		logger.Fatalf("Failed to create IMAP backend: %v", err)
	}
	defer backend.Close()

	// Create IMAP server
	imapServer := server.New(backend)
	imapServer.Addr = ":" + config.IMAPPort
	imapServer.ErrorLog = logger.StandardLogger()

	// Enable IMAP extensions
	imapServer.Enable(server.EnableIDLE, server.EnableMove, server.EnableSpecialUse)

	logger.Infof("IMAP server listening on port %s", config.IMAPPort)

	// Start email synchronization in background
	go startEmailSync(config, logger, db)

	// Start IMAP server
	listener, err := net.Listen("tcp", imapServer.Addr)
	if err != nil {
		logger.Fatalf("Failed to start IMAP server: %v", err)
	}

	logger.Fatal(imapServer.Serve(listener))
}

func startEmailSync(config *Config, logger *logrus.Logger, db *sql.DB) {
	syncer := NewEmailSyncer(config, logger, db)
	
	// Perform initial sync
	logger.Info("Starting initial email synchronization...")
	if err := syncer.SyncEmails(); err != nil {
		logger.Errorf("Initial sync failed: %v", err)
	}

	// Start periodic sync
	ticker := time.NewTicker(config.SyncInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			logger.Debug("Starting periodic email sync...")
			if err := syncer.SyncEmails(); err != nil {
				logger.Errorf("Periodic sync failed: %v", err)
			}
		}
	}
}