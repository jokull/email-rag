package main

import (
	"database/sql"
	"fmt"
	"log"
	"os"
	"strconv"
	"time"

	_ "github.com/lib/pq"
	"github.com/sirupsen/logrus"
)

type Config struct {
	DatabaseURL     string
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
		DatabaseURL:    os.Getenv("DATABASE_URL"),
		LogLevel:       getEnvDefault("LOG_LEVEL", "info"),
		SyncInterval:   parseDuration(getEnvDefault("SYNC_INTERVAL", "300s")),
	}

	// Upstream IMAP configuration
	config.UpstreamIMAP = UpstreamIMAPConfig{
		Host:     os.Getenv("IMAP_HOST"),
		Port:     parseInt(getEnvDefault("IMAP_PORT", "993")),
		Username: os.Getenv("IMAP_USER"),
		Password: os.Getenv("IMAP_PASS"),
		TLS:      getEnvDefault("IMAP_TLS", "true") == "true",
	}

	// Validate required fields
	if config.DatabaseURL == "" {
		return nil, fmt.Errorf("DATABASE_URL is required")
	}
	if config.UpstreamIMAP.Host == "" {
		return nil, fmt.Errorf("IMAP_HOST is required")
	}
	if config.UpstreamIMAP.Username == "" {
		return nil, fmt.Errorf("IMAP_USER is required")
	}
	if config.UpstreamIMAP.Password == "" {
		return nil, fmt.Errorf("IMAP_PASS is required")
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
	logger.Info("Starting Email RAG IMAP Sync Service")

	// Start email synchronization
	startEmailSync(config, logger)
}

func startEmailSync(config *Config, logger *logrus.Logger) {
	// Create separate database connection for email syncing
	db, err := sql.Open("postgres", config.DatabaseURL)
	if err != nil {
		logger.Errorf("Failed to connect to database for syncing: %v", err)
		return
	}
	defer db.Close()
	
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