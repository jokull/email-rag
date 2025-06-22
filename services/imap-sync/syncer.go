package main

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"io"
	"strconv"
	"strings"
	"time"

	"github.com/emersion/go-imap"
	"github.com/emersion/go-imap/client"
	"github.com/emersion/go-message/mail"
	"github.com/sirupsen/logrus"
)

type EmailSyncer struct {
	config *Config
	logger *logrus.Logger
	db     *sql.DB
}

type EmailEnvelope struct {
	MessageID string                 `json:"message_id"`
	From      []EmailAddress         `json:"from"`
	To        []EmailAddress         `json:"to"`
	CC        []EmailAddress         `json:"cc"`
	BCC       []EmailAddress         `json:"bcc"`
	Subject   string                 `json:"subject"`
	Date      time.Time              `json:"date"`
	InReplyTo string                 `json:"in_reply_to"`
	Headers   map[string]interface{} `json:"headers"`
}

type EmailAddress struct {
	Name  string `json:"name"`
	Email string `json:"email"`
}

func NewEmailSyncer(config *Config, logger *logrus.Logger, db *sql.DB) *EmailSyncer {
	return &EmailSyncer{
		config: config,
		logger: logger,
		db:     db,
	}
}

func (s *EmailSyncer) SyncEmails() error {
	s.logger.Info("Connecting to upstream IMAP server...")

	// Connect to upstream IMAP server
	var c *client.Client
	var err error

	if s.config.UpstreamIMAP.TLS {
		c, err = client.DialTLS(fmt.Sprintf("%s:%d", s.config.UpstreamIMAP.Host, s.config.UpstreamIMAP.Port), nil)
	} else {
		c, err = client.Dial(fmt.Sprintf("%s:%d", s.config.UpstreamIMAP.Host, s.config.UpstreamIMAP.Port))
	}
	if err != nil {
		return fmt.Errorf("failed to connect to upstream IMAP: %v", err)
	}
	defer c.Close()

	// Login
	if err := c.Login(s.config.UpstreamIMAP.Username, s.config.UpstreamIMAP.Password); err != nil {
		return fmt.Errorf("failed to login to upstream IMAP: %v", err)
	}
	s.logger.Info("Successfully connected to upstream IMAP")

	// List mailboxes
	mailboxes := make(chan *imap.MailboxInfo, 10)
	done := make(chan error, 1)
	go func() {
		done <- c.List("", "*", mailboxes)
	}()

	var mailboxList []*imap.MailboxInfo
	for m := range mailboxes {
		mailboxList = append(mailboxList, m)
	}

	if err := <-done; err != nil {
		return fmt.Errorf("failed to list mailboxes: %v", err)
	}

	s.logger.Infof("Found %d mailboxes", len(mailboxList))

	// Sync each mailbox
	for _, mailbox := range mailboxList {
		if err := s.syncMailbox(c, mailbox.Name); err != nil {
			s.logger.Errorf("Failed to sync mailbox %s: %v", mailbox.Name, err)
			continue
		}
	}

	if err := c.Logout(); err != nil {
		s.logger.Warnf("Failed to logout cleanly: %v", err)
	}

	return nil
}

func (s *EmailSyncer) syncMailbox(c *client.Client, mailboxName string) error {
	s.logger.Infof("Syncing mailbox: %s", mailboxName)

	// Select mailbox
	mbox, err := c.Select(mailboxName, true) // read-only
	if err != nil {
		return fmt.Errorf("failed to select mailbox %s: %v", mailboxName, err)
	}

	if mbox.Messages == 0 {
		s.logger.Infof("Mailbox %s is empty", mailboxName)
		return nil
	}

	// Create or get mailbox record
	mailboxID, err := s.ensureMailbox(mailboxName, mbox.UidValidity)
	if err != nil {
		return fmt.Errorf("failed to ensure mailbox: %v", err)
	}

	// Get highest UID we've already synced
	lastUID, err := s.getLastSyncedUID(mailboxID)
	if err != nil {
		return fmt.Errorf("failed to get last synced UID: %v", err)
	}

	s.logger.Infof("Last synced UID for %s: %d, mailbox has %d messages", mailboxName, lastUID, mbox.Messages)

	// Search for new messages
	var seqSet *imap.SeqSet
	if lastUID > 0 {
		// Get messages newer than last UID
		seqSet = new(imap.SeqSet)
		seqSet.AddRange(lastUID+1, mbox.UidNext-1)
	} else {
		// Get all messages (initial sync)
		seqSet = new(imap.SeqSet)
		seqSet.AddRange(1, mbox.Messages)
	}

	if seqSet.Empty() {
		s.logger.Infof("No new messages in mailbox %s", mailboxName)
		return nil
	}

	// Fetch message envelopes, UIDs, and full content
	messages := make(chan *imap.Message, 10)
	done = make(chan error, 1)
	
	fetchItems := []imap.FetchItem{
		imap.FetchEnvelope, 
		imap.FetchUid, 
		imap.FetchFlags, 
		imap.FetchInternalDate, 
		imap.FetchRFC822Size,
		imap.FetchRFC822, // Fetch full message content for processing
	}
	go func() {
		done <- c.UidFetch(seqSet, fetchItems, messages)
	}()

	messageCount := 0
	for msg := range messages {
		if err := s.processMessage(mailboxID, mailboxName, msg); err != nil {
			s.logger.Errorf("Failed to process message UID %d: %v", msg.Uid, err)
			continue
		}
		messageCount++
	}

	if err := <-done; err != nil {
		return fmt.Errorf("failed to fetch messages: %v", err)
	}

	s.logger.Infof("Processed %d new messages from mailbox %s", messageCount, mailboxName)
	return nil
}

func (s *EmailSyncer) ensureMailbox(name string, uidValidity uint32) (int, error) {
	// Check if mailbox exists
	var mailboxID int
	err := s.db.QueryRow(`
		SELECT id FROM imap_mailboxes 
		WHERE username = $1 AND name = $2
	`, "email_sync_user", name).Scan(&mailboxID)

	if err == sql.ErrNoRows {
		// Create new mailbox
		err = s.db.QueryRow(`
			INSERT INTO imap_mailboxes (username, name, uidvalidity, uidnext)
			VALUES ($1, $2, $3, 1)
			RETURNING id
		`, "email_sync_user", name, uidValidity).Scan(&mailboxID)
		if err != nil {
			return 0, fmt.Errorf("failed to create mailbox: %v", err)
		}
		s.logger.Infof("Created new mailbox record: %s (ID: %d)", name, mailboxID)
	} else if err != nil {
		return 0, fmt.Errorf("failed to query mailbox: %v", err)
	}

	return mailboxID, nil
}

func (s *EmailSyncer) getLastSyncedUID(mailboxID int) (uint32, error) {
	var lastUID uint32
	err := s.db.QueryRow(`
		SELECT COALESCE(MAX(uid), 0) FROM imap_messages 
		WHERE mailbox_id = $1
	`, mailboxID).Scan(&lastUID)
	
	if err != nil {
		return 0, fmt.Errorf("failed to get last synced UID: %v", err)
	}
	
	return lastUID, nil
}

func (s *EmailSyncer) processMessage(mailboxID int, mailboxName string, msg *imap.Message) error {
	// Convert IMAP envelope to our format
	envelope := s.convertEnvelope(msg.Envelope)
	
	// Serialize envelope as JSON
	envelopeJSON, err := json.Marshal(envelope)
	if err != nil {
		return fmt.Errorf("failed to marshal envelope: %v", err)
	}

	// Convert flags to string array
	flags := make([]string, len(msg.Flags))
	for i, flag := range msg.Flags {
		flags[i] = string(flag)
	}

	// Get raw message content
	var rawMessage []byte
	if r := msg.GetBody(&imap.BodySectionName{}); r != nil {
		rawMessage, err = io.ReadAll(r)
		if err != nil {
			s.logger.Warnf("Failed to read message body for UID %d: %v", msg.Uid, err)
			rawMessage = []byte{} // Continue with empty body
		}
	}

	// Insert into imap_messages table (which will trigger sync to emails table)
	_, err = s.db.Exec(`
		INSERT INTO imap_messages (
			mailbox_id, uid, flags, internal_date, size, 
			envelope, raw_message, created_at
		) VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
		ON CONFLICT (mailbox_id, uid) DO NOTHING
	`, mailboxID, msg.Uid, flags, msg.InternalDate, msg.Size, envelopeJSON, rawMessage)

	if err != nil {
		return fmt.Errorf("failed to insert message: %v", err)
	}

	return nil
}

func (s *EmailSyncer) convertEnvelope(env *imap.Envelope) EmailEnvelope {
	envelope := EmailEnvelope{
		MessageID: env.MessageId,
		Subject:   env.Subject,
		Date:      env.Date,
		InReplyTo: env.InReplyTo,
		Headers:   make(map[string]interface{}),
	}

	// Convert addresses
	envelope.From = s.convertAddresses(env.From)
	envelope.To = s.convertAddresses(env.To)
	envelope.CC = s.convertAddresses(env.Cc)
	envelope.BCC = s.convertAddresses(env.Bcc)

	return envelope
}

func (s *EmailSyncer) convertAddresses(addresses []*imap.Address) []EmailAddress {
	if addresses == nil {
		return nil
	}

	result := make([]EmailAddress, len(addresses))
	for i, addr := range addresses {
		email := ""
		if addr.MailboxName != "" && addr.HostName != "" {
			email = addr.MailboxName + "@" + addr.HostName
		}
		
		result[i] = EmailAddress{
			Name:  addr.PersonalName,
			Email: email,
		}
	}
	return result
}