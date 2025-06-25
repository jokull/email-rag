use pyo3::prelude::*;
use mail_parser::{Message, MessageParser};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct EmailHeaders {
    #[pyo3(get)]
    pub message_id: Option<String>,
    #[pyo3(get)]
    pub subject: Option<String>,
    #[pyo3(get)]
    pub from_email: Option<String>,
    #[pyo3(get)]
    pub from_name: Option<String>,
    #[pyo3(get)]
    pub to_emails: Vec<String>,
    #[pyo3(get)]
    pub cc_emails: Vec<String>,
    #[pyo3(get)]
    pub reply_to: Option<String>,
    #[pyo3(get)]
    pub references: Option<String>,
    #[pyo3(get)]
    pub in_reply_to: Option<String>,
    #[pyo3(get)]
    pub thread_topic: Option<String>,
    #[pyo3(get)]
    pub date_sent: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct ParsedEmail {
    #[pyo3(get)]
    pub headers: EmailHeaders,
    #[pyo3(get)]
    pub body_text: Option<String>,
    #[pyo3(get)]
    pub body_html: Option<String>,
    #[pyo3(get)]
    pub participants: Vec<(String, Option<String>)>, // (email, name) pairs
}

#[pymethods]
impl EmailHeaders {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("EmailHeaders(message_id={:?}, subject={:?})", 
                  self.message_id, self.subject))
    }
}

#[pymethods]
impl ParsedEmail {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("ParsedEmail(message_id={:?}, participants={})", 
                  self.headers.message_id, self.participants.len()))
    }
    
    fn to_dict(&self) -> PyResult<HashMap<String, PyObject>> {
        Python::with_gil(|py| {
            let mut dict = HashMap::new();
            
            dict.insert("message_id".to_string(), self.headers.message_id.to_object(py));
            dict.insert("subject".to_string(), self.headers.subject.to_object(py));
            dict.insert("from_email".to_string(), self.headers.from_email.to_object(py));
            dict.insert("from_name".to_string(), self.headers.from_name.to_object(py));
            dict.insert("to_emails".to_string(), self.headers.to_emails.to_object(py));
            dict.insert("cc_emails".to_string(), self.headers.cc_emails.to_object(py));
            dict.insert("reply_to".to_string(), self.headers.reply_to.to_object(py));
            dict.insert("references".to_string(), self.headers.references.to_object(py));
            dict.insert("in_reply_to".to_string(), self.headers.in_reply_to.to_object(py));
            dict.insert("thread_topic".to_string(), self.headers.thread_topic.to_object(py));
            dict.insert("date_sent".to_string(), self.headers.date_sent.to_object(py));
            dict.insert("body_text".to_string(), self.body_text.to_object(py));
            dict.insert("body_html".to_string(), self.body_html.to_object(py));
            dict.insert("participants".to_string(), self.participants.to_object(py));
            
            Ok(dict)
        })
    }
}

fn extract_addresses(addr_header: &mail_parser::Address) -> Vec<String> {
    let mut emails = Vec::new();
    
    // The Address enum has different variants, let's handle them
    match addr_header {
        mail_parser::Address::List(addr_list) => {
            for addr in addr_list {
                if let Some(email) = &addr.address {
                    emails.push(email.to_string());
                }
            }
        }
        mail_parser::Address::Group(group) => {
            for addr in &group.addresses {
                if let Some(email) = &addr.address {
                    emails.push(email.to_string());
                }
            }
        }
    }
    
    emails
}

fn extract_participants(addr_header: &mail_parser::Address) -> Vec<(String, Option<String>)> {
    let mut participants = Vec::new();
    
    match addr_header {
        mail_parser::Address::List(addr_list) => {
            for addr in addr_list {
                if let Some(email) = &addr.address {
                    let name = addr.name.as_ref().map(|n| n.to_string());
                    participants.push((email.to_string(), name));
                }
            }
        }
        mail_parser::Address::Group(group) => {
            for addr in &group.addresses {
                if let Some(email) = &addr.address {
                    let name = addr.name.as_ref().map(|n| n.to_string());
                    participants.push((email.to_string(), name));
                }
            }
        }
    }
    
    participants
}

#[pyfunction]
fn parse_email_bytes(raw_bytes: &[u8]) -> PyResult<ParsedEmail> {
    let message = MessageParser::default()
        .parse(raw_bytes)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Failed to parse email: No message found"))?;
    
    // Extract basic headers
    let message_id = message.message_id()
        .and_then(|ids| ids.first())
        .map(|id| id.to_string());
    
    let subject = message.subject().map(|s| s.to_string());
    
    // Extract from address
    let (from_email, from_name) = if let Some(from_header) = message.from() {
        match from_header {
            mail_parser::Address::List(addr_list) => {
                if let Some(addr) = addr_list.first() {
                    (
                        addr.address.as_ref().map(|e| e.to_string()),
                        addr.name.as_ref().map(|n| n.to_string())
                    )
                } else {
                    (None, None)
                }
            }
            mail_parser::Address::Group(group) => {
                if let Some(addr) = group.addresses.first() {
                    (
                        addr.address.as_ref().map(|e| e.to_string()),
                        addr.name.as_ref().map(|n| n.to_string())
                    )
                } else {
                    (None, None)
                }
            }
        }
    } else {
        (None, None)
    };
    
    // Extract to and cc emails
    let to_emails = message.to()
        .map(extract_addresses)
        .unwrap_or_default();
    
    let cc_emails = message.cc()
        .map(extract_addresses)
        .unwrap_or_default();
    
    // Extract reply-to
    let reply_to = message.reply_to()
        .and_then(|reply_header| {
            match reply_header {
                mail_parser::Address::List(addr_list) => {
                    addr_list.first().and_then(|a| a.address.as_ref().map(|e| e.to_string()))
                }
                mail_parser::Address::Group(group) => {
                    group.addresses.first().and_then(|a| a.address.as_ref().map(|e| e.to_string()))
                }
            }
        });
    
    // Extract threading headers
    let references = message.references()
        .map(|refs| refs.iter().map(|r| r.to_string()).collect::<Vec<_>>().join(" "));
    
    let in_reply_to = message.in_reply_to()
        .map(|id| id.to_string());
    
    // Thread topic is not available in mail-parser 0.9
    let thread_topic = None;
    
    let date_sent = message.date().map(|d| d.to_rfc3339());
    
    let headers = EmailHeaders {
        message_id,
        subject,
        from_email,
        from_name,
        to_emails,
        cc_emails,
        reply_to,
        references,
        in_reply_to,
        thread_topic,
        date_sent,
    };
    
    // Extract body content
    let body_text = message.body_text(0).map(|s| s.to_string());
    let body_html = message.body_html(0).map(|s| s.to_string());
    
    // Extract all participants (deduplicated)
    let mut participants = Vec::new();
    let mut seen_emails = std::collections::HashSet::new();
    
    // Add from participants
    if let Some(from_header) = message.from() {
        for (email, name) in extract_participants(from_header) {
            let email_lower = email.to_lowercase();
            if seen_emails.insert(email_lower.clone()) {
                participants.push((email_lower, name));
            }
        }
    }
    
    // Add to participants
    if let Some(to_header) = message.to() {
        for (email, name) in extract_participants(to_header) {
            let email_lower = email.to_lowercase();
            if seen_emails.insert(email_lower.clone()) {
                participants.push((email_lower, name));
            }
        }
    }
    
    // Add cc participants
    if let Some(cc_header) = message.cc() {
        for (email, name) in extract_participants(cc_header) {
            let email_lower = email.to_lowercase();
            if seen_emails.insert(email_lower.clone()) {
                participants.push((email_lower, name));
            }
        }
    }
    
    Ok(ParsedEmail {
        headers,
        body_text,
        body_html,
        participants,
    })
}

#[pyfunction]
fn parse_email_string(raw_email: &str) -> PyResult<ParsedEmail> {
    parse_email_bytes(raw_email.as_bytes())
}

#[pymodule]
fn email_parser_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_email_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(parse_email_string, m)?)?;
    m.add_class::<EmailHeaders>()?;
    m.add_class::<ParsedEmail>()?;
    Ok(())
}