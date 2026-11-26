// Shared by the examples: a cached client for an OpenAI-compatible chat
// endpoint and two small statistics.

use anyhow::{Context, Result, bail};
use std::path::PathBuf;

pub fn fnv64(s: &str) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}")
}

// not every example reads the token counts
#[allow(dead_code)]
pub struct Reply {
    pub content: String,
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
}

pub struct Client {
    pub agent: ureq::Agent,
    pub base: String,
    pub key: String,
    pub model: String,
    pub reasoning_effort: Option<String>,
    pub cache: PathBuf,
    pub min_gap: std::time::Duration,
    pub last: Option<std::time::Instant>,
    pub max_tokens: u64,
}

impl Client {
    pub fn ask(&mut self, prompt: &str) -> Result<Reply> {
        let key = fnv64(&format!("{}\0{}\0{prompt}", self.model, self.reasoning_effort.as_deref().unwrap_or("")));
        let path = self.cache.join(format!("{key}.json"));
        if let Ok(s) = std::fs::read_to_string(&path) {
            let v: serde_json::Value = serde_json::from_str(&s)?;
            return Ok(Reply {
                content: v["content"].as_str().unwrap_or("").to_string(),
                prompt_tokens: v["prompt_tokens"].as_u64(),
                completion_tokens: v["completion_tokens"].as_u64(),
            });
        }
        let mut body = serde_json::json!({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": self.max_tokens,
        });
        if let Some(e) = &self.reasoning_effort {
            body["reasoning_effort"] = serde_json::Value::String(e.clone());
        }
        let mut wait = std::time::Duration::from_secs(5);
        for attempt in 0..10 {
            if let Some(t) = self.last {
                let since = t.elapsed();
                if since < self.min_gap {
                    std::thread::sleep(self.min_gap - since);
                }
            }
            self.last = Some(std::time::Instant::now());
            let mut req = self.agent.post(&format!("{}/chat/completions", self.base));
            if !self.key.is_empty() {
                req = req.header("Authorization", &format!("Bearer {}", self.key));
            }
            let resp = req.send_json(&body);
            let mut resp = match resp {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("request failed ({e}), retry in {}s", wait.as_secs());
                    std::thread::sleep(wait);
                    wait = (wait * 2).min(std::time::Duration::from_secs(120));
                    continue;
                }
            };
            let status = resp.status().as_u16();
            let text = resp.body_mut().read_to_string().unwrap_or_default();
            if status == 429 || status >= 500 {
                let short: String = text.chars().take(300).collect();
                eprintln!("http {status} (attempt {}), retry in {}s: {}", attempt + 1, wait.as_secs(), short.trim());
                std::thread::sleep(wait);
                wait = (wait * 2).min(std::time::Duration::from_secs(120));
                continue;
            }
            if status != 200 {
                bail!("http {status}: {text}");
            }
            let v: serde_json::Value = serde_json::from_str(&text).context("response is not JSON")?;
            let content = v["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string();
            let prompt_tokens = v["usage"]["prompt_tokens"].as_u64();
            let completion_tokens = v["usage"]["completion_tokens"].as_u64();
            std::fs::write(&path, serde_json::json!({"content": content, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}).to_string())?;
            return Ok(Reply { content, prompt_tokens, completion_tokens });
        }
        bail!("gave up after 10 attempts")
    }
}

impl Client {
    /// A client from the environment: LLM_BASE_URL (default OpenRouter),
    /// LLM_API_KEY (may be empty), requests at `rpm` per minute.
    pub fn from_env(model: &str, reasoning_effort: Option<String>, cache: &std::path::Path, rpm: u64, max_tokens: u64) -> Result<Client> {
        let base = std::env::var("LLM_BASE_URL").unwrap_or_else(|_| "https://openrouter.ai/api/v1".to_string());
        let key = std::env::var("LLM_API_KEY").unwrap_or_default();
        if key.is_empty() {
            eprintln!("LLM_API_KEY is not set, sending requests without one");
        }
        std::fs::create_dir_all(cache)?;
        let agent = ureq::Agent::config_builder()
            .http_status_as_error(false)
            .timeout_global(Some(std::time::Duration::from_secs(600)))
            .build()
            .new_agent();
        Ok(Client {
            agent,
            base: base.trim_end_matches('/').to_string(),
            key,
            model: model.to_string(),
            reasoning_effort,
            cache: cache.to_path_buf(),
            min_gap: std::time::Duration::from_millis(60_000 / rpm.max(1)),
            last: None,
            max_tokens,
        })
    }
}
