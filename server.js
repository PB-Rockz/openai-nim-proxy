// server.js - OpenAI-compatible proxy to NVIDIA NIM (debuggable + broad client support)
const express = require('express');
const cors = require('cors');
const axios = require('axios');
const crypto = require('crypto');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '2mb' }));

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// Debug logging controls
const DEBUG_LOG_REQUESTS = (process.env.DEBUG_LOG_REQUESTS || 'true').toLowerCase() === 'true';
const DEBUG_LOG_HEADERS = (process.env.DEBUG_LOG_HEADERS || 'true').toLowerCase() === 'true';
const DEBUG_LOG_BODY = (process.env.DEBUG_LOG_BODY || 'true').toLowerCase() === 'true';
const DEBUG_LOG_MAX_BODY_CHARS = Number(process.env.DEBUG_LOG_MAX_BODY_CHARS || 12000);

// Behavior toggles
const SHOW_REASONING = (process.env.SHOW_REASONING || 'false').toLowerCase() === 'true'; // wrap reasoning in <think>
const ENABLE_THINKING_MODE = (process.env.ENABLE_THINKING_MODE || 'false').toLowerCase() === 'true'; // extra_body.chat_template_kwargs.thinking=true

// If true, we will *not* log sensitive headers. You can turn this off if you really want full raw headers.
const REDACT_SENSITIVE = (process.env.REDACT_SENSITIVE || 'true').toLowerCase() === 'true';

// --------- helpers ----------
function redactHeaders(headers) {
  if (!headers || typeof headers !== 'object') return headers;
  const redacted = { ...headers };
  const sensitive = ['authorization', 'cookie', 'set-cookie', 'x-api-key', 'proxy-authorization'];
  for (const k of Object.keys(redacted)) {
    if (sensitive.includes(k.toLowerCase()) && REDACT_SENSITIVE) {
      redacted[k] = '[REDACTED]';
    }
  }
  return redacted;
}

function safeStringify(obj) {
  try {
    return JSON.stringify(obj);
  } catch {
    return '[Unserializable]';
  }
}

function clip(str, maxChars) {
  if (typeof str !== 'string') str = safeStringify(str);
  if (str.length <= maxChars) return str;
  return str.slice(0, maxChars) + `... [clipped ${str.length - maxChars} chars]`;
}

// Request logger (logs full req info including body; sensitive headers redacted by default)
app.use((req, res, next) => {
  const requestId = req.headers['x-request-id'] || crypto.randomUUID();
  req._requestId = requestId;

  const start = Date.now();

  if (DEBUG_LOG_REQUESTS) {
    console.log('='.repeat(80));
    console.log(`[${requestId}] INCOMING ${req.method} ${req.originalUrl}`);
    console.log(`[${requestId}] IP: ${req.ip}`);
    console.log(`[${requestId}] Query: ${safeStringify(req.query)}`);

    if (DEBUG_LOG_HEADERS) {
      console.log(`[${requestId}] Headers: ${clip(safeStringify(redactHeaders(req.headers)), DEBUG_LOG_MAX_BODY_CHARS)}`);
    }

    if (DEBUG_LOG_BODY) {
      console.log(`[${requestId}] Body: ${clip(safeStringify(req.body), DEBUG_LOG_MAX_BODY_CHARS)}`);
    }
  }

  res.on('finish', () => {
    if (DEBUG_LOG_REQUESTS) {
      console.log(
        `[${requestId}] RESPONSE ${res.statusCode} (${Date.now() - start}ms) for ${req.method} ${req.originalUrl}`
      );
      console.log('='.repeat(80));
    }
  });

  next();
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI -> NVIDIA NIM Proxy (broad compatibility + debug logging)',
    nim_api_base: NIM_API_BASE,
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE,
    debug: {
      DEBUG_LOG_REQUESTS,
      DEBUG_LOG_HEADERS,
      DEBUG_LOG_BODY,
      REDACT_SENSITIVE
    }
  });
});

// List models endpoint (OpenAI compatible)
// We don't maintain a list; you provide model IDs directly.
app.get(['/v1/models', '/models'], (req, res) => {
  res.json({ object: 'list', data: [] });
});

// Alias routes for maximum compatibility with clients
// Some clients call /chat/completions (no /v1), some call /v1/chat/completions.
function chatCompletionsHandler(req, res) {
  return handleChatCompletions(req, res);
}
app.post('/v1/chat/completions', chatCompletionsHandler);
app.post('/chat/completions', chatCompletionsHandler);

// Optional: handle OPTIONS preflight explicitly (some clients are picky)
app.options(['/v1/chat/completions', '/chat/completions'], (req, res) => {
  res.sendStatus(204);
});

// Main handler
async function handleChatCompletions(req, res) {
  const requestId = req._requestId || 'no-request-id';

  try {
    // Accept multiple client field variants
    const {
      model,
      messages,
      temperature,
      max_tokens,
      max_completion_tokens,
      stream,
      top_p,
      presence_penalty,
      frequency_penalty,
      stop
    } = req.body || {};

    if (!model) {
      return res.status(400).json({
        error: {
          message: 'Missing required field: model',
          type: 'invalid_request_error',
          code: 400
        }
      });
    }

    if (!Array.isArray(messages)) {
      return res.status(400).json({
        error: {
          message: 'Missing/invalid field: messages (must be an array)',
          type: 'invalid_request_error',
          code: 400
        }
      });
    }

    // Build NIM request (model is used AS-IS)
    const nimRequest = {
      model,
      messages,
      temperature: temperature ?? 0.6,
      max_tokens: max_tokens ?? max_completion_tokens ?? 9024,
      stream: !!stream
    };

    // Pass through common OpenAI-ish params when provided
    if (top_p !== undefined) nimRequest.top_p = top_p;
    if (presence_penalty !== undefined) nimRequest.presence_penalty = presence_penalty;
    if (frequency_penalty !== undefined) nimRequest.frequency_penalty = frequency_penalty;
    if (stop !== undefined) nimRequest.stop = stop;

    // Enable "thinking mode" (some NIM models support this)
    if (ENABLE_THINKING_MODE) {
      nimRequest.extra_body = { chat_template_kwargs: { thinking: true } };
    }

    if (!NIM_API_KEY) {
      return res.status(500).json({
        error: {
          message: 'Server misconfigured: NIM_API_KEY is not set',
          type: 'server_error',
          code: 500
        }
      });
    }

    if (!nimRequest.stream) {
      // Non-streaming request
      const upstream = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
        headers: {
          Authorization: `Bearer ${NIM_API_KEY}`,
          'Content-Type': 'application/json',
          'X-Request-ID': requestId
        }
      });

      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model,
        choices: (upstream.data.choices || []).map((choice, i) => {
          const msg = choice.message || {};
          let fullContent = msg.content || '';

          if (SHOW_REASONING && msg.reasoning_content) {
            fullContent = `<think>\n${msg.reasoning_content}\n</think>\n\n` + fullContent;
          }

          return {
            index: choice.index ?? i,
            message: { role: msg.role || 'assistant', content: fullContent },
            finish_reason: choice.finish_reason || null
          };
        }),
        usage: upstream.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };

      return res.json(openaiResponse);
    }

    // Streaming request
    const upstream = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        Authorization: `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json',
        'X-Request-ID': requestId
      },
      responseType: 'stream',
      // prevent axios from buffering large streams
      maxBodyLength: Infinity,
      maxContentLength: Infinity
    });

    // OpenAI-style SSE
    res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
    res.setHeader('Cache-Control', 'no-cache, no-transform');
    res.setHeader('Connection', 'keep-alive');
    res.flushHeaders?.();

    // Some clients like an initial keepalive comment
    res.write(':\n\n');

    let buffer = '';
    let reasoningStarted = false;

    upstream.data.on('data', (chunk) => {
      buffer += chunk.toString();

      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;

        // Pass through DONE marker
        if (line.includes('[DONE]')) {
          res.write('data: [DONE]\n\n');
          continue;
        }

        try {
          const data = JSON.parse(line.slice(6));
          const delta = data?.choices?.[0]?.delta;

          if (delta) {
            const reasoning = delta.reasoning_content;
            const content = delta.content;

            if (SHOW_REASONING) {
              let combined = '';

              if (reasoning && !reasoningStarted) {
                combined = '<think>\n' + reasoning;
                reasoningStarted = true;
              } else if (reasoning) {
                combined = reasoning;
              }

              if (content && reasoningStarted) {
                combined += '</think>\n\n' + content;
                reasoningStarted = false;
              } else if (content) {
                combined += content;
              }

              delta.content = combined || '';
            } else {
              delta.content = content || '';
            }

            // Remove reasoning field so OpenAI clients don't choke
            delete delta.reasoning_content;
          }

          res.write(`data: ${JSON.stringify(data)}\n\n`);
        } catch (e) {
          // If parsing fails, forward the raw line (some upstreams send non-JSON keepalives)
          res.write(line + '\n\n');
        }
      }
    });

    upstream.data.on('end', () => res.end());
    upstream.data.on('error', (err) => {
      console.error(`[${requestId}] Upstream stream error:`, err?.message || err);
      try {
        res.end();
      } catch {}
    });

    // If client disconnects, stop upstream
    req.on('close', () => {
      try {
        upstream.data.destroy();
      } catch {}
    });
  } catch (error) {
    const status = error.response?.status || 500;
    const upstreamData = error.response?.data;

    // Best-effort extract upstream error message
    let upstreamMsg = '';
    try {
      upstreamMsg =
        upstreamData?.error?.message ||
        upstreamData?.message ||
        (typeof upstreamData === 'string' ? upstreamData : '');
    } catch {}

    console.error(
      `[${req._requestId}] Proxy error:`,
      status,
      upstreamMsg || error.message
    );

    res.status(status).json({
      error: {
        message: upstreamMsg || error.message || 'Internal server error',
        type: 'invalid_request_error',
        code: status
      }
    });
  }
}

// Catch-all for unsupported endpoints (includes method in message to reduce confusion)
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.method} ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

app.listen(PORT, () => {
  console.log(`OpenAI -> NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`NIM base: ${NIM_API_BASE}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Debug logging: ${DEBUG_LOG_REQUESTS ? 'ENABLED' : 'DISABLED'}`);
});
