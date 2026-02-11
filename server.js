// server.js - OpenAI-compatible proxy to NVIDIA NIM (no model mapping)
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

if (!NIM_API_KEY) {
  console.warn('⚠️  NIM_API_KEY is not set. Requests will fail until you set it.');
}

// 🔥 REASONING DISPLAY TOGGLE - Shows/hides reasoning in output
const SHOW_REASONING = false; // Set true to include <think> tags

// 🔥 THINKING MODE TOGGLE - Enables thinking for models that support it
const ENABLE_THINKING_MODE = false; // Adds extra_body.chat_template_kwargs.thinking=true

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy (no model mapping)',
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE,
    nim_api_base: NIM_API_BASE
  });
});

// List models endpoint (OpenAI compatible)
// Since you provide the model name directly, we don't maintain a model list.
app.get('/v1/models', (req, res) => {
  res.json({ object: 'list', data: [] });
});

// Chat completions endpoint (main proxy)
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;

    if (!model) {
      return res.status(400).json({
        error: { message: 'Missing required field: model', type: 'invalid_request_error', code: 400 }
      });
    }
    if (!Array.isArray(messages)) {
      return res.status(400).json({
        error: { message: 'Missing/invalid field: messages (must be an array)', type: 'invalid_request_error', code: 400 }
      });
    }

    // Transform OpenAI request -> NIM request (model passed through as-is)
    const nimRequest = {
      model,
      messages,
      temperature: temperature ?? 0.6,
      max_tokens: max_tokens ?? 9024,
      stream: !!stream,
      ...(ENABLE_THINKING_MODE
        ? { extra_body: { chat_template_kwargs: { thinking: true } } }
        : {})
    };

    // Make request to NVIDIA NIM API
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        Authorization: `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json'
    });

    if (stream) {
      // Stream back as SSE (OpenAI-style)
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buffer = '';
      let reasoningStarted = false;

      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;

          if (line.includes('[DONE]')) {
            res.write(line + '\n\n');
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

                if (combined) {
                  delta.content = combined;
                } else {
                  delta.content = '';
                }
              } else {
                delta.content = content || '';
              }

              // Remove reasoning field so OpenAI clients don't choke
              delete delta.reasoning_content;
            }

            res.write(`data: ${JSON.stringify(data)}\n\n`);
          } catch (e) {
            // If parsing fails, just forward the line
            res.write(line + '\n\n');
          }
        }
      });

      response.data.on('end', () => res.end());
      response.data.on('error', (err) => {
        console.error('Stream error:', err);
        res.end();
      });

      return;
    }

    // Non-streaming: transform NIM response -> OpenAI format
    const openaiResponse = {
      id: `chatcmpl-${Date.now()}`,
      object: 'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model,
      choices: (response.data.choices || []).map((choice, i) => {
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
      usage: response.data.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
    };

    res.json(openaiResponse);
  } catch (error) {
    const status = error.response?.status || 500;
    const msg =
      error.response?.data?.error?.message ||
      error.response?.data?.message ||
      error.message ||
      'Internal server error';

    console.error('Proxy error:', status, msg);

    res.status(status).json({
      error: { message: msg, type: 'invalid_request_error', code: status }
    });
  }
});

// Catch-all for unsupported endpoints
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

app.listen(PORT, () => {
  console.log(`OpenAI->NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
});
