// server.js - OpenAI to NVIDIA NIM API Proxy (Fixed & Improved)
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '10mb' }));

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// 🔥 REASONING DISPLAY TOGGLE
// When true, <think>...</think> reasoning blocks are injected into the content
// so clients like JanitorAI can see the model's thought process
const SHOW_REASONING = process.env.SHOW_REASONING !== 'false'; // default ON

// 🔥 THINKING MODE TOGGLE
// When true, passes chat_template_kwargs: { thinking: true } to models that support it.
// Only enable for models that actually support extended thinking (e.g. Qwen3, DeepSeek-R1)
const ENABLE_THINKING_MODE = process.env.ENABLE_THINKING_MODE !== 'false'; // default ON

// Model mapping — maps OpenAI model names to NVIDIA NIM model IDs
// Models marked with [THINKING] support the thinking parameter
const MODEL_MAPPING = {
  'gpt-3.5-turbo':   'nvidia/llama-3.1-nemotron-ultra-253b-v1', // [THINKING]
  'gpt-4':           'qwen/qwen3-235b-a22b-instruct-2503',       // [THINKING]
  'gpt-4-turbo':     'deepseek-ai/deepseek-r1',                  // [THINKING]
  'gpt-4o':          'deepseek-ai/deepseek-v3',
  'gpt-4o-mini':     'meta/llama-3.3-70b-instruct',
  'claude-3-opus':   'nvidia/llama-3.1-nemotron-ultra-253b-v1', // [THINKING]
  'claude-3-sonnet': 'qwen/qwen3-235b-a22b-instruct-2503',      // [THINKING]
  'claude-3-haiku':  'meta/llama-3.3-70b-instruct',
  'gemini-pro':      'deepseek-ai/deepseek-r1',                  // [THINKING]
};

// Models that support the chat_template_kwargs thinking parameter
const THINKING_SUPPORTED_MODELS = new Set([
  'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'qwen/qwen3-235b-a22b-instruct-2503',
  'qwen/qwen3-coder-480b-a35b-instruct',
  'deepseek-ai/deepseek-r1',
  'deepseek-ai/deepseek-r1-0528',
]);

// Health check
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI → NVIDIA NIM Proxy',
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE,
    nim_base: NIM_API_BASE,
    api_key_set: !!NIM_API_KEY,
  });
});

// List models (OpenAI compatible)
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(id => ({
    id,
    object: 'model',
    created: 1700000000,
    owned_by: 'nvidia-nim-proxy',
  }));
  res.json({ object: 'list', data: models });
});

// ─── Main proxy endpoint ────────────────────────────────────────────────────
app.post('/v1/chat/completions', async (req, res) => {
  if (!NIM_API_KEY) {
    return res.status(500).json({ error: { message: 'NIM_API_KEY environment variable not set', type: 'server_error' } });
  }

  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;

    // Resolve model
    let nimModel = MODEL_MAPPING[model] || model;

    // Build NIM request body — chat_template_kwargs goes directly in the body, NOT in extra_body
    const nimRequestBody = {
      model: nimModel,
      messages: messages,
      temperature: temperature ?? 0.6,
      max_tokens: max_tokens || 16384,
      stream: stream || false,
    };

    // Only add thinking parameter for models that actually support it
    if (ENABLE_THINKING_MODE && THINKING_SUPPORTED_MODELS.has(nimModel)) {
      nimRequestBody.chat_template_kwargs = { thinking: true };
    }

    const response = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      nimRequestBody,
      {
        headers: {
          'Authorization': `Bearer ${NIM_API_KEY}`,
          'Content-Type': 'application/json',
          'Accept': stream ? 'text/event-stream' : 'application/json',
        },
        responseType: stream ? 'stream' : 'json',
        timeout: 120000,
      }
    );

    // ── Streaming ──────────────────────────────────────────────────────────
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      res.setHeader('X-Accel-Buffering', 'no');

      let buffer = '';
      let thinkOpen = false; // tracks whether we've opened a <think> tag

      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) continue;

          if (trimmed === 'data: [DONE]') {
            // If reasoning was streaming and never got a closing tag, close it now
            if (thinkOpen && SHOW_REASONING) {
              const closingChunk = buildDeltaChunk(model, '</think>\n\n');
              res.write(`data: ${JSON.stringify(closingChunk)}\n\n`);
              thinkOpen = false;
            }
            res.write('data: [DONE]\n\n');
            continue;
          }

          if (!trimmed.startsWith('data: ')) {
            res.write(trimmed + '\n');
            continue;
          }

          try {
            const data = JSON.parse(trimmed.slice(6));
            const delta = data.choices?.[0]?.delta;
            if (!delta) { res.write(`data: ${JSON.stringify(data)}\n\n`); continue; }

            const reasoningChunk = delta.reasoning_content ?? null;
            const contentChunk  = delta.content ?? null;

            // Remove reasoning_content from delta — we handle it ourselves
            delete delta.reasoning_content;

            if (SHOW_REASONING) {
              let inject = '';

              if (reasoningChunk) {
                if (!thinkOpen) {
                  inject += '<think>\n';
                  thinkOpen = true;
                }
                inject += reasoningChunk;
              }

              if (contentChunk !== null && contentChunk !== '') {
                if (thinkOpen) {
                  inject += '\n</think>\n\n';
                  thinkOpen = false;
                }
                inject += contentChunk;
              }

              delta.content = inject || (contentChunk === '' ? '' : null);
              // Remove null content to avoid issues
              if (delta.content === null) delete delta.content;
            } else {
              // Hide reasoning — just pass through content
              if (contentChunk !== null) delta.content = contentChunk;
              else delta.content = '';
            }

            res.write(`data: ${JSON.stringify(data)}\n\n`);
          } catch (e) {
            // Malformed JSON line — pass through as-is
            res.write(trimmed + '\n');
          }
        }
      });

      response.data.on('end', () => res.end());
      response.data.on('error', (err) => {
        console.error('Stream error:', err.message);
        res.end();
      });

    // ── Non-streaming ──────────────────────────────────────────────────────
    } else {
      const choices = response.data.choices.map(choice => {
        const msg = choice.message ?? {};
        let finalContent = msg.content ?? '';

        if (SHOW_REASONING && msg.reasoning_content) {
          finalContent = `<think>\n${msg.reasoning_content}\n</think>\n\n${finalContent}`;
        }

        return {
          index: choice.index,
          message: {
            role: msg.role ?? 'assistant',
            content: finalContent,
          },
          finish_reason: choice.finish_reason,
        };
      });

      res.json({
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices,
        usage: response.data.usage ?? { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
      });
    }

  } catch (error) {
    const status = error.response?.status || 500;
    const detail = error.response?.data || error.message;
    console.error(`Proxy error [${status}]:`, typeof detail === 'object' ? JSON.stringify(detail) : detail);

    if (!res.headersSent) {
      res.status(status).json({
        error: {
          message: typeof detail === 'object' ? JSON.stringify(detail) : (detail || 'Internal server error'),
          type: 'proxy_error',
          code: status,
        },
      });
    }
  }
});

// Helper: build a minimal SSE delta chunk for injecting content mid-stream
function buildDeltaChunk(model, content) {
  return {
    id: `chatcmpl-inject-${Date.now()}`,
    object: 'chat.completion.chunk',
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [{ index: 0, delta: { content }, finish_reason: null }],
  };
}

// 404 fallback
app.all('*', (req, res) => {
  res.status(404).json({ error: { message: `Endpoint ${req.path} not supported`, type: 'not_found', code: 404 } });
});

app.listen(PORT, () => {
  console.log(`\n🚀 OpenAI → NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`   Health:         http://localhost:${PORT}/health`);
  console.log(`   Reasoning:      ${SHOW_REASONING ? '✅ ENABLED' : '❌ DISABLED'} (set SHOW_REASONING=false to disable)`);
  console.log(`   Thinking mode:  ${ENABLE_THINKING_MODE ? '✅ ENABLED' : '❌ DISABLED'} (set ENABLE_THINKING_MODE=false to disable)`);
  console.log(`   NIM API key:    ${NIM_API_KEY ? '✅ SET' : '❌ NOT SET — set NIM_API_KEY env var!'}\n`);
});
