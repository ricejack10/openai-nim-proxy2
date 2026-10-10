// server.js
// OpenAI-compatible proxy for NVIDIA NIM
// Designed for JanitorAI / OpenAI-compatible clients.
//
// Environment variables:
//   NIM_API_KEY   = your NVIDIA API key (required)
//   PORT          = local port (default 3000)
//   NIM_API_BASE  = NVIDIA API base (default https://integrate.api.nvidia.com/v1)
//
// Important:
// - No automatic retries.
// - No fallback-model attempts.
// - Each NVIDIA request has a 3-minute timeout.
// - This proxy does NOT impose max_tokens/min_tokens.
// - Client-supplied supported parameters are forwarded unchanged.
// - Model IDs can be passed directly, or aliases below can be used.

'use strict';

const express = require('express');
const cors = require('cors');
const axios = require('axios');
const crypto = require('crypto');

const app = express();

const PORT = Number(process.env.PORT || 3000);
const NIM_API_BASE = (
  process.env.NIM_API_BASE ||
  'https://integrate.api.nvidia.com/v1'
).replace(/\/+$/, '');

const NIM_API_KEY = process.env.NIM_API_KEY;

// Fixed 3-minute timeout. No retries or fallback attempts.
const REQUEST_TIMEOUT_MS = 180000;

// -----------------------------------------------------------------------------
// Model aliases
// -----------------------------------------------------------------------------
// Janitor can use either an alias or the actual NVIDIA model ID.
// Direct model IDs are passed through automatically.

const MODEL_MAPPING = {
  // DeepSeek V4.1 Flash
  'deepseek-v4.1-flash': 'deepseek-ai/deepseek-v4.1-flash',
  'deepseek-v4_1-flash': 'deepseek-ai/deepseek-v4.1-flash',

  // Existing aliases
  'kimi-k2': 'moonshotai/kimi-k2-instruct',
  'mistral-large': 'mistralai/mistral-large-3-675b-instruct-2512',
  'llama-405b': 'meta/llama-3.1-405b-instruct',
  'nemotron-49b': 'nvidia/llama-3.3-nemotron-super-49b-v1',
  'seed-36b': 'bytedance/seed-oss-36b-instruct',
  'magistral': 'mistralai/magistral-small-2506',
  'qwen-coder': 'qwen/qwen3-coder-480b-a35b-instruct',
  'glm-5': 'z-ai/glm-5.2',
};

// -----------------------------------------------------------------------------
// Request parameters
// -----------------------------------------------------------------------------
// Only parameters actually supplied by the client are forwarded.
// In particular, max_tokens/min_tokens are NEVER added by this proxy.
//
// If the client omits max_tokens, NVIDIA/model defaults are used.
// No temperature or top_p defaults are imposed by this proxy.

const FORWARDED_PARAMS = [
  'temperature',
  'top_p',
  'max_tokens',
  'stop',
  'frequency_penalty',
  'presence_penalty',
  'seed',
  'response_format',
  'tools',
  'tool_choice',
  'parallel_tool_calls',
  'user',
  'stream_options',
];

// -----------------------------------------------------------------------------
// Express setup
// -----------------------------------------------------------------------------

app.disable('x-powered-by');
app.use(cors());
app.use(express.json({ limit: '25mb' }));

// -----------------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------------

function now() {
  return new Date().toISOString();
}

function reqId() {
  return crypto.randomBytes(4).toString('hex');
}

function preview(value, limit = 500) {
  if (typeof value !== 'string') {
    try {
      value = JSON.stringify(value);
    } catch {
      value = String(value);
    }
  }

  return value.length > limit
    ? value.slice(0, limit) + '...'
    : value;
}

function contentPreview(content) {
  if (typeof content === 'string') {
    return preview(content, 300);
  }

  if (Array.isArray(content)) {
    const text = content
      .filter(part => part && part.type === 'text')
      .map(part => part.text || '')
      .join(' ');

    return preview(text || '[non-text content]', 300);
  }

  if (content == null) {
    return '[empty]';
  }

  return '[non-text content]';
}

function stripThinkingFull(text) {
  if (typeof text !== 'string') return '';

  // Remove complete <think> blocks.
  text = text.replace(/<think>[\s\S]*?<\/think>/gi, '');

  // Remove an unclosed think block at the end.
  text = text.replace(/<think>[\s\S]*/gi, '');

  return text.trimStart();
}

function makeThinkState() {
  return {
    phase: 'pass',
    pending: '',
  };
}

function trailingOverlap(haystack, needle) {
  let best = '';

  for (let len = 1; len <= Math.min(haystack.length, needle.length); len++) {
    if (haystack.slice(haystack.length - len) === needle.slice(0, len)) {
      best = needle.slice(0, len);
    }
  }

  return best;
}

// Streaming <think> stripper.
// It keeps tag fragments between chunks so split tags are handled correctly.
function processThinkChunk(raw, state) {
  if (!raw) return '';

  let input = state.pending + raw;
  state.pending = '';

  let output = '';

  while (input.length > 0) {
    if (state.phase === 'pass') {
      const startTag = '<think>';
      const lower = input.toLowerCase();
      const idx = lower.indexOf(startTag);

      if (idx === -1) {
        const overlap = trailingOverlap(input, startTag);

        output += input.slice(0, input.length - overlap.length);
        state.pending = overlap;
        input = '';
      } else {
        output += input.slice(0, idx);
        input = input.slice(idx + startTag.length);
        state.phase = 'think';
      }
    } else {
      const endTag = '</think>';
      const lower = input.toLowerCase();
      const idx = lower.indexOf(endTag);

      if (idx === -1) {
        state.pending = trailingOverlap(input, endTag);
        input = '';
      } else {
        input = input
          .slice(idx + endTag.length)
          .replace(/^\s+/, '');

        state.phase = 'pass';
      }
    }
  }

  return output;
}

function resolveModel(model) {
  return MODEL_MAPPING[model] || model;
}

// -----------------------------------------------------------------------------
// NVIDIA request
// -----------------------------------------------------------------------------
// Each incoming request gets exactly one NVIDIA request attempt.
// There is no retry loop and no fallback to another model.

async function callNIM(body) {
  const isStream = body.stream === true;

  return axios.post(
    `${NIM_API_BASE}/chat/completions`,
    body,
    {
      headers: {
        Authorization: `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json',
        Accept: isStream ? 'text/event-stream' : 'application/json',
      },
      responseType: isStream ? 'stream' : 'json',
      timeout: REQUEST_TIMEOUT_MS,
      validateStatus: status => status >= 200 && status < 300,
    }
  );
}

// -----------------------------------------------------------------------------
// Error body handling
// -----------------------------------------------------------------------------

async function readStreamToString(stream, timeoutMs = 3000) {
  if (!stream || typeof stream.on !== 'function') {
    return '';
  }

  return new Promise(resolve => {
    const chunks = [];
    let finished = false;

    const finish = () => {
      if (finished) return;
      finished = true;
      clearTimeout(timer);

      try {
        resolve(Buffer.concat(chunks).toString('utf8'));
      } catch {
        resolve('');
      }
    };

    const timer = setTimeout(finish, timeoutMs);

    stream.on('data', chunk => {
      chunks.push(
        Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk))
      );
    });

    stream.on('end', finish);
    stream.on('close', finish);
    stream.on('error', finish);
  });
}

async function getAxiosErrorDetail(err, status) {
  if (err?.code === 'ECONNABORTED' || err?.code === 'ETIMEDOUT') {
    return `NVIDIA request timed out after ${REQUEST_TIMEOUT_MS / 1000}s`;
  }

  if (err?.code === 'ECONNRESET') {
    return 'NVIDIA closed the connection unexpectedly';
  }

  const raw = err?.response?.data;

  if (!raw) {
    return err?.message || `NVIDIA returned HTTP ${status || 500}`;
  }

  if (typeof raw === 'string') {
    return raw;
  }

  if (Buffer.isBuffer(raw)) {
    return raw.toString('utf8');
  }

  if (raw && typeof raw.on === 'function') {
    const text = await readStreamToString(raw);

    if (text) {
      return text;
    }

    return `[Empty stream body — HTTP ${status}]`;
  }

  if (typeof raw === 'object') {
    try {
      return JSON.stringify(raw);
    } catch {
      return `[Unserializable NVIDIA error — HTTP ${status}]`;
    }
  }

  return String(raw);
}

function sendProxyError(res, status, detail) {
  const httpStatus = Number.isInteger(status) && status >= 400
    ? status
    : 500;

  return res.status(httpStatus).json({
    error: {
      message: detail || 'An error occurred communicating with NVIDIA NIM',
      type: httpStatus >= 500 ? 'server_error' : 'invalid_request_error',
      code: httpStatus,
    },
  });
}

async function handleAxiosError(err, id, res, model) {
  const status = err?.response?.status;
  const detail = await getAxiosErrorDetail(err, status);

  console.error(`[${now()}] [${id}] NVIDIA ERROR`);
  console.error(`  Model  : ${model}`);
  console.error(`  HTTP   : ${status || 'N/A'}`);
  console.error(`  Code   : ${err?.code || 'N/A'}`);
  console.error(`  Detail : ${preview(detail, 2000)}`);

  if (!res.headersSent) {
    return sendProxyError(res, status, detail);
  }

  // Headers have already been sent, usually because streaming started.
  if (!res.writableEnded) {
    try {
      res.write(
        `data: ${JSON.stringify({
          error: {
            message: detail || 'NVIDIA stream failed',
            type: 'stream_error',
            code: status || 500,
          },
        })}\n\n`
      );

      res.write('data: [DONE]\n\n');
      res.end();
    } catch {
      try {
        res.end();
      } catch {}
    }
  }
}

// -----------------------------------------------------------------------------
// Health / model endpoints
// -----------------------------------------------------------------------------

app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    nim_base: NIM_API_BASE,
    api_key_set: Boolean(NIM_API_KEY),
    models: Object.keys(MODEL_MAPPING).length,
    request_timeout_ms: REQUEST_TIMEOUT_MS,
    retries: 'disabled',
    fallback_models: 'disabled',
    token_limits: 'none imposed by proxy',
  });
});

app.get('/v1/models', (req, res) => {
  const data = Object.entries(MODEL_MAPPING).map(([id, mappedModel]) => ({
    id,
    object: 'model',
    created: 1700000000,
    owned_by: 'nvidia-nim-proxy',
    root: mappedModel,
  }));

  res.json({
    object: 'list',
    data,
  });
});

// -----------------------------------------------------------------------------
// Main OpenAI-compatible endpoint
// -----------------------------------------------------------------------------

app.post('/v1/chat/completions', async (req, res) => {
  const id = reqId();

  if (!NIM_API_KEY) {
    console.error(`[${now()}] [${id}] NIM_API_KEY is not set`);

    return res.status(500).json({
      error: {
        message: 'NIM_API_KEY environment variable is not set',
        type: 'server_error',
        code: 500,
      },
    });
  }

  const {
    model,
    messages,
    stream,
  } = req.body || {};

  // Basic validation.
  if (!model || typeof model !== 'string') {
    return res.status(400).json({
      error: {
        message: 'Request body must include a "model" string',
        type: 'invalid_request_error',
        code: 400,
      },
    });
  }

  if (!Array.isArray(messages) || messages.length === 0) {
    return res.status(400).json({
      error: {
        message: 'Request body must include a non-empty "messages" array',
        type: 'invalid_request_error',
        code: 400,
      },
    });
  }

  for (let i = 0; i < messages.length; i++) {
    const message = messages[i];

    if (!message || typeof message.role !== 'string') {
      return res.status(400).json({
        error: {
          message: `messages[${i}] is missing a "role" field`,
          type: 'invalid_request_error',
          code: 400,
        },
      });
    }
  }

  const requestedModel = model;
  const activeModel = resolveModel(model);
  const isStream = stream === true;

  console.log(`\n${'─'.repeat(70)}`);
  console.log(`[${now()}] [${id}] REQUEST`);
  console.log(`  Model    : ${requestedModel} -> ${activeModel}`);
  console.log(`  Stream   : ${isStream}`);
  console.log(`  Messages : ${messages.length}`);

  if (req.body.max_tokens !== undefined) {
    console.log(`  max_tokens: ${req.body.max_tokens} (client supplied)`);
  } else {
    console.log('  max_tokens: omitted (NVIDIA/model default)');
  }

  messages.forEach((msg, i) => {
    console.log(
      `  [${i}] ${(msg.role || 'unknown').toUpperCase()}: ${contentPreview(msg.content)}`
    );
  });

  // Build a request without adding token limits or sampling defaults.
  const nimBody = {
    model: activeModel,
    messages,
    stream: isStream,
  };

  for (const param of FORWARDED_PARAMS) {
    if (req.body[param] !== undefined) {
      nimBody[param] = req.body[param];
    }
  }

  // ---------------------------------------------------------------------------
  // One NVIDIA request attempt only — no retries or fallback models.
  // ---------------------------------------------------------------------------

  let response;

  try {
    response = await callNIM(nimBody);
  } catch (err) {
    return handleAxiosError(err, id, res, activeModel);
  }

  // ---------------------------------------------------------------------------
  // Streaming response
  // ---------------------------------------------------------------------------

  if (isStream) {
    res.status(200);
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache, no-transform');
    res.setHeader('Connection', 'keep-alive');
    res.setHeader('X-Accel-Buffering', 'no');

    if (typeof res.flushHeaders === 'function') {
      res.flushHeaders();
    }

    console.log(`[${now()}] [${id}] RESPONSE streaming...`);

    let lineBuffer = '';
    let logAccumulator = '';
    let streamEnded = false;

    const thinkState = makeThinkState();

    response.data.on('data', chunk => {
      if (streamEnded) return;

      lineBuffer += chunk.toString();

      const lines = lineBuffer.split('\n');
      lineBuffer = lines.pop() ?? '';

      for (const line of lines) {
        const trimmed = line.trim();

        if (!trimmed) {
          continue;
        }

        // NVIDIA/OpenAI SSE terminator.
        if (trimmed === 'data: [DONE]') {
          res.write('data: [DONE]\n\n');
          continue;
        }

        // Preserve SSE comments/other lines rather than trying to parse them.
        if (!trimmed.startsWith('data: ')) {
          res.write(line + '\n\n');
          continue;
        }

        let parsed;

        try {
          parsed = JSON.parse(trimmed.slice(6));
        } catch {
          // If a provider sends malformed JSON, don't crash the stream.
          console.warn(
            `[${now()}] [${id}] Could not parse SSE JSON: ${preview(trimmed, 500)}`
          );

          res.write(line + '\n\n');
          continue;
        }

        const choice = parsed.choices?.[0];

        if (!choice) {
          res.write(`data: ${JSON.stringify(parsed)}\n\n`);
          continue;
        }

        const delta = choice.delta;

        if (!delta) {
          res.write(`data: ${JSON.stringify(parsed)}\n\n`);
          continue;
        }

        // Don't expose provider reasoning_content to JanitorAI.
        delete delta.reasoning_content;

        let content = typeof delta.content === 'string'
          ? delta.content
          : '';

        // Remove <think> blocks if the provider emits them inside content.
        content = processThinkChunk(content, thinkState);

        if (content) {
          delta.content = content;
          logAccumulator += content;

          res.write(`data: ${JSON.stringify(parsed)}\n\n`);
          continue;
        }

        // Forward tool calls / function calls / finish reasons even if
        // delta.content is empty.
        if (
          choice.finish_reason !== undefined &&
          choice.finish_reason !== null
        ) {
          delta.content = '';
          res.write(`data: ${JSON.stringify(parsed)}\n\n`);
          continue;
        }

        if (
          delta.tool_calls ||
          delta.function_call ||
          delta.role
        ) {
          res.write(`data: ${JSON.stringify(parsed)}\n\n`);
        }
      }
    });

    response.data.on('end', () => {
      if (streamEnded) return;
      streamEnded = true;

      // Flush a possible partial "<think" boundary.
      if (thinkState.pending && thinkState.phase === 'pass') {
        const flush = thinkState.pending;

        const flushChunk = {
          id: `chatcmpl-${Date.now()}`,
          object: 'chat.completion.chunk',
          created: Math.floor(Date.now() / 1000),
          model: requestedModel,
          choices: [
            {
              index: 0,
              delta: {
                content: flush,
              },
              finish_reason: null,
            },
          ],
        };

        res.write(`data: ${JSON.stringify(flushChunk)}\n\n`);
        logAccumulator += flush;
      }

      console.log(`[${now()}] [${id}] STREAM END`);
      console.log(`  ASSISTANT: ${preview(logAccumulator, 1000)}`);

      if (!res.writableEnded) {
        res.end();
      }
    });

    response.data.on('error', err => {
      if (streamEnded) return;
      streamEnded = true;

      console.error(
        `[${now()}] [${id}] NVIDIA STREAM ERROR: ${err.message}`
      );

      if (!res.writableEnded) {
        try {
          res.write(
            `data: ${JSON.stringify({
              error: {
                message: `NVIDIA stream error: ${err.message}`,
                type: 'stream_error',
              },
            })}\n\n`
          );

          res.write('data: [DONE]\n\n');
        } catch {}

        res.end();
      }
    });

    // Destroy the upstream stream if the client disconnects.
    res.on('close', () => {
      if (!streamEnded && response?.data && !response.data.destroyed) {
        console.log(`[${now()}] [${id}] CLIENT DISCONNECTED`);

        try {
          response.data.destroy();
        } catch {}
      }
    });

    return;
  }

  // ---------------------------------------------------------------------------
  // Non-streaming response
  // ---------------------------------------------------------------------------

  const choices = (response.data?.choices || []).map(choice => {
    const msg = choice.message || {};

    let content = msg.content ?? '';

    if (typeof content === 'string') {
      content = stripThinkingFull(content);
    }

    const outputMessage = {
      role: msg.role || 'assistant',
      content,
    };

    // Preserve tool calls when present.
    if (msg.tool_calls !== undefined) {
      outputMessage.tool_calls = msg.tool_calls;
    }

    if (msg.function_call !== undefined) {
      outputMessage.function_call = msg.function_call;
    }

    return {
      index: choice.index ?? 0,
      message: outputMessage,
      finish_reason: choice.finish_reason ?? null,
    };
  });

  const usage = response.data?.usage || {
    prompt_tokens: 0,
    completion_tokens: 0,
    total_tokens: 0,
  };

  console.log(`[${now()}] [${id}] RESPONSE (non-stream)`);

  choices.forEach((choice, i) => {
    console.log(
      `  [${i}] ASSISTANT: ${preview(choice.message.content, 1000)}`
    );
  });

  console.log(
    `  Usage: prompt=${usage.prompt_tokens ?? 0} ` +
    `completion=${usage.completion_tokens ?? 0} ` +
    `total=${usage.total_tokens ?? 0}`
  );

  return res.json({
    id: `chatcmpl-${Date.now()}`,
    object: 'chat.completion',
    created: Math.floor(Date.now() / 1000),
    model: requestedModel,
    choices,
    usage,
  });
});

// -----------------------------------------------------------------------------
// 404 catch-all
// -----------------------------------------------------------------------------

app.use((req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} is not supported by this proxy`,
      type: 'not_found',
      code: 404,
    },
  });
});

// -----------------------------------------------------------------------------
// Final Express error handler
// -----------------------------------------------------------------------------

app.use((err, req, res, next) => {
  console.error(`[${now()}] EXPRESS ERROR:`, err);

  if (res.headersSent) {
    return next(err);
  }

  res.status(500).json({
    error: {
      message: err?.message || 'Internal proxy error',
      type: 'server_error',
      code: 500,
    },
  });
});

// -----------------------------------------------------------------------------
// Start
// -----------------------------------------------------------------------------

app.listen(PORT, () => {
  console.log('');
  console.log('OpenAI -> NVIDIA NIM Proxy');
  console.log(`  Port             : ${PORT}`);
  console.log(`  Health           : http://localhost:${PORT}/health`);
  console.log(`  Models           : http://localhost:${PORT}/v1/models`);
  console.log(`  NVIDIA API       : ${NIM_API_BASE}`);
  console.log(`  API key          : ${NIM_API_KEY ? 'SET' : 'NOT SET'}`);
  console.log(`  Request timeout  : ${REQUEST_TIMEOUT_MS}ms`);
  console.log('  Retries          : DISABLED');
  console.log('  Fallback models  : DISABLED');
  console.log('  Token limits     : NONE imposed by proxy');
  console.log('');
});
