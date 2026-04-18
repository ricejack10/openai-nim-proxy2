// server.js - OpenAI to NVIDIA NIM API Proxy
'use strict';

const express = require('express');
const cors    = require('cors');
const axios   = require('axios');
const crypto  = require('crypto');

const app  = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '10mb' }));

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY  = process.env.NIM_API_KEY;

// How many times to retry a request when NIM returns 429, 503, or 504.
// These are transient congestion errors — a short wait and one retry
// resolves most of them without the caller ever seeing an error.
const MAX_RETRIES      = 2;
const RETRY_DELAY_MS   = 2000; // 2 seconds between retries

// OpenAI alias -> NVIDIA NIM model ID.
//
// Last verified: April 16 2026.
// Confirmed live:
//   deepseek-ai/deepseek-v3.1-terminus  — best quality available
//   meta/llama-3.3-70b-instruct         — stable lightweight fallback
//
// To re-enable reasoning models when new ones appear on the hosted API,
// add them to NATIVE_THINKERS below and the think-stripping code will
// activate automatically.
const MODEL_MAPPING = {
  // Primary — best quality
  'gpt-4':           'deepseek-ai/deepseek-v3.1-terminus',
  'gpt-4-turbo':     'deepseek-ai/deepseek-v3.1-terminus',
  'gpt-4o':          'deepseek-ai/deepseek-v3.1-terminus',
  'claude-3-opus':   'deepseek-ai/deepseek-v3.1-terminus',
  'claude-3-sonnet': 'deepseek-ai/deepseek-v3.1-terminus',

  // Secondary — faster, lighter, good for long conversations
  'gpt-3.5-turbo':   'meta/llama-3.3-70b-instruct',
  'gpt-4o-mini':     'meta/llama-3.3-70b-instruct',
  'claude-3-haiku':  'meta/llama-3.3-70b-instruct',
  'gemini-pro':      'meta/llama-3.3-70b-instruct',
};

// Models that natively emit <think>...</think> blocks which must be stripped.
// Currently empty because all R1 distill models were retired on April 15 2026.
// Add future reasoning models here to re-enable stripping.
const NATIVE_THINKERS = new Set([]);

// Generation parameters forwarded verbatim to NIM when present on the request.
const FORWARDED_PARAMS = [
  'temperature', 'top_p', 'top_k', 'min_p',
  'max_tokens', 'stop',
  'frequency_penalty', 'presence_penalty',
  'seed', 'n', 'stream_options',
];

// HTTP status codes that indicate transient server congestion and are safe to retry.
const RETRYABLE_STATUSES = new Set([429, 503, 504]);

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

function now() {
  return new Date().toISOString();
}

function reqId() {
  return crypto.randomBytes(4).toString('hex');
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Truncate a value to a string for log output.
function preview(str, limit = 300) {
  if (typeof str !== 'string') str = String(str);
  return str.length > limit ? str.slice(0, limit) + '...' : str;
}

// Extract a plain-text preview from a message content value.
// JanitorAI sometimes sends content as an array of typed content-part objects.
function contentPreview(content) {
  if (typeof content === 'string') return preview(content);
  if (Array.isArray(content)) {
    const text = content
      .filter(p => p && p.type === 'text')
      .map(p => p.text || '')
      .join(' ');
    return preview(text || '[non-text content]');
  }
  return '[non-text content]';
}

// Strip <think>...</think> blocks from a fully buffered string.
// Handles multiple think blocks and unclosed tags at the end of the string.
function stripThinkingFull(text) {
  text = text.replace(/<think>[\s\S]*?<\/think>/gi, '');
  text = text.replace(/<think>[\s\S]*/gi, '');
  return text.trimStart();
}

// ---------------------------------------------------------------------------
// Streaming think-stripper — state machine
//
// Processes content one chunk at a time. Returns the portion safe to forward.
// The caller must pass the same state object on every subsequent call for
// the same response so that tag boundaries split across chunks are handled.
// ---------------------------------------------------------------------------

function makeThinkState() {
  return { phase: 'pass', pending: '' };
}

// Returns the longest prefix of `needle` that is also a suffix of `haystack`.
// Used to hold back bytes at the end of a chunk that might be the start of a tag.
function trailingOverlap(haystack, needle) {
  let best = '';
  for (let len = 1; len <= Math.min(haystack.length, needle.length); len++) {
    if (haystack.slice(haystack.length - len) === needle.slice(0, len)) {
      best = needle.slice(0, len);
    }
  }
  return best;
}

function processThinkChunk(raw, state) {
  if (!raw) return '';

  let input  = state.pending + raw;
  state.pending = '';
  let output = '';

  while (input.length > 0) {
    if (state.phase === 'pass') {
      const startTag = '<think>';
      const idx = input.toLowerCase().indexOf(startTag);
      if (idx === -1) {
        const overlap = trailingOverlap(input, startTag);
        output       += input.slice(0, input.length - overlap.length);
        state.pending = overlap;
        input = '';
      } else {
        output += input.slice(0, idx);
        input   = input.slice(idx + startTag.length);
        state.phase = 'think';
      }
    } else {
      const endTag = '</think>';
      const idx = input.toLowerCase().indexOf(endTag);
      if (idx === -1) {
        state.pending = trailingOverlap(input, endTag);
        input = '';
      } else {
        input       = input.slice(idx + endTag.length).replace(/^\s+/, '');
        state.phase = 'pass';
      }
    }
  }

  return output;
}

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------

app.get('/health', (req, res) => {
  res.json({
    status:      'ok',
    service:     'OpenAI to NVIDIA NIM Proxy',
    nim_base:    NIM_API_BASE,
    api_key_set: !!NIM_API_KEY,
    models:      Object.keys(MODEL_MAPPING).length,
    max_retries: MAX_RETRIES,
  });
});

app.get('/v1/models', (req, res) => {
  const data = Object.keys(MODEL_MAPPING).map(id => ({
    id,
    object:   'model',
    created:  1700000000,
    owned_by: 'nvidia-nim-proxy',
  }));
  res.json({ object: 'list', data });
});

// ---------------------------------------------------------------------------
// NIM request helper with retry logic
// ---------------------------------------------------------------------------

async function callNIM(nimBody, isStream, attemptNum, id) {
  return axios.post(
    `${NIM_API_BASE}/chat/completions`,
    nimBody,
    {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type':  'application/json',
        'Accept':        isStream ? 'text/event-stream' : 'application/json',
      },
      responseType: isStream ? 'stream' : 'json',
      timeout:      120000,
    }
  );
}

// ---------------------------------------------------------------------------
// Main proxy endpoint
// ---------------------------------------------------------------------------

app.post('/v1/chat/completions', async (req, res) => {
  const id = reqId();

  // --- API key guard --------------------------------------------------------
  if (!NIM_API_KEY) {
    console.error(`[${now()}] [${id}] FATAL: NIM_API_KEY is not set`);
    return res.status(500).json({
      error: { message: 'NIM_API_KEY environment variable is not set', type: 'server_error', code: 500 },
    });
  }

  // --- Request validation ---------------------------------------------------
  const { model, messages, stream } = req.body;

  if (!model || typeof model !== 'string') {
    return res.status(400).json({
      error: { message: 'Request body must include a "model" string', type: 'invalid_request_error', code: 400 },
    });
  }

  if (!Array.isArray(messages) || messages.length === 0) {
    return res.status(400).json({
      error: { message: 'Request body must include a non-empty "messages" array', type: 'invalid_request_error', code: 400 },
    });
  }

  for (let i = 0; i < messages.length; i++) {
    if (!messages[i] || typeof messages[i].role !== 'string') {
      return res.status(400).json({
        error: { message: `messages[${i}] is missing a "role" field`, type: 'invalid_request_error', code: 400 },
      });
    }
  }

  // --- Model resolution -----------------------------------------------------
  const nimModel        = MODEL_MAPPING[model] || model;
  const isNativeThinker = NATIVE_THINKERS.has(nimModel);
  const isStream        = !!stream;

  // --- Log the incoming request ---------------------------------------------
  console.log(`\n${'─'.repeat(60)}`);
  console.log(`[${now()}] [${id}] REQUEST`);
  console.log(`  Model    : ${model} -> ${nimModel}`);
  console.log(`  Stream   : ${isStream}`);
  console.log(`  Messages : ${messages.length}`);
  messages.forEach((msg, i) => {
    console.log(`  [${i}] ${msg.role.toUpperCase()}: ${contentPreview(msg.content)}`);
  });

  // --- Build NIM request body -----------------------------------------------
  const nimBody = { model: nimModel, messages, stream: isStream };

  for (const param of FORWARDED_PARAMS) {
    if (req.body[param] !== undefined) nimBody[param] = req.body[param];
  }

  if (nimBody.max_tokens   === undefined) nimBody.max_tokens   = 4096;
  if (nimBody.temperature  === undefined) nimBody.temperature  = 0.6;

  // --- Call NIM with retry on transient errors ------------------------------
  // Streaming responses cannot be retried once the response has begun, so
  // retry only applies to the initial connection attempt.
  let response;
  let lastErr;

  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      response = await callNIM(nimBody, isStream, attempt, id);
      lastErr  = null;
      break;
    } catch (err) {
      lastErr = err;
      const status = err.response?.status;

      if (attempt < MAX_RETRIES && RETRYABLE_STATUSES.has(status)) {
        const wait = RETRY_DELAY_MS * (attempt + 1);
        console.warn(`[${now()}] [${id}] NIM returned ${status}, retrying in ${wait}ms (attempt ${attempt + 1}/${MAX_RETRIES})`);
        await sleep(wait);
        continue;
      }

      // Non-retryable error or retries exhausted
      break;
    }
  }

  if (lastErr) {
    return handleAxiosError(lastErr, id, res);
  }

  // =========================================================================
  // Streaming response
  // =========================================================================
  if (isStream) {
    res.setHeader('Content-Type',      'text/event-stream');
    res.setHeader('Cache-Control',     'no-cache');
    res.setHeader('Connection',        'keep-alive');
    res.setHeader('X-Accel-Buffering', 'no');

    console.log(`[${now()}] [${id}] RESPONSE streaming...`);

    let lineBuffer     = '';
    let logAccumulator = '';
    const thinkState   = isNativeThinker ? makeThinkState() : null;

    response.data.on('data', (chunk) => {
      lineBuffer += chunk.toString();
      const lines = lineBuffer.split('\n');
      lineBuffer  = lines.pop() ?? '';

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;

        if (trimmed === 'data: [DONE]') {
          res.write('data: [DONE]\n\n');
          continue;
        }

        if (!trimmed.startsWith('data: ')) {
          res.write(line + '\n');
          continue;
        }

        let parsed;
        try {
          parsed = JSON.parse(trimmed.slice(6));
        } catch (_) {
          res.write(line + '\n');
          continue;
        }

        const delta = parsed.choices?.[0]?.delta;
        if (!delta) {
          res.write(`data: ${JSON.stringify(parsed)}\n\n`);
          continue;
        }

        delete delta.reasoning_content;

        let content = typeof delta.content === 'string' ? delta.content : '';

        if (isNativeThinker) {
          content = processThinkChunk(content, thinkState);
        }

        if (content || !isNativeThinker) {
          delta.content   = content;
          logAccumulator += content;
          res.write(`data: ${JSON.stringify(parsed)}\n\n`);
        }
      }
    });

    response.data.on('end', () => {
      // Flush any think-state pending buffer (trailing partial tag in pass phase)
      if (isNativeThinker && thinkState.pending && thinkState.phase === 'pass') {
        const flush = thinkState.pending;
        if (flush) {
          const flushChunk = {
            id:      `chatcmpl-${Date.now()}`,
            object:  'chat.completion.chunk',
            created: Math.floor(Date.now() / 1000),
            model,
            choices: [{ index: 0, delta: { content: flush }, finish_reason: null }],
          };
          res.write(`data: ${JSON.stringify(flushChunk)}\n\n`);
          logAccumulator += flush;
        }
      }

      console.log(`[${now()}] [${id}] STREAM END`);
      console.log(`  ASSISTANT: ${preview(logAccumulator, 500)}`);
      res.end();
    });

    response.data.on('error', (err) => {
      console.error(`[${now()}] [${id}] STREAM ERROR: ${err.message}`);
      if (!res.writableEnded) {
        res.write(`data: ${JSON.stringify({ error: { message: `Stream error: ${err.message}`, type: 'stream_error' } })}\n\n`);
        res.write('data: [DONE]\n\n');
        res.end();
      }
    });

    return;
  }

  // =========================================================================
  // Non-streaming response
  // =========================================================================
  const choices = (response.data.choices || []).map(choice => {
    const msg = choice.message ?? {};
    let content = msg.content ?? '';

    if (isNativeThinker) content = stripThinkingFull(content);
    delete msg.reasoning_content;

    return {
      index:         choice.index ?? 0,
      message:       { role: msg.role ?? 'assistant', content },
      finish_reason: choice.finish_reason ?? null,
    };
  });

  const usage = response.data.usage ?? { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 };

  console.log(`[${now()}] [${id}] RESPONSE (non-stream)`);
  choices.forEach((c, i) => console.log(`  [${i}] ASSISTANT: ${preview(c.message.content)}`));
  console.log(`  Usage: prompt=${usage.prompt_tokens} completion=${usage.completion_tokens} total=${usage.total_tokens}`);

  return res.json({
    id:      `chatcmpl-${Date.now()}`,
    object:  'chat.completion',
    created: Math.floor(Date.now() / 1000),
    model,
    choices,
    usage,
  });
});

// ---------------------------------------------------------------------------
// 404 catch-all
// ---------------------------------------------------------------------------

app.all('*', (req, res) => {
  res.status(404).json({
    error: { message: `Endpoint ${req.path} is not supported by this proxy`, type: 'not_found', code: 404 },
  });
});

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

function handleAxiosError(err, id, res) {
  const status = err.response?.status;
  let detail   = err.message;

  // Distinguish timeout and connection reset from API errors for clearer logs
  if (err.code === 'ECONNABORTED') {
    detail = `Request timed out after ${Math.round(120000 / 1000)}s with no response from NIM`;
  } else if (err.code === 'ECONNRESET') {
    detail = 'NIM closed the connection unexpectedly (likely server-side congestion)';
  } else if (err.response?.data) {
    const raw = err.response.data;
    if (typeof raw === 'string')       detail = raw;
    else if (Buffer.isBuffer(raw))     detail = raw.toString('utf8');
    else if (typeof raw === 'object')  detail = JSON.stringify(raw);
  }

  const httpStatus = status || 500;
  console.error(`[${now()}] [${id}] PROXY ERROR [${httpStatus}]: ${preview(detail, 500)}`);

  if (!res.headersSent) {
    res.status(httpStatus).json({
      error: {
        message: detail || 'An error occurred communicating with the NIM API',
        type:    httpStatus >= 500 ? 'server_error' : 'invalid_request_error',
        code:    httpStatus,
      },
    });
  }
}

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------

app.listen(PORT, () => {
  console.log('');
  console.log('OpenAI to NVIDIA NIM Proxy');
  console.log(`  Port        : ${PORT}`);
  console.log(`  Health      : http://localhost:${PORT}/health`);
  console.log(`  API key     : ${NIM_API_KEY ? 'SET' : 'NOT SET -- set NIM_API_KEY environment variable'}`);
  console.log(`  Models      : ${Object.keys(MODEL_MAPPING).length} mapped`);
  console.log(`  Max retries : ${MAX_RETRIES} (on 429/503/504)`);
  console.log('');
});
