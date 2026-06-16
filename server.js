// server.js - OpenAI Compatible Proxy — OpenRouter backend
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

// Set OPENROUTER_API_KEY in Render's environment variables.
// Get a free key at openrouter.ai — no credit card required.
const API_BASE = process.env.NIM_API_BASE || 'https://openrouter.ai/api/v1';
const API_KEY  = process.env.NIM_API_KEY;

const MAX_RETRIES    = 2;
const RETRY_DELAY_MS = 1000;

// OpenRouter free tier allows 20 RPM — 3 second spacing is enough.
// Much more generous than NIM's 5 RPM.
const MIN_REQUEST_INTERVAL_MS = 3000;

// Tracks when the last request was dispatched.
let lastRequestTime = 0;

async function waitForRateLimit(id) {
  const elapsed = Date.now() - lastRequestTime;
  const wait    = MIN_REQUEST_INTERVAL_MS - elapsed;
  if (wait > 0) {
    console.log(`[${now()}] [${id}] Rate limiter: waiting ${Math.round(wait / 1000)}s`);
    await sleep(wait);
  }
  lastRequestTime = Date.now();
}

const REQUEST_TIMEOUT_MS = 120000;

// OpenRouter model IDs — :free suffix routes to the free tier.
// These are the actual DeepSeek models running on OpenRouter's infrastructure.
const MODEL_MAPPING = {
  'deepseek-v3':  'deepseek/deepseek-chat-v3-0324:free', // DeepSeek V3 — best for roleplay
  'deepseek-r1':  'deepseek/deepseek-r1:free',           // DeepSeek R1 — reasoning model
};

// Per-model token defaults.
const MODEL_MAX_TOKENS = {
  'deepseek/deepseek-chat-v3-0324:free': 800,
  'deepseek/deepseek-r1:free':           800,
};

// R1 outputs inline <think> blocks — strip them before delivery.
const NATIVE_THINKERS = new Set([
  'deepseek/deepseek-r1:free',
]);

// Fallback: if R1 fails, try V3.
const FALLBACK_MODEL = {
  'deepseek/deepseek-r1:free': 'deepseek/deepseek-chat-v3-0324:free',
};

// OpenRouter accepts the full standard parameter set.
const FORWARDED_PARAMS = [
  'temperature', 'top_p', 'top_k', 'min_p',
  'max_tokens', 'stop',
  'frequency_penalty', 'presence_penalty',
  'seed', 'n',
];

// 429 is now retried — OpenRouter's 429 is a brief per-minute limit,
// not an account-level daily cap like NIM's was.
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

async function callNIM(nimBody, isStream, id) {
  return axios.post(
    `${API_BASE}/chat/completions`,
    nimBody,
    {
      headers: {
        'Authorization':  `Bearer ${API_KEY}`,
        'Content-Type':   'application/json',
        'Accept':         isStream ? 'text/event-stream' : 'application/json',
        'HTTP-Referer':   'https://openrouter.ai',
        'X-Title':        'JanitorAI Proxy',
      },
      responseType: isStream ? 'stream' : 'json',
      timeout:      REQUEST_TIMEOUT_MS,
    }
  );
}

// ---------------------------------------------------------------------------
// Main proxy endpoint
// ---------------------------------------------------------------------------

app.post('/v1/chat/completions', async (req, res) => {
  const id = reqId();

  // --- API key guard --------------------------------------------------------
  if (!API_KEY) {
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

  // --- Build request body ---------------------------------------------------
  const nimBody = { model: nimModel, messages, stream: isStream };

  for (const param of FORWARDED_PARAMS) {
    if (req.body[param] !== undefined) nimBody[param] = req.body[param];
  }

  if (nimBody.max_tokens  === undefined) {
    nimBody.max_tokens = MODEL_MAX_TOKENS[nimModel] ?? 4096;
  }
  if (nimBody.temperature === undefined) nimBody.temperature = 0.6;

  // --- Call API with retry --------------------------------------------------
  let response;
  let lastErr;
  let activeModel = nimModel;

  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    nimBody.model = activeModel;

    try {
      await waitForRateLimit(id);
      response = await callNIM(nimBody, isStream, id);
      lastErr  = null;
      break;
    } catch (err) {
      lastErr = err;
      const status    = err.response?.status;
      const isTimeout = err.code === 'ECONNABORTED';

      if (isTimeout && FALLBACK_MODEL[activeModel]) {
        const fallback = FALLBACK_MODEL[activeModel];
        console.warn(`[${now()}] [${id}] ${activeModel} timed out — falling back to ${fallback}`);
        activeModel = fallback;
        nimBody.max_tokens = MODEL_MAX_TOKENS[fallback] ?? 800;
        continue;
      }

      if (isTimeout) break;

      if (attempt < MAX_RETRIES && RETRYABLE_STATUSES.has(status)) {
        const wait = RETRY_DELAY_MS * (attempt + 1);
        console.warn(`[${now()}] [${id}] OpenRouter returned ${status}, retrying in ${wait}ms (attempt ${attempt + 1}/${MAX_RETRIES})`);
        await sleep(wait);
        continue;
      }

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
    detail = `Request timed out after ${REQUEST_TIMEOUT_MS / 1000}s with no response from NIM`;
  } else if (err.code === 'ECONNRESET') {
    detail = 'NIM closed the connection unexpectedly (likely server-side congestion)';
  } else if (err.response?.data) {
    const raw = err.response.data;
    if (typeof raw === 'string') {
      detail = raw;
    } else if (Buffer.isBuffer(raw)) {
      detail = raw.toString('utf8');
    } else if (raw && typeof raw.read === 'function') {
      // Readable stream — happens when responseType is 'stream' and NIM returns a 4xx/5xx.
      // The error body is already buffered internally; read it out synchronously.
      try {
        const chunks = [];
        let chunk;
        while (null !== (chunk = raw.read())) {
          chunks.push(typeof chunk === 'string' ? Buffer.from(chunk) : chunk);
        }
        detail = chunks.length > 0
          ? Buffer.concat(chunks).toString('utf8')
          : `[Empty stream body — HTTP ${status}]`;
      } catch (_) {
        detail = `[Could not read stream error body — HTTP ${status}]`;
      }
    } else if (typeof raw === 'object') {
      // Plain object — safe to stringify with circular reference guard
      try {
        const seen = new WeakSet();
        detail = JSON.stringify(raw, (key, value) => {
          if (typeof value === 'object' && value !== null) {
            if (seen.has(value)) return '[Circular]';
            seen.add(value);
          }
          return value;
        });
      } catch (_) {
        detail = `[Unserializable error object: ${raw?.constructor?.name || 'unknown'}]`;
      }
    }
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
