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

// OpenAI alias -> NVIDIA NIM model ID.
// All entries confirmed present in the hosted LLM API docs as of April 2026.
const MODEL_MAPPING = {
  // Best for roleplay and creative writing
  'gpt-4':           'qwen/qwen3-235b-a22b',
  'claude-3-opus':   'qwen/qwen3-235b-a22b',

  // Strong general-purpose models
  'gpt-3.5-turbo':   'deepseek-ai/deepseek-v3.1-terminus',
  'claude-3-sonnet': 'deepseek-ai/deepseek-v3.1-terminus',

  // R1 distilled reasoning models (think blocks stripped before delivery)
  'gpt-4-turbo':     'deepseek-ai/deepseek-r1-distill-qwen-14b',
  'claude-3-haiku':  'deepseek-ai/deepseek-r1-distill-llama-8b',
  'gemini-pro':      'deepseek-ai/deepseek-r1-distill-qwen-7b',

  // Fast, lightweight
  'gpt-4o':          'deepseek-ai/deepseek-v3.1',
  'gpt-4o-mini':     'meta/llama-3.3-70b-instruct',
};

// Models that natively emit <think>...</think> blocks.
// These blocks are stripped from the response before it reaches the client.
const NATIVE_THINKERS = new Set([
  'deepseek-ai/deepseek-r1-distill-qwen-7b',
  'deepseek-ai/deepseek-r1-distill-qwen-14b',
  'deepseek-ai/deepseek-r1-distill-qwen-32b',
  'deepseek-ai/deepseek-r1-distill-llama-8b',
]);

// Generation parameters that are safe to forward to any NIM model.
const FORWARDED_PARAMS = [
  'temperature', 'top_p', 'max_tokens', 'stop',
  'frequency_penalty', 'presence_penalty', 'seed', 'n',
];

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

function now() {
  return new Date().toISOString();
}

function reqId() {
  return crypto.randomBytes(4).toString('hex');
}

// Truncate a string for log output.
function preview(str, limit = 300) {
  if (typeof str !== 'string') return String(str);
  return str.length > limit ? str.slice(0, limit) + '...' : str;
}

// Extract a plain-text preview from a message content value, which may be
// a string or an array of content-part objects (as JanitorAI sometimes sends).
function contentPreview(content) {
  if (typeof content === 'string') return preview(content);
  if (Array.isArray(content)) {
    const text = content
      .filter(p => p.type === 'text')
      .map(p => p.text || '')
      .join(' ');
    return preview(text || '[non-text content]');
  }
  return '[non-text content]';
}

// Strip <think>...</think> blocks from a completed (non-streaming) string.
// Handles multiple think blocks and unclosed tags at the end.
function stripThinkingFull(text) {
  // Remove complete blocks
  text = text.replace(/<think>[\s\S]*?<\/think>/gi, '');
  // Remove any unclosed block that reaches end of string
  text = text.replace(/<think>[\s\S]*/gi, '');
  return text.trimStart();
}

// ---------------------------------------------------------------------------
// Streaming think-stripper — state machine
//
// Processes one raw content string at a time and returns the portion that
// should be forwarded to the client. Maintains state across chunks via the
// returned state object, which the caller must pass back on the next call.
//
// State fields:
//   phase       : 'pass' | 'think'  — whether we are inside a think block
//   pending     : string            — buffered bytes that might be part of
//                                     an opening or closing tag boundary
// ---------------------------------------------------------------------------

function makeThinkState() {
  return { phase: 'pass', pending: '' };
}

// Longest prefix of `needle` that is a suffix of `haystack`.
// Used to detect tag boundaries split across chunks.
function trailingOverlap(haystack, needle) {
  let best = '';
  for (let len = 1; len <= Math.min(haystack.length, needle.length); len++) {
    const suffix = haystack.slice(haystack.length - len);
    const prefix = needle.slice(0, len);
    if (suffix === prefix) best = suffix;
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
        // No complete opening tag. Check whether the tail could be the
        // beginning of one so we do not emit it prematurely.
        const overlap = trailingOverlap(input, startTag);
        if (overlap.length > 0) {
          output       += input.slice(0, input.length - overlap.length);
          state.pending = overlap;
        } else {
          output += input;
        }
        input = '';
      } else {
        // Emit everything before the tag, then enter think phase.
        output += input.slice(0, idx);
        input   = input.slice(idx + startTag.length);
        state.phase = 'think';
      }

    } else {
      // phase === 'think': discard until closing tag
      const endTag = '</think>';
      const idx = input.toLowerCase().indexOf(endTag);

      if (idx === -1) {
        // No closing tag yet; check for partial match at the tail.
        const overlap = trailingOverlap(input, endTag);
        state.pending = overlap.length > 0 ? overlap : '';
        input = '';
      } else {
        // Exit think phase; resume after the tag, trimming leading whitespace.
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
  });
});

app.get('/v1/models', (req, res) => {
  const data = Object.keys(MODEL_MAPPING).map(id => ({
    id,
    object:     'model',
    created:    1700000000,
    owned_by:   'nvidia-nim-proxy',
  }));
  res.json({ object: 'list', data });
});

// ---------------------------------------------------------------------------
// Main proxy endpoint
// ---------------------------------------------------------------------------

app.post('/v1/chat/completions', async (req, res) => {
  const id = reqId();

  // --- API key check -------------------------------------------------------
  if (!NIM_API_KEY) {
    console.error(`[${now()}] [${id}] FATAL: NIM_API_KEY is not set`);
    return res.status(500).json({
      error: { message: 'NIM_API_KEY environment variable is not set', type: 'server_error', code: 500 },
    });
  }

  // --- Basic request validation --------------------------------------------
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

  // Validate each message has at minimum a role field
  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];
    if (!msg || typeof msg.role !== 'string') {
      return res.status(400).json({
        error: { message: `messages[${i}] is missing a "role" field`, type: 'invalid_request_error', code: 400 },
      });
    }
  }

  // --- Model resolution ----------------------------------------------------
  const nimModel       = MODEL_MAPPING[model] || model;
  const isNativeThinker = NATIVE_THINKERS.has(nimModel);
  const isStream        = !!stream;

  // --- Log the request -----------------------------------------------------
  console.log(`\n${'─'.repeat(60)}`);
  console.log(`[${now()}] [${id}] REQUEST`);
  console.log(`  Model    : ${model} -> ${nimModel}`);
  console.log(`  Stream   : ${isStream} | Native thinker: ${isNativeThinker}`);
  console.log(`  Messages : ${messages.length}`);
  messages.forEach((msg, i) => {
    console.log(`  [${i}] ${msg.role.toUpperCase()}: ${contentPreview(msg.content)}`);
  });

  // --- Build NIM request body ----------------------------------------------
  const nimBody = { model: nimModel, messages, stream: isStream };

  // Forward whitelisted generation parameters if the caller supplied them.
  for (const param of FORWARDED_PARAMS) {
    if (req.body[param] !== undefined) {
      nimBody[param] = req.body[param];
    }
  }

  // Apply a sensible default for max_tokens if not provided.
  if (nimBody.max_tokens === undefined) nimBody.max_tokens = 4096;

  // Apply a sensible default temperature for reasoning-friendly output.
  if (nimBody.temperature === undefined) nimBody.temperature = 0.6;

  // --- Call NIM ------------------------------------------------------------
  let response;
  try {
    response = await axios.post(
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
  } catch (err) {
    return handleAxiosError(err, id, res);
  }

  // =========================================================================
  // Streaming path
  // =========================================================================
  if (isStream) {
    res.setHeader('Content-Type',  'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection',    'keep-alive');
    res.setHeader('X-Accel-Buffering', 'no');

    console.log(`[${now()}] [${id}] RESPONSE streaming...`);

    let lineBuffer     = '';
    let logAccumulator = '';
    const thinkState   = makeThinkState();

    response.data.on('data', (chunk) => {
      lineBuffer += chunk.toString();
      const lines = lineBuffer.split('\n');
      lineBuffer  = lines.pop() ?? '';

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;

        // --- DONE sentinel -------------------------------------------------
        if (trimmed === 'data: [DONE]') {
          res.write('data: [DONE]\n\n');
          continue;
        }

        // --- Non-data lines (comments, etc.) --------------------------------
        if (!trimmed.startsWith('data: ')) {
          res.write(line + '\n');
          continue;
        }

        // --- Parse SSE data line -------------------------------------------
        let parsed;
        try {
          parsed = JSON.parse(trimmed.slice(6));
        } catch (_) {
          // Malformed JSON — forward as-is and continue
          res.write(line + '\n');
          continue;
        }

        const delta = parsed.choices?.[0]?.delta;
        if (!delta) {
          res.write(`data: ${JSON.stringify(parsed)}\n\n`);
          continue;
        }

        // Always remove reasoning_content — clients do not expect it
        delete delta.reasoning_content;

        let content = typeof delta.content === 'string' ? delta.content : '';

        // Strip think blocks from native thinkers
        if (isNativeThinker) {
          content = processThinkChunk(content, thinkState);
        }

        // Only emit the chunk if there is visible content to send, or if
        // this is a non-thinker (where empty role/finish_reason chunks
        // must still be forwarded so the client can track turn boundaries).
        if (content || !isNativeThinker) {
          delta.content = content;
          logAccumulator += content;
          res.write(`data: ${JSON.stringify(parsed)}\n\n`);
        }
      }
    });

    response.data.on('end', () => {
      // Flush any pending buffer content (e.g. a trailing partial tag that
      // never resolved — emit it as-is so the response is not silently cut)
      if (isNativeThinker && thinkState.pending && thinkState.phase === 'pass') {
        const flush = thinkState.pending;
        thinkState.pending = '';
        if (flush) {
          const flushChunk = {
            id: `chatcmpl-${Date.now()}`,
            object: 'chat.completion.chunk',
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
      // Send a structured error event so the client knows something went wrong
      if (!res.writableEnded) {
        const errEvent = {
          error: { message: `Stream error: ${err.message}`, type: 'stream_error' },
        };
        res.write(`data: ${JSON.stringify(errEvent)}\n\n`);
        res.write('data: [DONE]\n\n');
        res.end();
      }
    });

    return; // response handled asynchronously
  }

  // =========================================================================
  // Non-streaming path
  // =========================================================================
  const choices = (response.data.choices || []).map(choice => {
    const msg = choice.message ?? {};
    let content = msg.content ?? '';

    // Strip think blocks from native thinkers
    if (isNativeThinker) {
      content = stripThinkingFull(content);
    }

    // Some models return reasoning in a separate field — always discard it
    delete msg.reasoning_content;

    return {
      index:         choice.index ?? 0,
      message:       { role: msg.role ?? 'assistant', content },
      finish_reason: choice.finish_reason ?? null,
    };
  });

  const usage = response.data.usage ?? {
    prompt_tokens: 0, completion_tokens: 0, total_tokens: 0,
  };

  console.log(`[${now()}] [${id}] RESPONSE (non-stream)`);
  choices.forEach((c, i) => {
    console.log(`  [${i}] ASSISTANT: ${preview(c.message.content)}`);
  });
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
// Error handling helpers
// ---------------------------------------------------------------------------

function handleAxiosError(err, id, res) {
  const status = err.response?.status || 500;
  let detail   = err.message;

  if (err.response?.data) {
    const raw = err.response.data;
    if (typeof raw === 'string') {
      detail = raw;
    } else if (Buffer.isBuffer(raw)) {
      detail = raw.toString('utf8');
    } else if (typeof raw === 'object') {
      detail = JSON.stringify(raw);
    }
  }

  console.error(`[${now()}] [${id}] PROXY ERROR [${status}]: ${preview(detail, 500)}`);

  if (!res.headersSent) {
    res.status(status).json({
      error: {
        message: detail || 'An error occurred communicating with the NIM API',
        type:    status >= 500 ? 'server_error' : 'invalid_request_error',
        code:    status,
      },
    });
  }
}

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------

app.listen(PORT, () => {
  console.log('');
  console.log(`OpenAI to NVIDIA NIM Proxy`);
  console.log(`  Port    : ${PORT}`);
  console.log(`  Health  : http://localhost:${PORT}/health`);
  console.log(`  API key : ${NIM_API_KEY ? 'SET' : 'NOT SET -- set NIM_API_KEY environment variable'}`);
  console.log(`  Models  : ${Object.keys(MODEL_MAPPING).length} mapped`);
  console.log('');
});
