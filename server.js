// server.js
// OpenAI-compatible proxy for NVIDIA NIM
// Designed for JanitorAI / OpenAI-compatible clients.
//
// IMPORTANT:
// - This proxy imposes NO max_tokens limit.
// - This proxy imposes NO min_tokens limit.
// - This proxy imposes NO default temperature.
// - NVIDIA/NIM receives whatever parameters the client actually supplies.
// - NVIDIA request timeout is DISABLED so long generations are not killed by Render.
//
// -----------------------------------------------------------------------------
// Dependencies
// -----------------------------------------------------------------------------
//
// npm install express cors axios
//
// -----------------------------------------------------------------------------

'use strict';

const express = require('express');
const cors = require('cors');
const axios = require('axios');
const crypto = require('crypto');

const app = express();

const PORT = process.env.PORT || 3000;

app.use(cors());

app.use(
  express.json({
    limit: '10mb',
  })
);

// -----------------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------------

const NIM_API_BASE =
  process.env.NIM_API_BASE ||
  'https://integrate.api.nvidia.com/v1';

const NIM_API_KEY = process.env.NIM_API_KEY;

// Retry only transient server-side failures.
// Do NOT retry a request after it has started streaming.
const MAX_RETRIES = 2;
const RETRY_DELAY_MS = 1000;

// IMPORTANT:
//
// 0 means NO Axios timeout.
//
// Previously this was:
//     120000
//
// That caused:
//     ECONNABORTED
//     NVIDIA request timed out after 120s
//
// even though NVIDIA had already generated a response.
//
// With 0, the connection is allowed to remain open until NVIDIA
// finishes or the connection itself fails.
const REQUEST_TIMEOUT_MS = 0;

// -----------------------------------------------------------------------------
// Model aliases
// -----------------------------------------------------------------------------
//
// Janitor can use either these short aliases or the actual NVIDIA model ID.
//
// If a model is not in this mapping, its name is passed directly to NVIDIA.
//

const MODEL_MAPPING = {
  'kimi-k2':
    'moonshotai/kimi-k2-instruct',

  'mistral-large':
    'mistralai/mistral-large-3-675b-instruct-2512',

  'llama-405b':
    'meta/llama-3.1-405b-instruct',

  'nemotron-49b':
    'nvidia/llama-3.3-nemotron-super-49b-v1',

  'seed-36b':
    'bytedance/seed-oss-36b-instruct',

  'magistral':
    'mistralai/magistral-small-2506',

  'qwen-coder':
    'qwen/qwen3-coder-480b-a35b-instruct',

  'glm-5':
    'z-ai/glm-5.2',

  // DeepSeek V4.1 Flash
  'deepseek-v4.1-flash':
    'deepseek-ai/deepseek-v4.1-flash',

  // Also support this spelling if Janitor sends it as an alias.
  'deepseek-v4_1-flash':
    'deepseek-ai/deepseek-v4.1-flash',
};

// -----------------------------------------------------------------------------
// Fallback models
// -----------------------------------------------------------------------------
//
// Intentionally empty.
//
// A timeout should NOT cause us to silently switch models, because:
//
// 1. We have disabled the proxy timeout.
// 2. The user selected a specific model.
// 3. Falling back changes model behavior unexpectedly.
// 4. A fallback can also change token behavior.
//
// If NVIDIA itself returns an error, that error is passed back to Janitor.
//

const FALLBACK_MODEL = {};

// -----------------------------------------------------------------------------
// Models whose reasoning fields should be removed
// -----------------------------------------------------------------------------

const REQUIRES_THINKING_PARAM = new Set();

const NATIVE_THINKERS = new Set();

// -----------------------------------------------------------------------------
// Parameters forwarded to NVIDIA
// -----------------------------------------------------------------------------
//
// IMPORTANT:
// There are NO proxy-generated token values here.
//
// If Janitor sends max_tokens, we forward it.
// If Janitor does not send max_tokens, we do not add one.
//
// Same applies to temperature and top_p.
//
// We intentionally do not send:
//
// - min_tokens
// - min_p
// - top_k
// - n
//
// unless explicitly added later after verifying that the target NIM
// endpoint accepts them.
//
// -----------------------------------------------------------------------------

const FORWARDED_PARAMS = [
  'temperature',
  'top_p',
  'max_tokens',
  'stop',
  'frequency_penalty',
  'presence_penalty',
  'seed',

  // OpenAI-compatible fields that some clients may send.
  'response_format',
  'tools',
  'tool_choice',
  'parallel_tool_calls',
  'user',
];

// -----------------------------------------------------------------------------
// Retryable HTTP statuses
// -----------------------------------------------------------------------------

const RETRYABLE_STATUSES = new Set([
  502,
  503,
  504,
]);

// -----------------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------------

function now() {
  return new Date().toISOString();
}

function reqId() {
  return crypto.randomBytes(4).toString('hex');
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function preview(value, limit = 500) {
  if (value === undefined || value === null) {
    return '';
  }

  let str;

  if (typeof value === 'string') {
    str = value;
  } else {
    try {
      str = JSON.stringify(value);
    } catch (_) {
      str = String(value);
    }
  }

  return str.length > limit
    ? str.slice(0, limit) + '...'
    : str;
}

// -----------------------------------------------------------------------------
// Message content preview
// -----------------------------------------------------------------------------

function contentPreview(content) {
  if (typeof content === 'string') {
    return preview(content);
  }

  if (Array.isArray(content)) {
    const text = content
      .filter((part) => part && part.type === 'text')
      .map((part) => part.text || '')
      .join(' ');

    return preview(text || '[non-text content]');
  }

  if (content === null || content === undefined) {
    return '[empty]';
  }

  return '[non-text content]';
}

// -----------------------------------------------------------------------------
// Remove <think>...</think> from complete responses
// -----------------------------------------------------------------------------

function stripThinkingFull(text) {
  if (typeof text !== 'string') {
    return text;
  }

  // Remove complete blocks.
  text = text.replace(
    /<think>[\s\S]*?<\/think>/gi,
    ''
  );

  // Remove an unclosed thinking block at the end.
  text = text.replace(
    /<think>[\s\S]*/gi,
    ''
  );

  return text.trimStart();
}

// -----------------------------------------------------------------------------
// Streaming <think> stripper
// -----------------------------------------------------------------------------
//
// Streaming responses can split:
//
//     <think>
//
// across two separate network chunks.
//
// This state machine prevents partial <think> tags from leaking to Janitor.
//
// -----------------------------------------------------------------------------

function makeThinkState() {
  return {
    phase: 'pass',
    pending: '',
  };
}

function trailingOverlap(haystack, needle) {
  let best = '';

  const maxLength = Math.min(
    haystack.length,
    needle.length
  );

  for (let len = 1; len <= maxLength; len++) {
    if (
      haystack.slice(haystack.length - len) ===
      needle.slice(0, len)
    ) {
      best = needle.slice(0, len);
    }
  }

  return best;
}

function processThinkChunk(raw, state) {
  if (!raw) {
    return '';
  }

  let input = state.pending + raw;

  state.pending = '';

  let output = '';

  while (input.length > 0) {
    if (state.phase === 'pass') {
      const startTag = '<think>';

      const idx = input
        .toLowerCase()
        .indexOf(startTag);

      if (idx === -1) {
        const overlap = trailingOverlap(
          input,
          startTag
        );

        output += input.slice(
          0,
          input.length - overlap.length
        );

        state.pending = overlap;

        input = '';
      } else {
        output += input.slice(0, idx);

        input = input.slice(
          idx + startTag.length
        );

        state.phase = 'think';
      }
    } else {
      const endTag = '</think>';

      const idx = input
        .toLowerCase()
        .indexOf(endTag);

      if (idx === -1) {
        state.pending = trailingOverlap(
          input,
          endTag
        );

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

// -----------------------------------------------------------------------------
// Health endpoint
// -----------------------------------------------------------------------------

app.get('/health', (req, res) => {
  res.json({
    status: 'ok',

    service:
      'OpenAI to NVIDIA NIM Proxy',

    nim_base:
      NIM_API_BASE,

    api_key_set:
      !!NIM_API_KEY,

    models:
      Object.keys(MODEL_MAPPING).length,

    max_retries:
      MAX_RETRIES,

    request_timeout_ms:
      REQUEST_TIMEOUT_MS,

    token_limits:
      'none imposed by proxy',

    temperature_default:
      'none imposed by proxy',
  });
});

// -----------------------------------------------------------------------------
// Model list
// -----------------------------------------------------------------------------

app.get('/v1/models', (req, res) => {
  const mappedModels = Object.keys(
    MODEL_MAPPING
  ).map((id) => ({
    id,

    object: 'model',

    created: 1700000000,

    owned_by:
      'nvidia-nim-proxy',
  }));

  res.json({
    object: 'list',

    data: mappedModels,
  });
});

// -----------------------------------------------------------------------------
// NVIDIA API request helper
// -----------------------------------------------------------------------------

async function callNIM(
  nimBody,
  isStream
) {
  return axios.post(
    `${NIM_API_BASE}/chat/completions`,
    nimBody,
    {
      headers: {
        Authorization:
          `Bearer ${NIM_API_KEY}`,

        'Content-Type':
          'application/json',

        Accept: isStream
          ? 'text/event-stream'
          : 'application/json',

        // Tell proxies/load balancers not to buffer.
        'Cache-Control':
          'no-cache',

        'X-Accel-Buffering':
          'no',
      },

      responseType:
        isStream
          ? 'stream'
          : 'json',

      // IMPORTANT:
      //
      // 0 = no timeout.
      //
      // This allows DeepSeek to take as long as NVIDIA
      // needs to finish the request.
      timeout:
        REQUEST_TIMEOUT_MS,

      // Keep the connection alive.
      maxContentLength:
        Infinity,

      maxBodyLength:
        Infinity,
    }
  );
}

// -----------------------------------------------------------------------------
// Main OpenAI-compatible endpoint
// -----------------------------------------------------------------------------

app.post(
  '/v1/chat/completions',
  async (req, res) => {
    const id = reqId();

    // -------------------------------------------------------------------------
    // API key
    // -------------------------------------------------------------------------

    if (!NIM_API_KEY) {
      console.error(
        `[${now()}] [${id}] FATAL: NIM_API_KEY is not set`
      );

      return res.status(500).json({
        error: {
          message:
            'NIM_API_KEY environment variable is not set',

          type:
            'server_error',

          code:
            500,
        },
      });
    }

    // -------------------------------------------------------------------------
    // Validate request
    // -------------------------------------------------------------------------

    const {
      model,
      messages,
      stream,
    } = req.body;

    if (
      !model ||
      typeof model !== 'string'
    ) {
      return res.status(400).json({
        error: {
          message:
            'Request body must include a "model" string',

          type:
            'invalid_request_error',

          code:
            400,
        },
      });
    }

    if (
      !Array.isArray(messages) ||
      messages.length === 0
    ) {
      return res.status(400).json({
        error: {
          message:
            'Request body must include a non-empty "messages" array',

          type:
            'invalid_request_error',

          code:
            400,
        },
      });
    }

    for (
      let i = 0;
      i < messages.length;
      i++
    ) {
      if (
        !messages[i] ||
        typeof messages[i].role !== 'string'
      ) {
        return res.status(400).json({
          error: {
            message:
              `messages[${i}] is missing a "role" field`,

            type:
              'invalid_request_error',

            code:
              400,
          },
        });
      }
    }

    // -------------------------------------------------------------------------
    // Resolve model
    // -------------------------------------------------------------------------

    const nimModel =
      MODEL_MAPPING[model] ||
      model;

    const isStream =
      !!stream;

    // -------------------------------------------------------------------------
    // Request logging
    // -------------------------------------------------------------------------

    console.log('');
    console.log(
      '─'.repeat(70)
    );

    console.log(
      `[${now()}] [${id}] REQUEST`
    );

    console.log(
      `  Model    : ${model} -> ${nimModel}`
    );

    console.log(
      `  Stream   : ${isStream}`
    );

    console.log(
      `  Messages : ${messages.length}`
    );

    messages.forEach(
      (message, index) => {
        console.log(
          `  [${index}] ${String(message.role).toUpperCase()}: ${contentPreview(message.content)}`
        );
      }
    );

    // -------------------------------------------------------------------------
    // Build NVIDIA request
    // -------------------------------------------------------------------------
    //
    // IMPORTANT:
    //
    // We deliberately start with ONLY:
    //
    //     model
    //     messages
    //     stream
    //
    // Nothing else is invented.
    //
    // In particular:
    //
    //     NO max_tokens default
    //     NO min_tokens
    //     NO temperature default
    //     NO top_p default
    //
    // -------------------------------------------------------------------------

    const nimBody = {
      model:
        nimModel,

      messages:
        messages,

      stream:
        isStream,
    };

    // Forward only parameters actually supplied by Janitor.
    for (
      const param of FORWARDED_PARAMS
    ) {
      if (
        req.body[param] !== undefined
      ) {
        nimBody[param] =
          req.body[param];
      }
    }

    console.log(
      `[${now()}] [${id}] NVIDIA REQUEST`
    );

    console.log(
      `  Payload model : ${nimBody.model}`
    );

    console.log(
      `  Stream        : ${nimBody.stream}`
    );

    console.log(
      `  max_tokens    : ${
        nimBody.max_tokens === undefined
          ? '[not supplied]'
          : nimBody.max_tokens
      }`
    );

    console.log(
      `  temperature   : ${
        nimBody.temperature === undefined
          ? '[not supplied]'
          : nimBody.temperature
      }`
    );

    console.log(
      `  top_p         : ${
        nimBody.top_p === undefined
          ? '[not supplied]'
          : nimBody.top_p
      }`
    );

    // -------------------------------------------------------------------------
    // Call NVIDIA
    // -------------------------------------------------------------------------

    let response = null;
    let lastErr = null;

    for (
      let attempt = 0;
      attempt <= MAX_RETRIES;
      attempt++
    ) {
      try {
        response =
          await callNIM(
            nimBody,
            isStream
          );

        lastErr = null;

        break;
      } catch (err) {
        lastErr = err;

        const status =
          err.response?.status;

        const isTimeout =
          err.code === 'ECONNABORTED' ||
          err.code === 'ETIMEDOUT';

        // ---------------------------------------------------------------------
        // Timeout
        // ---------------------------------------------------------------------
        //
        // Normally impossible because REQUEST_TIMEOUT_MS = 0.
        //
        // If an upstream/network timeout occurs anyway, DO NOT silently
        // switch models.
        // ---------------------------------------------------------------------

        if (isTimeout) {
          console.error(
            `[${now()}] [${id}] NVIDIA TIMEOUT`
          );

          console.error(
            `  Model : ${nimBody.model}`
          );

          console.error(
            `  Code  : ${err.code}`
          );

          console.error(
            `  Detail: ${err.message}`
          );

          break;
        }

        // ---------------------------------------------------------------------
        // Retry transient NVIDIA server errors
        // ---------------------------------------------------------------------

        if (
          attempt < MAX_RETRIES &&
          RETRYABLE_STATUSES.has(status)
        ) {
          const wait =
            RETRY_DELAY_MS *
            (attempt + 1);

          console.warn(
            `[${now()}] [${id}] NVIDIA returned HTTP ${status}; retrying in ${wait}ms (${attempt + 1}/${MAX_RETRIES})`
          );

          await sleep(wait);

          continue;
        }

        break;
      }
    }

    // -------------------------------------------------------------------------
    // Request failed before a response was established
    // -------------------------------------------------------------------------

    if (lastErr) {
      return handleAxiosError(
        lastErr,
        id,
        res
      );
    }

    if (!response) {
      return res.status(502).json({
        error: {
          message:
            'NVIDIA returned no response',

          type:
            'server_error',

          code:
            502,
        },
      });
    }

    // =========================================================================
    // STREAMING RESPONSE
    // =========================================================================

    if (isStream) {
      return handleStreamingResponse(
        response,
        res,
        id,
        model
      );
    }

    // =========================================================================
    // NON-STREAMING RESPONSE
    // =========================================================================

    return handleNonStreamingResponse(
      response,
      res,
      id,
      model
    );
  }
);

// -----------------------------------------------------------------------------
// Streaming response handler
// -----------------------------------------------------------------------------

function handleStreamingResponse(
  response,
  res,
  id,
  requestedModel
) {
  // ---------------------------------------------------------------------------
  // Headers
  // ---------------------------------------------------------------------------

  res.status(200);

  res.setHeader(
    'Content-Type',
    'text/event-stream; charset=utf-8'
  );

  res.setHeader(
    'Cache-Control',
    'no-cache, no-transform'
  );

  res.setHeader(
    'Connection',
    'keep-alive'
  );

  res.setHeader(
    'X-Accel-Buffering',
    'no'
  );

  // Helpful for some reverse proxies.
  res.setHeader(
    'Transfer-Encoding',
    'chunked'
  );

  // Send headers immediately.
  if (
    typeof res.flushHeaders === 'function'
  ) {
    res.flushHeaders();
  }

  console.log(
    `[${now()}] [${id}] NVIDIA RESPONSE: streaming`
  );

  // ---------------------------------------------------------------------------
  // State
  // ---------------------------------------------------------------------------

  let lineBuffer = '';

  let logAccumulator = '';

  let streamFinished = false;

  let clientDisconnected = false;

  const thinkState =
    makeThinkState();

  // ---------------------------------------------------------------------------
  // Client disconnect handling
  // ---------------------------------------------------------------------------
  //
  // If Janitor closes its connection, stop consuming NVIDIA's response.
  //
  // This prevents unnecessary generation after the client has gone away.
  //
  // IMPORTANT:
  // Do not treat normal response completion as a client disconnect.
  // ---------------------------------------------------------------------------

  const onClientClose = () => {
    if (streamFinished) {
      return;
    }

    clientDisconnected = true;

    console.warn(
      `[${now()}] [${id}] CLIENT CONNECTION CLOSED`
    );

    if (
      response.data &&
      typeof response.data.destroy === 'function'
    ) {
      response.data.destroy();
    }
  };

  reqSafeOnClose(res, onClientClose);

  // ---------------------------------------------------------------------------
  // Process a single SSE line
  // ---------------------------------------------------------------------------

  function processLine(line) {
    const trimmed =
      line.trim();

    if (!trimmed) {
      return;
    }

    // -------------------------------------------------------------------------
    // NVIDIA/OpenAI stream termination
    // -------------------------------------------------------------------------

    if (
      trimmed === 'data: [DONE]'
    ) {
      if (!streamFinished) {
        res.write(
          'data: [DONE]\n\n'
        );
      }

      return;
    }

    // -------------------------------------------------------------------------
    // Non-data SSE line
    // -------------------------------------------------------------------------

    if (
      !trimmed.startsWith('data:')
    ) {
      // Forward unknown SSE fields rather than silently deleting them.
      if (!clientDisconnected) {
        res.write(
          line + '\n'
        );
      }

      return;
    }

    // -------------------------------------------------------------------------
    // Extract JSON
    // -------------------------------------------------------------------------

    const jsonText =
      trimmed.slice(5).trim();

    if (!jsonText) {
      return;
    }

    let parsed;

    try {
      parsed =
        JSON.parse(jsonText);
    } catch (err) {
      // If NVIDIA ever gives us malformed/non-JSON data,
      // preserve it rather than crashing the stream.
      console.warn(
        `[${now()}] [${id}] Could not parse SSE JSON: ${preview(jsonText, 200)}`
      );

      if (!clientDisconnected) {
        res.write(
          `data: ${jsonText}\n\n`
        );
      }

      return;
    }

    // -------------------------------------------------------------------------
    // Error payload from NVIDIA
    // -------------------------------------------------------------------------

    if (parsed.error) {
      console.error(
        `[${now()}] [${id}] NVIDIA STREAM ERROR: ${preview(parsed.error)}`
      );

      if (!clientDisconnected) {
        res.write(
          `data: ${JSON.stringify(parsed)}\n\n`
        );
      }

      return;
    }

    // -------------------------------------------------------------------------
    // Locate delta
    // -------------------------------------------------------------------------

    const choice =
      parsed.choices?.[0];

    const delta =
      choice?.delta;

    if (!delta) {
      // Some valid OpenAI-compatible chunks may not contain delta.
      // Forward them unchanged.
      if (!clientDisconnected) {
        res.write(
          `data: ${JSON.stringify(parsed)}\n\n`
        );
      }

      return;
    }

    // -------------------------------------------------------------------------
    // Remove reasoning_content
    // -------------------------------------------------------------------------

    delete delta.reasoning_content;

    // -------------------------------------------------------------------------
    // Process text content
    // -------------------------------------------------------------------------

    if (
      typeof delta.content === 'string'
    ) {
      const originalContent =
        delta.content;

      const filteredContent =
        processThinkChunk(
          originalContent,
          thinkState
        );

      delta.content =
        filteredContent;

      if (filteredContent) {
        logAccumulator +=
          filteredContent;
      }
    }

    // -------------------------------------------------------------------------
    // Forward EVERYTHING
    // -------------------------------------------------------------------------
    //
    // This is an important difference from the old handler.
    //
    // Previously, chunks could be discarded when:
    //
    //     delta.content === ''
    //
    // That can accidentally discard:
    //
    // - finish_reason
    // - tool calls
    // - function calls
    // - role information
    // - other OpenAI-compatible delta fields
    //
    // We now forward the chunk after cleaning reasoning content.
    // -------------------------------------------------------------------------

    if (!clientDisconnected) {
      res.write(
        `data: ${JSON.stringify(parsed)}\n\n`
      );
    }

    // -------------------------------------------------------------------------
    // Finish reason
    // -------------------------------------------------------------------------

    if (
      choice?.finish_reason
    ) {
      console.log(
        `[${now()}] [${id}] STREAM FINISH: ${choice.finish_reason}`
      );
    }
  }

  // ---------------------------------------------------------------------------
  // NVIDIA data events
  // ---------------------------------------------------------------------------

  response.data.on(
    'data',
    (chunk) => {
      if (
        clientDisconnected ||
        streamFinished
      ) {
        return;
      }

      lineBuffer +=
        chunk.toString('utf8');

      const lines =
        lineBuffer.split('\n');

      lineBuffer =
        lines.pop() || '';

      for (
        const line of lines
      ) {
        if (
          clientDisconnected ||
          streamFinished
        ) {
          break;
        }

        processLine(line);
      }
    }
  );

  // ---------------------------------------------------------------------------
  // NVIDIA stream complete
  // ---------------------------------------------------------------------------

  response.data.on(
    'end',
    () => {
      if (streamFinished) {
        return;
      }

      streamFinished = true;

      // -----------------------------------------------------------------------
      // Process any final line that didn't end with \n.
      // -----------------------------------------------------------------------

      if (
        lineBuffer.trim()
      ) {
        processLine(
          lineBuffer
        );
      }

      // -----------------------------------------------------------------------
      // Flush a pending partial <think> tag.
      //
      // Example:
      //
      // NVIDIA ended with:
      //
      //     "Hello<thi"
      //
      // We held "<thi" because it could become "<think>".
      //
      // If the stream is now definitely finished, it is safe to forward
      // that pending text.
      // -----------------------------------------------------------------------

      if (
        thinkState.pending &&
        thinkState.phase === 'pass' &&
        !clientDisconnected
      ) {
        const flush =
          thinkState.pending;

        if (flush) {
          const flushChunk = {
            id:
              `chatcmpl-${Date.now()}`,

            object:
              'chat.completion.chunk',

            created:
              Math.floor(
                Date.now() / 1000
              ),

            model:
              requestedModel,

            choices: [
              {
                index: 0,

                delta: {
                  content:
                    flush,
                },

                finish_reason:
                  null,
              },
            ],
          };

          res.write(
            `data: ${JSON.stringify(flushChunk)}\n\n`
          );

          logAccumulator +=
            flush;
        }

        thinkState.pending = '';
      }

      // -----------------------------------------------------------------------
      // Ensure Janitor gets [DONE].
      //
      // If NVIDIA already sent it, sending another one is unnecessary.
      // We track completion here and send one final marker because it is safer
      // for clients that depend on it.
      // -----------------------------------------------------------------------

      if (!clientDisconnected) {
        res.write(
          'data: [DONE]\n\n'
        );
      }

      console.log(
        `[${now()}] [${id}] STREAM END`
      );

      console.log(
        `  ASSISTANT: ${preview(logAccumulator, 1000)}`
      );

      if (!res.writableEnded) {
        res.end();
      }
    }
  );

  // ---------------------------------------------------------------------------
  // NVIDIA stream error
  // ---------------------------------------------------------------------------

  response.data.on(
    'error',
    (err) => {
      if (streamFinished) {
        return;
      }

      streamFinished = true;

      console.error(
        `[${now()}] [${id}] NVIDIA STREAM ERROR`
      );

      console.error(
        `  Code   : ${err.code || 'N/A'}`
      );

      console.error(
        `  Detail : ${err.message}`
      );

      // If Janitor is still connected, send a valid SSE error.
      if (
        !clientDisconnected &&
        !res.writableEnded
      ) {
        const errorPayload = {
          error: {
            message:
              `NVIDIA stream error: ${err.message}`,

            type:
              'stream_error',

            code:
              err.code || 500,
          },
        };

        try {
          res.write(
            `data: ${JSON.stringify(errorPayload)}\n\n`
          );

          res.write(
            'data: [DONE]\n\n'
          );
        } catch (_) {
          // Ignore secondary socket errors.
        }

        res.end();
      }
    }
  );

  // ---------------------------------------------------------------------------
  // Response close
  // ---------------------------------------------------------------------------

  res.on(
    'close',
    () => {
      if (
        !streamFinished &&
        !clientDisconnected
      ) {
        clientDisconnected = true;

        console.warn(
          `[${now()}] [${id}] JANITOR CLOSED STREAM`
        );

        if (
          response.data &&
          typeof response.data.destroy === 'function'
        ) {
          response.data.destroy();
        }
      }
    }
  );
}

// -----------------------------------------------------------------------------
// Helper for detecting client disconnect
// -----------------------------------------------------------------------------

function reqSafeOnClose(
  res,
  callback
) {
  // We intentionally use the response close event instead of
  // relying on req.close, because req.close can behave differently
  // depending on how Express/Node has consumed the request body.

  if (
    res &&
    typeof res.on === 'function'
  ) {
    // Do not attach another callback here.
    // The actual stream cleanup is handled by res.on('close')
    // inside handleStreamingResponse.
    return;
  }
}

// -----------------------------------------------------------------------------
// Non-streaming response
// -----------------------------------------------------------------------------

function handleNonStreamingResponse(
  response,
  res,
  id,
  requestedModel
) {
  const data =
    response.data || {};

  const choices =
    Array.isArray(data.choices)
      ? data.choices
      : [];

  const cleanedChoices =
    choices.map(
      (choice) => {
        const originalMessage =
          choice.message || {};

        const message = {
          ...originalMessage,
        };

        // Remove reasoning content.
        delete message.reasoning_content;

        // Clean normal text.
        if (
          typeof message.content ===
          'string'
        ) {
          message.content =
            stripThinkingFull(
              message.content
            );
        }

        return {
          ...choice,

          index:
            choice.index ?? 0,

          message,

          finish_reason:
            choice.finish_reason ??
            null,
        };
      }
    );

  const usage =
    data.usage || {
      prompt_tokens: 0,

      completion_tokens: 0,

      total_tokens: 0,
    };

  console.log(
    `[${now()}] [${id}] RESPONSE (non-stream)`
  );

  cleanedChoices.forEach(
    (choice, index) => {
      console.log(
        `  [${index}] ASSISTANT: ${preview(choice.message?.content)}`
      );
    }
  );

  console.log(
    `  Usage: prompt=${usage.prompt_tokens ?? 0} completion=${usage.completion_tokens ?? 0} total=${usage.total_tokens ?? 0}`
  );

  // ---------------------------------------------------------------------------
  // Return an OpenAI-compatible response.
  //
  // Preserve additional NVIDIA fields where possible.
  // ---------------------------------------------------------------------------

  return res.json({
    ...data,

    id:
      data.id ||
      `chatcmpl-${Date.now()}`,

    object:
      data.object ||
      'chat.completion',

    created:
      data.created ||
      Math.floor(
        Date.now() / 1000
      ),

    model:
      requestedModel,

    choices:
      cleanedChoices,

    usage,
  });
}

// -----------------------------------------------------------------------------
// Axios error handling
// -----------------------------------------------------------------------------

async function handleAxiosError(
  err,
  id,
  res
) {
  const status =
    err.response?.status;

  const detail =
    await resolveErrorDetail(
      err,
      status
    );

  console.error(
    `[${now()}] [${id}] NVIDIA ERROR`
  );

  console.error(
    `  HTTP   : ${status || 'N/A'}`
  );

  console.error(
    `  Code   : ${err.code || 'N/A'}`
  );

  console.error(
    `  Detail : ${preview(detail, 1000)}`
  );

  if (res.headersSent) {
    // Headers were already sent, so we cannot convert this into
    // a normal JSON HTTP response.
    if (!res.writableEnded) {
      try {
        res.write(
          `data: ${JSON.stringify({
            error: {
              message: detail,
              type: 'stream_error',
              code: status || 500,
            },
          })}\n\n`
        );

        res.write(
          'data: [DONE]\n\n'
        );

        res.end();
      } catch (_) {
        // Ignore socket errors.
      }
    }

    return;
  }

  const httpStatus =
    status || 502;

  return res
    .status(httpStatus)
    .json({
      error: {
        message:
          detail ||
          'An error occurred communicating with the NVIDIA NIM API',

        type:
          httpStatus >= 500
            ? 'server_error'
            : 'invalid_request_error',

        code:
          httpStatus,
      },
    });
}

// -----------------------------------------------------------------------------
// Resolve readable Axios error body
// -----------------------------------------------------------------------------

async function resolveErrorDetail(
  err,
  status
) {
  // ---------------------------------------------------------------------------
  // Timeout
  // ---------------------------------------------------------------------------

  if (
    err.code === 'ECONNABORTED' ||
    err.code === 'ETIMEDOUT'
  ) {
    return (
      'The NVIDIA connection timed out at the network/transport level. ' +
      'The proxy itself has no request timeout.'
    );
  }

  // ---------------------------------------------------------------------------
  // Connection reset
  // ---------------------------------------------------------------------------

  if (
    err.code === 'ECONNRESET'
  ) {
    return (
      'NVIDIA closed the connection unexpectedly ' +
      '(ECONNRESET).'
    );
  }

  // ---------------------------------------------------------------------------
  // No response body
  // ---------------------------------------------------------------------------

  const raw =
    err.response?.data;

  if (!raw) {
    return (
      err.message ||
      `NVIDIA returned HTTP ${status || 500}`
    );
  }

  // ---------------------------------------------------------------------------
  // String
  // ---------------------------------------------------------------------------

  if (
    typeof raw === 'string'
  ) {
    return raw;
  }

  // ---------------------------------------------------------------------------
  // Buffer
  // ---------------------------------------------------------------------------

  if (
    Buffer.isBuffer(raw)
  ) {
    return raw.toString(
      'utf8'
    );
  }

  // ---------------------------------------------------------------------------
  // Stream
  // ---------------------------------------------------------------------------

  if (
    raw &&
    typeof raw.on === 'function'
  ) {
    try {
      const chunks =
        await new Promise(
          (resolve, reject) => {
            const collected = [];

            let settled = false;

            const finish = (
              value
            ) => {
              if (settled) {
                return;
              }

              settled = true;

              resolve(value);
            };

            raw.on(
              'data',
              (chunk) => {
                collected.push(
                  Buffer.isBuffer(chunk)
                    ? chunk
                    : Buffer.from(
                        String(chunk)
                      )
                );
              }
            );

            raw.on(
              'end',
              () => {
                finish(
                  collected
                );
              }
            );

            raw.on(
              'error',
              reject
            );

            // Safety timeout for a broken error stream.
            setTimeout(
              () => {
                finish(
                  collected
                );
              },
              3000
            );
          }
        );

      if (
        chunks.length > 0
      ) {
        return Buffer
          .concat(chunks)
          .toString('utf8');
      }

      return (
        `[Empty stream body — HTTP ${status || 500}]`
      );
    } catch (_) {
      return (
        `[Could not read NVIDIA error body — HTTP ${status || 500}]`
      );
    }
  }

  // ---------------------------------------------------------------------------
  // Object
  // ---------------------------------------------------------------------------

  if (
    typeof raw === 'object'
  ) {
    try {
      const seen =
        new WeakSet();

      return JSON.stringify(
        raw,
        (key, value) => {
          if (
            typeof value === 'object' &&
            value !== null
          ) {
            if (
              seen.has(value)
            ) {
              return '[Circular]';
            }

            seen.add(value);
          }

          return value;
        }
      );
    } catch (_) {
      return (
        `[Unserializable NVIDIA error object: ${
          raw?.constructor?.name ||
          'unknown'
        }]`
      );
    }
  }

  return String(raw);
}

// -----------------------------------------------------------------------------
// 404 catch-all
// -----------------------------------------------------------------------------

app.all(
  '*',
  (req, res) => {
    res.status(404).json({
      error: {
        message:
          `Endpoint ${req.path} is not supported by this proxy`,

        type:
          'not_found',

        code:
          404,
      },
    });
  }
);

// -----------------------------------------------------------------------------
// Express error handler
// -----------------------------------------------------------------------------

app.use(
  (err, req, res, next) => {
    console.error(
      `[${now()}] EXPRESS ERROR:`,
      err
    );

    if (
      res.headersSent
    ) {
      return next(err);
    }

    return res
      .status(500)
      .json({
        error: {
          message:
            err.message ||
            'Internal proxy error',

          type:
            'server_error',

          code:
            500,
        },
      });
  }
);

// -----------------------------------------------------------------------------
// Start server
// -----------------------------------------------------------------------------

app.listen(
  PORT,
  () => {
    console.log('');
    console.log(
      '============================================================'
    );
    console.log(
      ' OpenAI -> NVIDIA NIM Proxy'
    );
    console.log(
      '============================================================'
    );

    console.log(
      ` Port          : ${PORT}`
    );

    console.log(
      ` Health        : /health`
    );

    console.log(
      ` API key       : ${
        NIM_API_KEY
          ? 'SET'
          : 'NOT SET'
      }`
    );

    console.log(
      ` NVIDIA API    : ${NIM_API_BASE}`
    );

    console.log(
      ` Models        : ${Object.keys(MODEL_MAPPING).length} aliases`
    );

    console.log(
      ` Retries       : ${MAX_RETRIES}`
    );

    console.log(
      ` Request timeout: ${
        REQUEST_TIMEOUT_MS === 0
          ? 'NONE'
          : `${REQUEST_TIMEOUT_MS}ms`
      }`
    );

    console.log(
      ' Token limits  : NONE imposed by proxy'
    );

    console.log(
      ' Temperature   : NONE imposed by proxy'
    );

    console.log(
      '============================================================'
    );

    console.log('');
  }
);
