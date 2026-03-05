// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '10mb' }));

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY  = process.env.NIM_API_KEY;

// Model mapping — OpenAI names → NVIDIA NIM model IDs
const MODEL_MAPPING = {
  'gpt-4':           'qwen/qwen3-235b-a22b',
  'gpt-3.5-turbo':   'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'claude-3-opus':   'qwen/qwen3-235b-a22b',
  'claude-3-sonnet': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4-turbo':     'deepseek-ai/deepseek-r1-distill-qwen-14b',
  'claude-3-haiku':  'deepseek-ai/deepseek-r1-distill-llama-8b',
  'gemini-pro':      'deepseek-ai/deepseek-r1-distill-qwen-14b',
  'gpt-4o':          'deepseek-ai/deepseek-v3.2',
  'gpt-4o-mini':     'meta/llama-3.3-70b-instruct',
};

// These models always output <think> blocks natively — strip them silently
const NATIVE_THINKERS = new Set([
  'deepseek-ai/deepseek-r1-distill-qwen-14b',
  'deepseek-ai/deepseek-r1-distill-llama-8b',
  'deepseek-ai/deepseek-r1-distill-qwen-32b',
]);

// Strip <think>...</think> blocks from a completed string
function stripThinking(text) {
  text = text.replace(/<think>[\s\S]*?<\/think>/gi, '');
  text = text.replace(/<think>[\s\S]*/gi, '');
  return text.trimStart();
}

// Health check
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI → NVIDIA NIM Proxy',
    thinking: 'disabled',
    nim_base: NIM_API_BASE,
    api_key_set: !!NIM_API_KEY,
  });
});

// List models
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(id => ({
    id,
    object: 'model',
    created: 1700000000,
    owned_by: 'nvidia-nim-proxy',
  }));
  res.json({ object: 'list', data: models });
});

// Main proxy
app.post('/v1/chat/completions', async (req, res) => {
  if (!NIM_API_KEY) {
    return res.status(500).json({ error: { message: 'NIM_API_KEY not set', type: 'server_error' } });
  }

  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;
    const nimModel = MODEL_MAPPING[model] || model;
    const isNativeThinker = NATIVE_THINKERS.has(nimModel);

    const ts = () => new Date().toISOString();
    console.log(`\n${'─'.repeat(60)}`);
    console.log(`[${ts()}] REQUEST — ${model} → ${nimModel}`);
    console.log(`Stream: ${stream || false} | Max tokens: ${max_tokens || 16384}`);
    console.log(`Messages (${messages.length}):`);
    messages.forEach((msg, i) => {
      const preview = typeof msg.content === 'string'
        ? msg.content.slice(0, 300) + (msg.content.length > 300 ? '…' : '')
        : '[non-text content]';
      console.log(`  [${i}] ${msg.role.toUpperCase()}: ${preview}`);
    });

    const nimRequestBody = {
      model: nimModel,
      messages,
      temperature: temperature ?? 0.6,
      max_tokens: max_tokens || 16384,
      stream: stream || false,
    };

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

    // ── Streaming ────────────────────────────────────────────────────────────
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      res.setHeader('X-Accel-Buffering', 'no');

      console.log(`[${ts()}] RESPONSE (streaming)…`);
      let streamedContent = '';
      let buffer = '';
      let thinkBuffer = '';
      let inThink = false;

      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) continue;

          if (trimmed === 'data: [DONE]') {
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

            delete delta.reasoning_content;

            let content = delta.content ?? '';

            if (isNativeThinker && content) {
              thinkBuffer += content;

              if (!inThink && thinkBuffer.includes('<think>')) {
                inThink = true;
                const before = thinkBuffer.split('<think>')[0];
                thinkBuffer = '';
                content = before;
              } else if (inThink) {
                if (thinkBuffer.includes('</think>')) {
                  inThink = false;
                  const after = thinkBuffer.split('</think>').slice(1).join('</think>');
                  thinkBuffer = '';
                  content = after.trimStart();
                } else {
                  content = '';
                }
              } else {
                content = thinkBuffer;
                thinkBuffer = '';
              }
            }

            if (content) {
              streamedContent += content;
              delta.content = content;
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            } else if (!isNativeThinker) {
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            }

          } catch (e) {
            res.write(trimmed + '\n');
          }
        }
      });

      response.data.on('end', () => {
        const preview = streamedContent.slice(0, 500) + (streamedContent.length > 500 ? '…' : '');
        console.log(`[${ts()}] STREAM END`);
        console.log(`  ASSISTANT: ${preview}`);
        res.end();
      });

      response.data.on('error', (err) => {
        console.error('Stream error:', err.message);
        res.end();
      });

    // ── Non-streaming ────────────────────────────────────────────────────────
    } else {
      const choices = response.data.choices.map(choice => {
        const msg = choice.message ?? {};
        let content = msg.content ?? '';
        if (isNativeThinker) content = stripThinking(content);

        return {
          index: choice.index,
          message: { role: msg.role ?? 'assistant', content },
          finish_reason: choice.finish_reason,
        };
      });

      console.log(`[${ts()}] RESPONSE (non-stream)`);
      choices.forEach((c, i) => {
        const preview = c.message.content.slice(0, 300) + (c.message.content.length > 300 ? '…' : '');
        console.log(`  [${i}] ASSISTANT: ${preview}`);
      });
      console.log(`Usage: ${JSON.stringify(response.data.usage ?? {})}`);

      res.json({
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model,
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

// 404 fallback
app.all('*', (req, res) => {
  res.status(404).json({ error: { message: `Endpoint ${req.path} not supported`, type: 'not_found', code: 404 } });
});

app.listen(PORT, () => {
  console.log(`\n🚀 OpenAI → NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`   Health:      http://localhost:${PORT}/health`);
  console.log(`   Thinking:    ❌ DISABLED (stripped from all models)`);
  console.log(`   NIM API key: ${NIM_API_KEY ? '✅ SET' : '❌ NOT SET'}\n`);
});
