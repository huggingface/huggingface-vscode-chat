import * as assert from 'assert';
import * as vscode from 'vscode';
import { HuggingFaceChatModelProvider } from '../provider';

interface OpenAIToolCall { id: string; type: 'function'; function: { name: string; arguments: string } }
interface ConvertedMessage { role: 'user' | 'assistant' | 'tool'; content?: string; name?: string; tool_calls?: OpenAIToolCall[]; tool_call_id?: string }

suite('HuggingFaceChatModelProvider', () => {
  test('prepareLanguageModelChatInformation returns array (no key -> empty)', async () => {
    const provider = new HuggingFaceChatModelProvider({
      get: async () => undefined,
      store: async () => {},
      delete: async () => {},
      onDidChange: (_listener: unknown) => ({ dispose() {} })
    } as unknown as vscode.SecretStorage);

    const infos = await provider.prepareLanguageModelChatInformation({ silent: true }, new vscode.CancellationTokenSource().token);
    assert.ok(Array.isArray(infos));
  });

  test('provideTokenCount counts simple string', async () => {
    const provider = new HuggingFaceChatModelProvider({
      get: async () => undefined,
      store: async () => {},
      delete: async () => {},
      onDidChange: (_listener: unknown) => ({ dispose() {} })
    } as unknown as vscode.SecretStorage);

    const est = await provider.provideTokenCount({
      id: 'm', name: 'm', family: 'huggingface', version: '1.0.0',
      maxInputTokens: 1000, maxOutputTokens: 1000, capabilities: {}
    } as unknown as vscode.LanguageModelChatInformation, 'hello world', new vscode.CancellationTokenSource().token);
    assert.equal(typeof est, 'number');
    assert.ok(est > 0);
  });

  test('provideTokenCount counts message parts', async () => {
    const provider = new HuggingFaceChatModelProvider({
      get: async () => undefined,
      store: async () => {},
      delete: async () => {},
      onDidChange: (_listener: unknown) => ({ dispose() {} })
    } as unknown as vscode.SecretStorage);

    const msg: vscode.LanguageModelChatMessage = {
      role: vscode.LanguageModelChatMessageRole.User,
      content: [ new vscode.LanguageModelTextPart('hello world') ],
      name: undefined
    };
    const est = await provider.provideTokenCount({
      id: 'm', name: 'm', family: 'huggingface', version: '1.0.0',
      maxInputTokens: 1000, maxOutputTokens: 1000, capabilities: {}
    } as unknown as vscode.LanguageModelChatInformation, msg, new vscode.CancellationTokenSource().token);
    assert.equal(typeof est, 'number');
    assert.ok(est > 0);
  });

  test('convertMessages maps user/assistant text', () => {
    const provider = new HuggingFaceChatModelProvider({
      get: async () => undefined,
      store: async () => {},
      delete: async () => {},
      onDidChange: (_listener: unknown) => ({ dispose() {} })
    } as unknown as vscode.SecretStorage);

    const messages: vscode.LanguageModelChatMessage[] = [
      { role: vscode.LanguageModelChatMessageRole.User, content: [ new vscode.LanguageModelTextPart('hi') ], name: undefined },
      { role: vscode.LanguageModelChatMessageRole.Assistant, content: [ new vscode.LanguageModelTextPart('hello') ], name: undefined }
    ];
    const out = (provider as unknown as { convertMessages: (m: readonly vscode.LanguageModelChatMessage[]) => ConvertedMessage[] }).convertMessages(messages);
    assert.deepEqual(out, [
      { role: 'user', content: 'hi' },
      { role: 'assistant', content: 'hello' }
    ]);
  });

  test('convertMessages maps tool calls and results', () => {
    const provider = new HuggingFaceChatModelProvider({
      get: async () => undefined,
      store: async () => {},
      delete: async () => {},
      onDidChange: (_listener: unknown) => ({ dispose() {} })
    } as unknown as vscode.SecretStorage);

    const toolCall = new vscode.LanguageModelToolCallPart('abc', 'toolA', { foo: 1 });
    const toolResult = new vscode.LanguageModelToolResultPart('abc', [ new vscode.LanguageModelTextPart('result') ]);
    const messages: vscode.LanguageModelChatMessage[] = [
      { role: vscode.LanguageModelChatMessageRole.Assistant, content: [ toolCall ], name: undefined },
      { role: vscode.LanguageModelChatMessageRole.Assistant, content: [ toolResult ], name: undefined }
    ];
    const out = (provider as unknown as { convertMessages: (m: readonly vscode.LanguageModelChatMessage[]) => ConvertedMessage[] }).convertMessages(messages);
    const hasToolCalls = out.some((m: ConvertedMessage) => Array.isArray(m.tool_calls));
    const hasToolMsg = out.some((m: ConvertedMessage) => m.role === 'tool');
    assert.ok(hasToolCalls && hasToolMsg);
  });

  test('convertTools returns function tool definitions', () => {
    const provider = new HuggingFaceChatModelProvider({
      get: async () => undefined,
      store: async () => {},
      delete: async () => {},
      onDidChange: (_listener: unknown) => ({ dispose() {} })
    } as unknown as vscode.SecretStorage);

    const out = (provider as unknown as { convertTools: (o: vscode.LanguageModelChatRequestHandleOptions) => { tools?: Array<{ type: 'function'; function: { name: string; description?: string } }>; tool_choice?: 'auto' | 'required' } }).convertTools({
      tools: [
        {
          name: 'do_something',
          description: 'Does something',
          inputSchema: { type: 'object', properties: { x: { type: 'number' } }, additionalProperties: false }
        }
      ]
    } satisfies vscode.LanguageModelChatRequestHandleOptions);

    assert.ok(out);
    assert.equal(out.tool_choice, 'auto');
    assert.ok(Array.isArray(out.tools) && out.tools[0].type === 'function');
    assert.equal(out.tools[0].function.name, 'do_something');
  });

  test('provideLanguageModelChatResponse throws without API key', async () => {
    const provider = new HuggingFaceChatModelProvider({
      get: async () => undefined,
      store: async () => {},
      delete: async () => {},
      onDidChange: (_listener: unknown) => ({ dispose() {} })
    } as unknown as vscode.SecretStorage);

    let threw = false;
    try {
      await provider.provideLanguageModelChatResponse(
        { id: 'm', name: 'm', family: 'huggingface', version: '1.0.0', maxInputTokens: 1000, maxOutputTokens: 1000, capabilities: {} } as unknown as vscode.LanguageModelChatInformation,
        [],
        {} as unknown as vscode.LanguageModelChatRequestHandleOptions,
        { report: () => {} },
        new vscode.CancellationTokenSource().token
      );
    } catch {
      threw = true;
    }
    assert.ok(threw);
  });
});
