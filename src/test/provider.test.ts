import * as assert from 'assert';
import * as vscode from 'vscode';
import { HuggingFaceChatModelProvider } from '../provider';

suite('HuggingFaceChatModelProvider (integration)', () => {
  test('prepareLanguageModelChatInformation returns array (no key -> empty)', async () => {
    const provider = new HuggingFaceChatModelProvider({
      get: async () => undefined,
      store: async () => { },
      delete: async () => { },
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
});
