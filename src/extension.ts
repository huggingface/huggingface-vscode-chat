import * as vscode from 'vscode';
import { HuggingFaceChatModelProvider } from './provider';

export function activate(context: vscode.ExtensionContext) {
    const provider = new HuggingFaceChatModelProvider(context.secrets);
    // Register the Hugging Face provider under the vendor id used in package.json
    vscode.lm.registerLanguageModelChatProvider('huggingface', provider);

    // Management command to configure API key
    context.subscriptions.push(
        vscode.commands.registerCommand('huggingface.manage', async () => {
            const existing = await context.secrets.get('huggingface.apiKey');
            const apiKey = await vscode.window.showInputBox({
                title: 'Hugging Face API Key',
                prompt: existing ? 'Update your Hugging Face API key' : 'Enter your Hugging Face API key',
                ignoreFocusOut: true,
                password: true,
                value: existing ?? ''
            });
            if (apiKey === undefined) {
                return; // user canceled
            }
            if (!apiKey.trim()) {
                await context.secrets.delete('huggingface.apiKey');
                vscode.window.showInformationMessage('Hugging Face API key cleared.');
                return;
            }
            await context.secrets.store('huggingface.apiKey', apiKey.trim());
            vscode.window.showInformationMessage('Hugging Face API key saved.');
        })
    );
}

export function deactivate() { }
